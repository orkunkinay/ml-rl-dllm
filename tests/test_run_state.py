import json
import signal
from dataclasses import dataclass
from pathlib import Path

import torch

from common.run_state import ClusterStateCallback
from common.run_state import append_jsonl
from common.run_state import atomic_torch_save
from common.run_state import checkpoint_metadata
from common.run_state import completed_jsonl_sample_ids
from common.run_state import jsonl_is_nonempty_parseable
from common.run_state import read_json
from common.run_state import resolve_resume_checkpoint
from common.run_state import write_progress


@dataclass
class DummyConfig:
    alpha_compute_reward: float = 0.3
    seed: int = 7


class DummyState:
    global_step = 12
    epoch = 1
    max_steps = 20
    is_world_process_zero = True

    def save_to_json(self, path):
        Path(path).write_text(json.dumps({"global_step": self.global_step}))


class DummyTrainer:
    def __init__(self, tmp_path):
        self.state = DummyState()
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.lr_scheduler = None
        self.scaler = None
        self.tmp_path = tmp_path

    def save_model(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")


def test_checkpoint_save_load_roundtrip(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    payload = checkpoint_metadata(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        config=DummyConfig(),
        state=DummyState(),
        latest_metrics={"loss": 0.5},
        hf_checkpoint_path=tmp_path / "checkpoint-12",
    )

    path = tmp_path / "checkpoint_latest.pt"
    atomic_torch_save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    assert loaded["global_step"] == 12
    assert loaded["alpha"] == 0.3
    assert "policy_model_state_dict" in loaded
    assert "optimizer_state_dict" in loaded


def test_resume_restores_model_optimizer_and_step(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for param in model.parameters():
        param.data.fill_(2.0)

    payload = checkpoint_metadata(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        config=DummyConfig(),
        state=DummyState(),
        latest_metrics={},
        hf_checkpoint_path=tmp_path / "checkpoint-12",
    )
    path = tmp_path / "checkpoint_latest.pt"
    atomic_torch_save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    restored = torch.nn.Linear(2, 1)
    restored_optim = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored.load_state_dict(loaded["policy_model_state_dict"])
    restored_optim.load_state_dict(loaded["optimizer_state_dict"])

    assert loaded["global_step"] == 12
    assert all(torch.equal(p, torch.full_like(p, 2.0)) for p in restored.parameters())
    assert restored_optim.state_dict()["param_groups"][0]["lr"] == 1e-3


def test_progress_json_is_written(tmp_path):
    write_progress(
        tmp_path,
        "running",
        alpha=0.3,
        seed=7,
        global_step=5,
        total_steps=10,
        completed_fraction=0.5,
        latest_metrics={"reward": 0.2},
    )
    progress = read_json(tmp_path / "progress.json")
    assert progress["status"] == "running"
    assert progress["completed_fraction"] == 0.5
    assert progress["latest_metrics"]["reward"] == 0.2


def test_evaluation_jsonl_skip_logic_does_not_duplicate_completed_samples(tmp_path):
    path = tmp_path / "eval.jsonl"
    append_jsonl(path, {"sample_id": "gsm8k:1", "prediction": "a"})
    append_jsonl(path, {"sample_id": "gsm8k:2", "prediction": "b"})

    completed = completed_jsonl_sample_ids(path)
    for sample_id in ["gsm8k:1", "gsm8k:2", "gsm8k:3"]:
        if sample_id not in completed:
            append_jsonl(path, {"sample_id": sample_id, "prediction": "new"})

    rows = list(path.read_text().strip().splitlines())
    assert len(rows) == 3
    assert completed_jsonl_sample_ids(path) == {"gsm8k:1", "gsm8k:2", "gsm8k:3"}
    assert jsonl_is_nonempty_parseable(path)


def test_sigterm_handler_can_save_emergency_checkpoint(tmp_path):
    callback = ClusterStateCallback(tmp_path, DummyConfig())
    trainer = DummyTrainer(tmp_path)
    emergency = callback.save_emergency_checkpoint(trainer, signum=signal.SIGTERM)

    assert emergency.exists()
    assert (tmp_path / "checkpoints" / "checkpoint_latest.pt").exists()
    assert read_json(tmp_path / "progress.json")["status"] == "interrupted"


def test_resolve_resume_checkpoint_from_sidecar(tmp_path):
    hf_checkpoint = tmp_path / "checkpoint-12"
    hf_checkpoint.mkdir()
    sidecar = tmp_path / "checkpoints" / "checkpoint_latest.pt"
    atomic_torch_save({"hf_checkpoint_path": str(hf_checkpoint)}, sidecar)

    resolved = resolve_resume_checkpoint(str(sidecar), tmp_path)
    assert resolved == hf_checkpoint


def test_disable_tqdm_flag_is_available_in_eval_source():
    source = Path("eval/eval.py").read_text()
    assert "--disable_tqdm" in source
    assert "disable_tqdm=args.disable_tqdm" in source
