import ast
import json
import signal
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import common.run_state as run_state
from common.memory import format_cuda_memory
from common.memory import get_cuda_memory_stats
from common.memory import log_cuda_memory
from common.run_state import ClusterStateCallback
from common.run_state import append_jsonl
from common.run_state import atomic_torch_save
from common.run_state import checkpoint_metadata
from common.run_state import completed_jsonl_sample_ids
from common.run_state import jsonl_is_nonempty_parseable
from common.run_state import prepare_local_run_dir
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
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda _: 1.0
        )
        self.scaler = None
        self.tmp_path = tmp_path

    def save_model(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file(self.model.state_dict(), path / "model.safetensors")


def write_valid_hf_checkpoint(path: Path, *, value: float = 1.0) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    save_file({"weight": torch.tensor([value])}, path / "model.safetensors")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": value}))
    return path


def write_corrupt_hf_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_text("not a safetensors file")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": 999}))
    return path


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
    emergency_dir = tmp_path / "checkpoint-emergency-12"
    assert (emergency_dir / "model.safetensors").exists()
    assert (emergency_dir / "trainer_state.json").exists()
    assert (emergency_dir / "optimizer.pt").exists()
    assert (emergency_dir / "scheduler.pt").exists()
    assert (tmp_path / "checkpoints" / "checkpoint_latest.pt").exists()
    assert read_json(tmp_path / "progress.json")["status"] == "interrupted"
    assert read_json(tmp_path / "progress.json")["signal"] == signal.SIGTERM


def test_auto_resume_uses_signal_emergency_checkpoint_from_latest_sidecar(tmp_path):
    write_valid_hf_checkpoint(tmp_path / "checkpoint-10", value=10)
    callback = ClusterStateCallback(tmp_path, DummyConfig())
    trainer = DummyTrainer(tmp_path)

    callback.save_emergency_checkpoint(trainer, signum=signal.SIGTERM)

    resolved = resolve_resume_checkpoint("auto", tmp_path)
    assert resolved == (tmp_path / "checkpoint-emergency-12").resolve()


def test_auto_resume_skips_corrupt_newer_checkpoint_and_uses_latest_valid(tmp_path):
    write_valid_hf_checkpoint(tmp_path / "checkpoint-10", value=10)
    write_corrupt_hf_checkpoint(tmp_path / "checkpoint-20")

    with pytest.warns(UserWarning, match="Skipping unreadable checkpoint"):
        resolved = resolve_resume_checkpoint("auto", tmp_path)

    assert resolved == (tmp_path / "checkpoint-10").resolve()


def test_auto_resume_prefers_newer_numeric_checkpoint_over_older_sidecar(tmp_path):
    checkpoint_10 = write_valid_hf_checkpoint(tmp_path / "checkpoint-10", value=10)
    checkpoint_20 = write_valid_hf_checkpoint(tmp_path / "checkpoint-20", value=20)
    sidecar = tmp_path / "checkpoints" / "checkpoint_latest.pt"
    atomic_torch_save({"global_step": 10, "hf_checkpoint_path": str(checkpoint_10)}, sidecar)

    resolved = resolve_resume_checkpoint("auto", tmp_path)
    assert resolved == checkpoint_20.resolve()


def test_sidecar_torch_save_failure_warns_and_does_not_stop_training(
    tmp_path, monkeypatch
):
    callback = ClusterStateCallback(tmp_path, DummyConfig())
    trainer = DummyTrainer(tmp_path)

    def fail_sidecar_save(obj, path):
        raise RuntimeError(
            "[enforce fail at inline_container.cc:815] . "
            "PytorchStreamWriter failed writing file data/0: file write failed"
        )

    monkeypatch.setattr(run_state, "atomic_torch_save", fail_sidecar_save)

    with pytest.warns(UserWarning, match="Failed to write sidecar checkpoint"):
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path)),
            trainer.state,
            None,
            model=trainer.model,
            optimizer=trainer.optimizer,
            lr_scheduler=trainer.lr_scheduler,
        )

    progress = read_json(tmp_path / "progress.json")
    assert progress["status"] == "running"
    assert "PytorchStreamWriter failed writing file data/0" in progress[
        "sidecar_checkpoint_error"
    ]
    assert "sidecar_checkpoint_error" in progress["latest_metrics"]


def test_resolve_resume_checkpoint_from_sidecar(tmp_path):
    hf_checkpoint = tmp_path / "checkpoint-12"
    write_valid_hf_checkpoint(hf_checkpoint, value=12)
    sidecar = tmp_path / "checkpoints" / "checkpoint_latest.pt"
    atomic_torch_save({"hf_checkpoint_path": str(hf_checkpoint)}, sidecar)

    resolved = resolve_resume_checkpoint(str(sidecar), tmp_path)
    assert resolved == hf_checkpoint


def test_prepare_local_run_dir_requires_resume_or_overwrite_for_existing_run(tmp_path):
    run_dir = prepare_local_run_dir(
        DummyConfig(), resume=None, run_root=tmp_path, run_name="existing"
    )
    sentinel = run_dir / "progress.json"
    sentinel.write_text("{}")

    with pytest.raises(FileExistsError):
        prepare_local_run_dir(
            DummyConfig(), resume=None, run_root=tmp_path, run_name="existing"
        )

    resumed = prepare_local_run_dir(
        DummyConfig(), resume="auto", run_root=tmp_path, run_name="existing"
    )
    assert resumed == run_dir
    assert sentinel.exists()


def test_prepare_local_run_dir_overwrite_is_destructive_only_when_explicit(tmp_path):
    run_dir = prepare_local_run_dir(
        DummyConfig(), resume=None, run_root=tmp_path, run_name="overwrite"
    )
    sentinel = run_dir / "progress.json"
    sentinel.write_text("{}")

    prepared = prepare_local_run_dir(
        DummyConfig(),
        resume=None,
        overwrite=True,
        run_root=tmp_path,
        run_name="overwrite",
    )

    assert prepared == run_dir
    assert not sentinel.exists()


def test_disable_tqdm_flag_is_available_in_eval_signature():
    source = (Path(__file__).parents[1] / "eval" / "eval.py").read_text()
    module = ast.parse(source)
    evaluate_fn = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate"
    )
    arg_names = [arg.arg for arg in evaluate_fn.args.args]
    assert "disable_tqdm" in arg_names


def test_cuda_memory_helpers_are_safe_without_cuda():
    stats = get_cuda_memory_stats()
    assert "cuda_available" in stats
    assert "allocated_gb" in stats
    formatted = format_cuda_memory("test")
    assert "test" in formatted
    returned = log_cuda_memory("test")
    assert returned["cuda_available"] == stats["cuda_available"]
