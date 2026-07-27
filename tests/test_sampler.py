import pytest
import torch.distributed as dist

from eval.sampler import CustomDistributedSampler


def test_defaults_to_single_replica_without_process_group():
    """Single-process eval runs never call init_process_group, so the sampler must not require it."""
    assert not dist.is_initialized()

    sampler = CustomDistributedSampler(range(5), shuffle=False)

    assert sampler.num_replicas == 1
    assert sampler.rank == 0
    assert sampler.num_samples == 5
    assert list(sampler) == [0, 1, 2, 3, 4]


def test_splits_uneven_dataset_without_padding():
    samplers = [
        CustomDistributedSampler(range(5), num_replicas=2, rank=rank, shuffle=False)
        for rank in (0, 1)
    ]

    assert [list(sampler) for sampler in samplers] == [[0, 2, 4], [1, 3]]
    assert [sampler.num_samples for sampler in samplers] == [3, 2]


def test_rejects_rank_outside_world():
    with pytest.raises(ValueError):
        CustomDistributedSampler(range(5), num_replicas=2, rank=2)
