from unittest.mock import patch

import pytest

from helpers.Gpu import GpuInfo, ResolveGpuIds


def test_auto_gpu_filter_is_strict():
    snapshot = [
        GpuInfo(0, "gpu", 100, 29, 0.29, 4),
        GpuInfo(1, "gpu", 100, 30, 0.30, 4),
        GpuInfo(2, "gpu", 100, 29, 0.29, 5),
    ]
    with patch("helpers.Gpu.QueryGpus", return_value=snapshot):
        selected, returned = ResolveGpuIds("auto")
    assert selected == [0]
    assert returned == snapshot


def test_explicit_gpu_pool_is_validated():
    snapshot = [GpuInfo(0, "gpu", 100, 0, 0.0, 0)]
    with patch("helpers.Gpu.QueryGpus", return_value=snapshot):
        with pytest.raises(ValueError):
            ResolveGpuIds([0, 0])
        with pytest.raises(ValueError):
            ResolveGpuIds([1])
