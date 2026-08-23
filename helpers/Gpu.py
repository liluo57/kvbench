"""GPU discovery helpers for the benchmark scheduler."""

from dataclasses import asdict, dataclass
from typing import Iterable, List, Tuple, Union


@dataclass(frozen=True)
class GpuInfo:
    id: int
    name: str
    memoryTotal: int
    memoryUsed: int
    memoryRatio: float
    utilization: int
    computePids: Tuple[int, ...] = ()

    def AsDict(self) -> dict:
        return asdict(self)


def QueryGpus() -> List[GpuInfo]:
    """Return a point-in-time NVML snapshot without initializing CUDA."""
    try:
        import pynvml
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "GPU discovery requires nvidia-ml-py (import name: pynvml)"
        ) from exc

    infos: List[GpuInfo] = []
    try:
        pynvml.nvmlInit()
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                computePids = tuple(sorted({int(process.pid) for process in processes}))
            except Exception:  # noqa: BLE001 - unsupported on some NVML stacks
                computePids = ()
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            infos.append(
                GpuInfo(
                    id=index,
                    name=str(name),
                    memoryTotal=int(memory.total),
                    memoryUsed=int(memory.used),
                    memoryRatio=(float(memory.used) / float(memory.total)),
                    utilization=int(utilization.gpu),
                    computePids=computePids,
                )
            )
    except Exception as exc:  # noqa: BLE001 - NVML wraps driver errors
        raise RuntimeError(f"unable to query GPUs through NVML: {exc}") from exc
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
    return infos


def ResolveGpuIds(
    availableGpuIds: Union[str, Iterable[int]],
    *,
    maxMemoryRatio: float = 0.30,
    maxUtilization: int = 5,
) -> tuple[List[int], List[GpuInfo]]:
    """Resolve an explicit GPU pool or the strict ``auto`` free-GPU filter."""
    snapshot = QueryGpus()
    known = {gpu.id for gpu in snapshot}
    if isinstance(availableGpuIds, str):
        if availableGpuIds != "auto":
            raise ValueError("availableGpuIds must be 'auto' or an iterable of ids")
        selected = [
            gpu.id
            for gpu in snapshot
            if gpu.memoryRatio < maxMemoryRatio
            and gpu.utilization < maxUtilization
        ]
        return selected, snapshot

    selected = [int(gpu) for gpu in availableGpuIds]
    if len(set(selected)) != len(selected):
        raise ValueError(f"availableGpuIds contains duplicates: {selected}")
    unknown = [gpu for gpu in selected if gpu not in known]
    if unknown:
        raise ValueError(f"unknown GPU id(s): {unknown}; installed ids={sorted(known)}")
    return selected, snapshot
