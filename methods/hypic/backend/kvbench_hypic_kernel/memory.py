import torch


def weak_ref_tensor(tensor):
    return (
        torch.ops.kvbench_hypic_kernel.weak_ref_tensor(tensor)
        if isinstance(tensor, torch.Tensor)
        else tensor
    )
