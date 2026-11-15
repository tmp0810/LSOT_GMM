"""Device selection and synchronized wall-clock timing for experiments."""
import torch


def resolve_device(value):
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError('Use --device cpu, cuda, or cuda:N') from exc
    if device.type == 'cpu' and device.index is None:
        return device
    if device.type != 'cuda':
        raise ValueError('Use --device cpu, cuda, or cuda:N')
    if not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable to this PyTorch installation')
    index = torch.cuda.current_device() if device.index is None else device.index
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f'CUDA device {index} is not visible in this process')
    return torch.device('cuda', index)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
