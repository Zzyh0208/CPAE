"""Per-process PyTorch allocator measurements on the selected GPU."""
import torch


def reset_peak_memory(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device):
    device = torch.device(device)
    if device.type != 'cuda':
        return {'device_count': 0, 'scope': 'pytorch_allocator', 'devices': []}
    torch.cuda.synchronize(device)
    return {'device_count': 1, 'scope': 'pytorch_allocator', 'devices': [{
        'device': str(device), 'name': torch.cuda.get_device_name(device),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}]}
