import torch

def print_cuda_memory(msg, device='cuda'):
    mem = torch.cuda.memory_allocated(device=device)
    # max_mem = torch.cuda.max_memory_allocated(device=device)
    print(f'{msg}: Cuda memory allocated: {mem} byte {int(mem) // (1024 * 1024)} MB')
    # print(f'{msg}: Max cuda memory allocated: {max_mem} byte {int(max_mem) // (1024 * 1024)} MB')