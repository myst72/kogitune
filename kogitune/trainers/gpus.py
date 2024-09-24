import torch

def get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1

def bf16_is_available():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            if 'A100' in gpu_name or 'H100' in gpu_name:
                return True
    return False

def print_gpu_utilization():
    try:
        pynvml = adhoc.safe_import('pynvml', 'nvidia-ml-py')

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {adhoc.format_unit(info.used, scale=1024)}iB.")
    except:
        pass


def print_summary(result, use_flash=False):
    m = result.metrics
    print(
        f"Time: {m['train_runtime']:.2f}  {adhoc.format_unit(m['train_runtime'], scale=60)}",
        end=" ",
    )
    print(f"Samples/second: {m['train_samples_per_second']:.2f} FlashAttn={use_flash}")
    print(
        f"Global step: {result.global_step} batch_size: {1024//result.global_step}",
        end=" ",
    )
    if "total_flos" in m:
        print(
            f"FLOS: {m['total_flos']} {adhoc.format_unit(m['total_flos'])} Loss: {m['train_loss']:.5f}"
        )
    else:
        print(f"Loss: {m['train_loss']:.5f}")
    print_gpu_utilization()
