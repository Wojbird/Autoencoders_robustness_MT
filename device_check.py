import torch


def get_available_gpus():
    """
    Returns a list of dicts with available CUDA devices.
    Example:
    [
        {
            "id": 0,
            "name": "NVIDIA A100-SXM4-80GB",
            "total_memory_gb": 80.0,
            "capability": "8.0"
        },
        ...
    ]
    """
    gpus = []

    if not torch.cuda.is_available():
        return gpus

    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        gpus.append({
            "id": idx,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "capability": f"{props.major}.{props.minor}",
        })

    return gpus


def print_available_gpus():
    if not torch.cuda.is_available():
        print("CUDA is available: False")
        print("No GPU devices found.")
        return

    print("CUDA is available: True")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")

    for gpu in get_available_gpus():
        print(
            f"[{gpu['id']}] {gpu['name']} | "
            f"{gpu['total_memory_gb']} GB | "
            f"compute capability {gpu['capability']}"
        )


if __name__ == "__main__":
    print_available_gpus()