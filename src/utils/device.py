import torch


def get_device():
    """
    Returns the best available device.
    Priority:
        CUDA GPU
        Apple MPS
        CPU
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
        return device

    else:
        device = torch.device("cpu")
        print("Using CPU")
        return device