import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("VRAM", round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1), "GB")
