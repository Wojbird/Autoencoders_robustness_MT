import torch

print("CUDA is available: " + str(torch.cuda.is_available()))  # cuda: True
print("CUDA version: " + torch.version.cuda) # cuda version
print("Device name: " + torch.cuda.get_device_name(0))  # device name