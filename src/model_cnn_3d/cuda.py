import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

#Check availability for Intel GPU
print(torch.xpu.is_available())
