
import torch
import torchvision
import flwr
import numpy as np
import pandas as pd
import sklearn
import cv2
from PIL import Image

print("========== FedLung-XAI ==========")
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Flower:", flwr.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("OpenCV:", cv2.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("=================================")
print("Setup completed successfully!")