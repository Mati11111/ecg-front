import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Todos los imports funcionaron correctamente.")
print(f"Versión de Python: {pd.__version__}, NumPy: {np.__version__}")
print(f"Torch: {torch.__version__}, Scikit-learn: {StandardScaler.__module__.split('.')[0]}")
print(f"Matplotlib: {plt.__version__}, Seaborn: {sns.__version__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo PyTorch disponible: {device}")
