# import resources
import numpy as np
import torch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from constants import *

np.random.seed(SEED)
torch.manual_seed(SEED)

transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),  
    transforms.Grayscale(num_output_channels=NUM_CHANNELS),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5], std=[0.5]),  
])

# Load dataset
full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

train_size = int(SPLIT_SIZE * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
  
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


