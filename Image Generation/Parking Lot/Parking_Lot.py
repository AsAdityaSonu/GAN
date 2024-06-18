# Project Information
Project_Name = "Parking Lot Detection"
Project_Path = "Image Generation/Parking_Lot.py"
Project_Description = "Parking Lot Image Generation using GAN."
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

DATA_DIR = "Image Generation/Parking Lot/archive/"

# Print names of the top 10 files
print(os.listdir(DATA_DIR)[:10])
# ---------------------------------------------------------------------------------
image_size = 64
batch_size = 128

# the mean and std deviation for normalization
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# Create the dataset and dataloader
train_ds = ImageFolder(DATA_DIR, transform=T.Compose([
    T.Resize(image_size),             # Resize the images to the specified size
    T.CenterCrop(image_size),         # Center crop the images
    T.ToTensor(),                     # Convert images to PyTorch tensors
    T.Normalize(*stats)               # Normalize the images
]))

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)

# Function to denormalize the images
def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

# Function to show images in a grid
def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(images[:nmax], nrow=8).permute(1, 2, 0).clamp(0, 1))

# Function to show a batch of images from the dataloader
def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(denorm(images), nmax)
        break

if __name__ == '__main__':
    show_batch(train_dl)
