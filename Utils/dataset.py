import cv2
import os
import torch
from torch.utils.data import Dataset

class SyentheticDataset(Dataset):
  def __init__(self, image_dir, label_dir):
    self.image_dir = image_dir
    self.label_dir = label_dir
    self.image_files = sorted(os.listdir(image_dir))
    self.label_files = sorted(os.listdir(label_dir))
  
  def __len__(self):
    return len(self.image_files)
  
  def __getitem__(self, index):
    image_path = os.path.join(self.image_dir, self.image_files[index])
    label_path = os.path.join(self.label_dir, self.label_files[index])
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Read labels and draw boxes
    with open(label_path, 'r') as f:
        for line in f:
            label = line.strip().split()

    return (image, label)