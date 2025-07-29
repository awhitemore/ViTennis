from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch

class TennisActionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Default transforms for ViT if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to 224x224 for ViT
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        else:
            self.transform = transform
            
        self.data = ImageFolder(data_dir, transform=self.transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return {"pixel_values": image, "labels": label}  # Return dict for Hugging Face Trainer

    def classes(self):
        return self.data.classes

    
    
