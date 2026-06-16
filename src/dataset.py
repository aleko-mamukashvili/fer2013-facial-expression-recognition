import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class FERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.pixels = df["pixels"].tolist()
        self.labels = df["emotion"].tolist() if "emotion" in df.columns else None
        self.transform = transform

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        img = np.array(self.pixels[idx].split(), dtype=np.float32).reshape(48, 48)
        img = img / 255.0
        img = torch.tensor(img).unsqueeze(0)
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img

def get_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])
