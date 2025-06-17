from torch.utils.data import Dataset
from PIL import Image
import random

class Augmented(Dataset):
    def __init__(self, dataset, transform, N = 8):
        super().__init__()
        self.dataset = dataset
        self.len = len(self.dataset)
        self.transform = transform
        self.N = N

    def __len__(self):
        return self.len*self.N
    
    def __getitem__(self, index):
        img , _ = self.dataset[index%self.len]
        return self.transform(img), index
