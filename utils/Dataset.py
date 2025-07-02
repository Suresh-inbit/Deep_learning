from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision
import numpy as np
class Augmented_0(Dataset):
    def __init__(self, dataset, transform, N = 2):
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
#for cropping a segment
class Augmented(Dataset):
    def __init__(self, dataset, transform, N = 2):
        super().__init__()
        self.dataset = dataset
        self.len = len(self.dataset)
        self.transform = transform
        self.N = N

    def __len__(self):
        return self.len*self.N
    
    def __getitem__(self, index):
        img, _ = self.dataset[index % self.len]
        img = self.transform(img)  # shape should be [C, 1024, 1024]

        # Randomly choose one of the four 512x512 quadrants
        quadrant = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
        x_idx, y_idx = quadrant
        offsetx, offsety=  -quadrant[0]*100+50, -quadrant[1]*100+50
        # Crop the corresponding quadrant
        h_start = x_idx * 512 + offsetx
        h_end = (x_idx + 1) * 512 + offsetx
        w_start = y_idx * 512 + offsety
        w_end = (y_idx + 1) * 512 +offsety

        img = img[:, h_start:h_end, w_start:w_end]  # assuming C x H x W format
        img = torchvision.transforms.Resize((128,128))(img)
        return img, index
