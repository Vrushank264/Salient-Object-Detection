import torch
import os
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from natsort import natsorted


class Data(Dataset):
    
    def __init__(self, img_dir, label_dir, mode = 'train'):
        
        super().__init__()
        self.img_fnames = [os.path.join(img_dir, x) for x in natsorted(os.listdir(img_dir))]
        self.label_fnames = [os.path.join(label_dir, y) for y in natsorted(os.listdir(label_dir))]
        
        if mode == 'train':
            
            self.transform = T.Compose([T.Resize((320, 320)),
                                        T.RandomCrop((288,288)),
                                        T.ToTensor()])
        elif mode == 'test':
            
            self.transform = T.Compose([T.Resize((320, 320)),
                                        T.ToTensor()])
        
        
    def __len__(self):
        
        return len(self.img_fnames)
    
    
    def __getitem__(self, idx):
        
        img = Image.open(self.img_fnames[idx]).convert('RGB')
        mask = Image.open(self.label_fnames[idx]).convert('L')
        img = self.transform(img)
        mask = self.transform(mask)
        
        return img, mask