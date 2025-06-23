from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import glob
import random
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict
from utils import *

def get_frame(path):
    frame = Image.open(path)
    return frame

class FAS_Dataset_AttackTypes(Dataset):
    def __init__(self, root, protocol=['O', 'C', 'I', 'M', 'W'], train=True, size=224, k_shot=None):
        self.size = size
        self.train = train
        self.protocol = protocol
        self.k_shot = k_shot

        # Define attack type labels with explicit type numbers
        self.attack_types = {
            'real': 0,
            'print': 1,
            'replay': 2,
            'mask': 3,
            'partial': 4,
            'fakehead': 5
        }

        data_by_label = defaultdict(list)
        protocol_map = {'O': 'Oulu', 'C': 'casia', 'I': 'replay', 'M': 'MSU'}

        for proto in protocol:
            if proto == 'W':  # WMCA dataset
                for attack_type, label in self.attack_types.items():
                    rgb_file = os.path.join(root, f"WMCA_split/rgb/{attack_type}.npy")
                    if os.path.exists(rgb_file):
                        data_by_label[label].extend(np.load(rgb_file))
            
            elif proto in protocol_map:  # O, C, I, M datasets
                prefix = protocol_map[proto]
                # These datasets only contain 'real', 'print', and 'replay' attacks
                for attack_type in list(self.attack_types.keys())[:3]:
                    label = self.attack_types[attack_type]
                    if attack_type == 'real':
                        rgb_file = os.path.join(root, f"domain-generalization/{prefix}_images_live.npy")
                    else:
                        rgb_file = os.path.join(root, f"domain-generalization/{prefix}_{attack_type}_images.npy")
                    
                    if os.path.exists(rgb_file):
                        data_by_label[label].extend(np.load(rgb_file))

        self.data_rgb = []
        self.labels = []

        if self.k_shot is not None:
            # Use all available attack types that have enough samples
            for label, data in data_by_label.items():
                if len(data) >= self.k_shot:
                    sampled_data = random.sample(data, self.k_shot)
                    self.data_rgb.extend(sampled_data)
                    self.labels.extend([label] * self.k_shot)
                # Note: Attack types with fewer than k_shot samples are silently ignored.
        
        else:
            # Original behavior: Load all data
            for label, data in data_by_label.items():
                if data:
                    self.data_rgb.extend(data)
                    self.labels.extend([label] * len(data))

        # Convert labels to numpy array
        self.labels = np.array(self.labels, dtype=np.int64)

    def transform(self, img, train=True, size=224):
        if train:  # Training
            img = TF.center_crop(TF.resize(img, (256, 256)), (size, size))

            if random.random() > 0.5:
                img = TF.hflip(img)

            if random.random() > 0.5:
                img = TF.vflip(img)

            angle = transforms.RandomRotation.get_params(degrees=(-30, 30))
            img = TF.rotate(img, angle)
        else:  # Testing
            img = TF.resize(img, (size, size))

        img = TF.to_tensor(img)

        return img

    def __getitem__(self, idx):
        rgb = self.data_rgb[idx]
        label = self.labels[idx]

        if 'O' in self.protocol or 'C' in self.protocol or 'I' in self.protocol or 'M' in self.protocol:
            # Convert numpy array to PIL Image for OULU-NPU, CASIA-FASD, Idiap-Replay, and MSU-MFSD
            rgb = normalize_data(rgb)
            rgb = (rgb * 255).astype(np.uint8)

        # Convert numpy arrays to PIL Images for transformation
        rgb_img = Image.fromarray(rgb)

        rgb_img = self.transform(rgb_img, self.train, self.size)

        return rgb_img, label

    def __len__(self):
        return len(self.data_rgb)

def get_attack_type_loader(root, protocol=['W'], batch_size=10, shuffle=True, train=True, size=224):
    dataset = FAS_Dataset_AttackTypes(root=root, protocol=protocol, train=train, size=size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

if __name__ == "__main__":
    train_loader = get_attack_type_loader(root='/shared/shared', protocol=['W'], batch_size=1800, shuffle=True)

    total = 0
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # Count for each attack type
    for i, (rgb_img, labels) in enumerate(train_loader):
        print(rgb_img.shape)
        total += rgb_img.shape[0]
        for label in labels:
            label_counts[int(label)] += 1

    print("Label counts:", label_counts)
    print("Total samples:", total)