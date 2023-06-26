import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
import cv2
from torchvision.transforms import functional as F

class FishDataset(Dataset):
    def __init__(self, path):
        self.imglist = sorted(glob.glob(path + '*.png'))
    def __len__(self):
        return len(self.imglist)
    
    def __getitem__(self, index):
        image = cv2.imread(self.imglist[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        image = image / 255.0
        image_tensor = F.to_tensor(image)

        # print(self.imglist[index])

        with open(self.imglist[index][:-4] + '.txt', 'r') as f:
            targets = []
            lines = f.readlines()
            max_objects = 100
            num_objects = min(len(lines), max_objects)
            
            for i in range(max_objects):
                if i < num_objects:
                    line = lines[i]
                    line = line.strip().split(' ')
                    label = int(line[0])
                    bbox = [float(x) for x in line[1:]]
                    target = {
                        'boxes': bbox,
                        'labels': label
                    }
                else:
                    target = {
                        'boxes': [0.0, 0.0, 0.0, 0.0],
                        'labels': 0
                    }
                
                targets.append(target)

        print(targets)

        return image_tensor, targets
    

data = FishDataset("/home/student/Desktop/class/class_master1_bot/interaction/Final/FinalProj-FishDetection/TraingData/").__getitem__(0)
print(data[1])
