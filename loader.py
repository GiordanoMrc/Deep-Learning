import torch.nn as nn
import torch
from torchvision import datasets, models, transforms
import os

def get_data_transform(input_size):
   data_transforms = {
      'train': transforms.Compose([
         transforms.RandomResizedCrop(input_size),
         #transforms.RandomResizedCrop(input_size),
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
         transforms.Resize(input_size),
         transforms.CenterCrop(input_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),

   }
   return data_transforms

def dataset(data_dir,input_size):
   # Create training and validation datasets
   image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), get_data_transform(input_size)[x]) for x in ['train', 'val']}
   return image_datasets

def dataloader(data_dir,dataset,batch):   
   # Create training and validation dataloaders
   dataloaders_dict = {x: torch.utils.data.DataLoader(dataset[x], batch_size=batch, shuffle=True, num_workers=0) for x in ['train', 'val']}
   return dataloaders_dict


