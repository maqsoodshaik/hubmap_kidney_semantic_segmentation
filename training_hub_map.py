

from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import cv2

import torch
from utils import get_cartesian_coords
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from keras.metrics import MeanIoU
from sklearn import metrics
from dataset import CustomDataset
#import wandb and log loss and accuracy
import wandb
wandb.init(project="hubmap")
classes_dict = {
    "blood_vessel": 2,
    "glomerulus": 1,
    "unsure": 0,
}



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            nn.MaxPool2d(2),
            DoubleConv(256, 512),
            nn.MaxPool2d(2),
            DoubleConv(512, 1024),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            DoubleConv(128, 64),
        )
        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x5 = self.encoder[4](x4)
        x6 = self.encoder[5](x5)
        x7 = self.encoder[6](x6)
        x8 = self.encoder[7](x7)
        x9 = self.encoder[8](x8)

        x = self.decoder[0](x9)
        x = self.decoder[1](torch.cat([x, x7], dim=1))
        x = self.decoder[2](x)
        x = self.decoder[3](torch.cat([x, x5], dim=1))
        x = self.decoder[4](x)
        x = self.decoder[5](torch.cat([x, x3], dim=1))
        x = self.decoder[6](x)
        x = self.decoder[7](torch.cat([x, x1], dim=1))
        return self.output(x)
    
def train():

    # Usage:
    in_channels = 3  # Assuming grayscale images
    out_channels = 3  # Number of output channels for segmentation mask (binary in this case)
    model = UNet(in_channels, out_channels)

    # Assuming you have a custom dataset class (YourDatasetClass) for training data
    # Replace YourDatasetClass and other dataset-related parameters with your actual dataset
    dataset = CustomDataset(json_file="/content/drive/MyDrive/hubmap-hacking-the-human-vasculature/polygons.jsonl",
        data_path="/content/drive/MyDrive/hubmap-hacking-the-human-vasculature/train",
        class_names=classes_dict,
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for training and validation sets
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Set device for training (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    IOU_keras = MeanIoU(num_classes=3)   
    # Training loop
    num_epochs = 50
    iou_best = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_id , batch in enumerate(train_loader):
            images, masks = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, masks)
            train_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            if batch_id % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_id+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                #log using wandb
                wandb.log({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
        model.eval()
        val_loss = 0.0
        iou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                prediction = torch.argmax(outputs, dim=1)
                
                loss = criterion(outputs, masks)
                
                IOU_keras.update_state(masks.detach().cpu().numpy(), prediction.detach().cpu().numpy())
                print(IOU_keras.result().numpy())
                iou += IOU_keras.result().numpy()
                val_loss += loss.item()
        avg_iou = iou/len(val_loader)
        
        # accuracy = accuracy/len(val_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"validation_loss": avg_val_loss})
        wandb.log({"iou_validation":avg_iou})
        print(f"Validation IOU: {avg_iou:.4f}")
        if avg_iou > iou_best:
            torch.save(model.state_dict(), "unet_model.pth")
            iou_best = max(iou_best, avg_iou)

    # Validation loop (optional)
    model.eval()
    val_loss = 0.0
    iou = 0.0
    accuracy = 0.0
    IOU_keras = MeanIoU(num_classes=3)      
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            prediction = torch.argmax(outputs, dim=1)
            
            loss = criterion(outputs, masks)
            
            IOU_keras.update_state(masks, prediction)
            print(IOU_keras.result().numpy())
            iou += IOU_keras.result().numpy()
            val_loss += loss.item()
    avg_iou = iou/len(val_loader)
    
    # accuracy = accuracy/len(val_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    train()
