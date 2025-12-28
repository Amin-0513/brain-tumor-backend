from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report,f1_score
import numpy as np

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 18 * 18, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    


class TumorPrediction:
    def __init__(self):

        self.class_names = ["glioma", "meningioma", "notumor", "pituitary"]
        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Path to trained model
        self.model_path = "./models/brain_tumor_cnn_model.pth"

        # Initialize model
        self.model = BrainTumorCNN().to(self.device)

        # Load weights
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print("Model loaded successfully.")
        else:
            print(f"Error: Model file not found at {self.model_path}")

        # Define transform for input image
        self.transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, pred = torch.max(outputs, 1)
            prediction = self.class_names[pred.item()]
        return prediction
    
if __name__ == '__main__':
    tumor_model = TumorPrediction()
    test_image = rf"E:\Brain tumor Detection\dataset\brain-tumor-mri-dataset\glioma\gl-0001.jpg" 
    result = tumor_model.predict(test_image)
    print("Prediction class index:", result)