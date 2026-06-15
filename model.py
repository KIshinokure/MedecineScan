import torch
import torch.nn as nn

class MedicalCNN(nn.Module):
    def __init__(self):
        super(MedicalCNN, self).__init__()
        # Сверточные слои для извлечения признаков из рентген-снимка
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Этот слой сделаем целевым для Grad-CAM (последний сверточный слой)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Полносвязные слои для классификации
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # На выходе 2 класса: Normal и Pneumonia
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x