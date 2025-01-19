import sys
sys.path.append('D:/4th Sem/ML Trends/Assignment2')

import torch
import torch.nn as nn
import torch.nn.functional as F


class LRN(nn.Module):
    def __init__(self, size=5, alpha=1e-4, beta=0.75, k=2):
        super(LRN, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        return F.local_response_norm(x, self.size, self.alpha, self.beta, self.k)

class AlexNetLRN(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetLRN, self).__init__()
        self.features = nn.Sequential(
            # First convolution layer: input: 32x32, output: 64x16x16 (stride=2)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            LRN(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pooling layer: output: 64x7x7
            
            # Second convolution layer: input: 64x7x7, output: 192x7x7
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            LRN(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pooling layer: output: 192x3x3
            
            # Third convolution layer: input: 192x3x3, output: 384x3x3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fourth convolution layer: input: 384x3x3, output: 256x3x3
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fifth convolution layer: input: 256x3x3, output: 256x3x3
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pooling layer: output: 256x1x1
        )
        
        # Apply adaptive average pooling to get a fixed size output before FC layer
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # Output size is 6x6

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # 256 * 6 * 6 because of adaptive pooling
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # Final output layer with number of classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)  # Apply adaptive pooling to get a fixed size output (6x6)
        x = torch.flatten(x, 1)  # Flatten for the fully connected layers
        x = self.classifier(x)
        return x
    

    

    
