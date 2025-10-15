import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # Load ResNet50 

        # Keep everything except avgpool and fully connected classifier
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.backbone(x) # output shape [B, 2048, H/32, W/32]

class Head(nn.Module):
    def __init__(self, input, num_of_classes, num_of_boxes):
        super().__init__()
        self.num_of_classes = num_of_classes
        self.num_of_boxes = num_of_boxes

        self.reduce = nn.Conv2d(input, 512, kernel_size=1)

        self.conv_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),   
        )

        self.pred = nn.Conv2d(
            512,
            self.num_of_boxes * 5 + self.num_of_classes,
            kernel_size=1
        )
    
    def forward(self, x ):

        # Reduce channels from 2048 to 512
        x = self.reduce(x)
        # Refine features
        x = self.conv_block(x)

        # Prediction
        x = self.pred(x)

        # Reshape: (N, S, S, B*5 + C)
        out = x.permute(0, 2, 3, 1)

        return out

class Model(nn.Module):
    def __init__(self, num_of_classes, num_of_boxes):
        super().__init__()
        self.backbone = Backbone()
        self.head = Head(2048, num_of_classes, num_of_boxes)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)

        return predictions

