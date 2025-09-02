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
    def __init__(self, input, num_of_classes, num_of_anchors):
        super().__init__()
        self.num_of_classes = num_of_classes
        self.num_of_anchors = num_of_anchors

        self.conv_block = nn.Sequential(
            nn.Conv2d(input, input, kernel_size=3, padding=1),
            nn.BatchNorm2d(input),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(input, input, kernel_size=3, padding=1),
            nn.BatchNorm2d(input),
            nn.LeakyReLU(0.1, inplace=True),   
        )

        self.pred = nn.Conv2d(
            input,
            self.num_of_anchors * ( 5 + self.num_of_classes),
            kernel_size=1
        )
    
    def forward(self, x ):
        # Refine features
        x = self.conv_block(x)

        # Prediction
        x = self.pred(x)

        # Reshape: [B, num_of_anchors, 5 + num_of_classes, H, W]
        B, _, H, W = x.shape
        out = x.view(B, self.num_of_anchors, 5 + self.num_of_classes, H, W)

        return out

class Model(nn.Module):
    def __init__(self, num_of_classes, num_of_anchors):
        super().__init__()
        self.backbone = Backbone()
        self.head = Head(2048, num_of_classes, num_of_anchors)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)

        return predictions

