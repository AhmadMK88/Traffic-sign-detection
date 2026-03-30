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

        # Anchors (normalized w.r.t image size)
        self.register_buffer(
            "anchors",
            torch.tensor([
                [0.08, 0.08],
                [0.15, 0.15],
            ])  # shape: (B, 2)
        )
    
    def forward(self, x):
        x = self.reduce(x)
        x = self.conv_block(x)
        x = self.pred(x)

        # (B, C, S, S) -> (B, S, S, C)
        x = x.permute(0, 2, 3, 1)

        B = self.num_of_boxes
        C = self.num_of_classes
        S = x.shape[1]

        # Split predictions
        box_preds = x[..., :5 * B].view(*x.shape[:3], B, 5)
        class_preds = x[..., 5 * B:]

        # Apply activations
        box_preds[..., 0:2] = torch.sigmoid(box_preds[..., 0:2])  # x, y
        box_preds[..., 4] = torch.sigmoid(box_preds[..., 4])      # confidence


        # Apply anchors (w, h)
        anchors = self.anchors.view(1, 1, 1, B, 2).to(x.device)
        box_preds[..., 2:4] = anchors * torch.exp(box_preds[..., 2:4])

        # GRID DECODING 
        grid_y, grid_x = torch.meshgrid(
            torch.arange(S),
            torch.arange(S),
            indexing="ij"
        )

        grid_x = grid_x.to(x.device).view(1, S, S, 1)
        grid_y = grid_y.to(x.device).view(1, S, S, 1)

        box_preds[..., 0] = (box_preds[..., 0] + grid_x) / S
        box_preds[..., 1] = (box_preds[..., 1] + grid_y) / S

        class_preds = torch.softmax(class_preds, dim=-1)

        # Merge back
        box_preds = box_preds.view(*x.shape[:3], 5 * B)
        out = torch.cat([box_preds, class_preds], dim=-1)

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

