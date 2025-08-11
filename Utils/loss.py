import torch
import torch.nn as nn
import torch.nn.functional as F

class loss(nn.Module):
    def __init__(self, S=13, B=2, C=43, lambda_coord=5, lambda_noobj=0.5):

        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, targets):
        '''
        Calculate the sum of localization loss, objectness loss and classification loss

        Args:
            predictions (tensor): predicted bounding boxes with shape (batch_size, S, S, B*5 + C)           
            targets (tensor): ground truth bounding boxes with shape (batch_size, S, S, B*5 + C)
        '''
        # Number of images per batch
        N = predictions.size(0)

        # Split predictions and targets to boxes and classes
        predicted_boxes = predictions[..., :self.B*5].view(N, self.S, self.S, self.B, 5)
        predicted_classes = predictions[..., self.B*5:]  

        target_boxes = targets[..., :self.B*5].view(N, self.S, self.S, self.B, 5)
        target_classes = targets[..., self.B*5:]

        # Create masks by extracting object existance
        object_mask = target_boxes[..., 4]  

        # Localization Loss: only for cells with objects
        predicted_xy = predicted_boxes[..., :2]
        predicted_wh = torch.sqrt(predicted_boxes[..., 2:4].clamp(1e-6, 1e6))

        target_xy = target_boxes[..., :2]
        target_wh = torch.sqrt(target_boxes[..., 2:4])

        loc_loss = self.lambda_coord * (
            self.mse(object_mask.unsqueeze(-1) * predicted_xy, object_mask.unsqueeze(-1) * target_xy) +
            self.mse(object_mask.unsqueeze(-1) * predicted_wh, object_mask.unsqueeze(-1) * target_wh)
        )

        # Objectness Loss
        predicted_confidence  = predicted_boxes[..., 4]
        target_confidence  = target_boxes[..., 4]
        no_object_mask = 1 - object_mask

        object_loss = self.mse(object_mask * predicted_confidence, object_mask * target_confidence)
        no_object_loss = self.lambda_noobj * self.mse(
            no_object_mask * predicted_confidence, no_object_mask * target_confidence
        )

        # Classification Loss: only for cells with objects
        class_loss = self.mse(
            (object_mask.unsqueeze(-1) * predicted_classes),
            (object_mask.unsqueeze(-1) * target_classes)
        )

        total_loss = loc_loss + object_loss + no_object_loss + class_loss
        return total_loss
