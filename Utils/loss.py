import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, S, B=2, C=43, lambda_coord=5, lambda_noobj=0.5):
        super(Loss, self).__init__()
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
        # Batch size
        N = predictions.size(0)   
        
        # Split predictions into boxes and classes
        # predictions shape: [N, S, S, B*5 + C]
        predicted_boxes   = predictions[..., :self.B*5].view(N, self.S, self.S, self.B, 5)
        predicted_classes = predictions[..., self.B*5:]

        target_tensor = torch.zeros_like(predictions)

        for i, (label, bbox) in enumerate(zip(targets['class'], targets['bbox'])):
    
            cell_x = int(self.S * bbox[0])
            cell_y = int(self.S * bbox[1])

            target_tensor[i, cell_y, cell_x, 0:5] = torch.tensor(
                [bbox[0], bbox[1], bbox[2], bbox[3], 1.0],
                dtype=torch.float32
            )

            target_tensor[i, cell_y, cell_x, self.B*5 + label] = 1.0
        
        # Split targets into boxes and classes 
        target_boxes   = target_tensor[..., :self.B*5].view(N, self.S, self.S, self.B, 5)
        target_classes = target_tensor[..., self.B*5:]

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


        object_cell_mask = target_boxes[..., 4].max(dim=3).values
        # Classification Loss: only for cells with objects
        class_loss = self.mse(
            (object_cell_mask.unsqueeze(-1) * predicted_classes),
            (object_cell_mask.unsqueeze(-1) * target_classes)
        )

        total_loss = loc_loss + object_loss + no_object_loss + class_loss
        return total_loss
