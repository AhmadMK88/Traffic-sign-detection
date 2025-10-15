import numpy as np
import torch

def compute_IoU(box1, box2):
    """
    Compute IoU between two sets of boxes in xywh format.
    
    Args:
        box1(tensor): first set of bounding boxes of shape (batch, S, S, B, 4)
        box2(tensor): second set of bounding boxes of shape (batch, S, S, B, 4)
    
    Returns:
        IoU: tensor of shape (batch, S, S, B)
    """

    # Convert from xywh to x1y1x2y2
    # x1y1: top left corner
    # x2y2: bottom right corner

    box1_x1 = box1[..., 0] - box1[..., 2] / 2
    box1_y1 = box1[..., 1] - box1[..., 3] / 2
    box1_x2 = box1[..., 0] + box1[..., 2] / 2
    box1_y2 = box1[..., 1] + box1[..., 3] / 2

    box2_x1 = box2[..., 0] - box2[..., 2] / 2
    box2_y1 = box2[..., 1] - box2[..., 3] / 2
    box2_x2 = box2[..., 0] + box2[..., 2] / 2
    box2_y2 = box2[..., 1] + box2[..., 3] / 2

    # Intersection coordinates
    intersection_x1 = torch.max(box1_x1, box2_x1)
    intersection_y1 = torch.max(box1_y1, box2_y1)
    intersection_x2 = torch.min(box1_x2, box2_x2)
    intersection_y2 = torch.min(box1_y2, box2_y2)

    # Clamp to avoid negative values
    intersection_w = (intersection_x2 - intersection_x1).clamp(0)
    intersection_h = (intersection_y2 - intersection_y1).clamp(0)

    # Intersection area
    intersection_area = intersection_w * intersection_h

    # Union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - intersection_area

    # IoU coverage
    return intersection_area / (union_area + 1e-6)


def count_tp_fp_fn(
    predicted_boxes,
    predicted_scores,
    predicted_classes,
    target_box,
    target_class,
    iou_threshold=0.5,
    score_threshold=0.5,
):
    """
    Compute precision and recall when there is only ONE ground truth per image.

    Args:
        predicted_boxes (tensor): (num_preds, 4) in xywh format
        predicted_scores (tensor): (num_preds,)
        predicted_classes (tensor): (num_preds,)
        target_box (tensor): (4,) in xywh format
        target_class (tensor): scalar (the class of the GT object)
    """
    # Filter low-confidence predictions
    mask = predicted_scores >= score_threshold
    predicted_boxes = predicted_boxes[mask]
    predicted_classes = predicted_classes[mask]

    TP, FP, FN = 0, 0, 0

    if len(predicted_boxes) == 0:
        # no predictions, but we have GT box 
        FN = 1
    
    else:
        # Compare each prediction with the single GT
        for index, predicted_box in enumerate(predicted_boxes):
            predicted_class = predicted_classes[index]

            iou = compute_IoU(predicted_box.unsqueeze(0), target_box.unsqueeze(0))

            if iou >= iou_threshold and predicted_class == target_class:
                TP = 1  # only one GT → at most 1 true positive
                break
        else:
            # loop finished without a match → all are false positives
            FP = len(predicted_boxes)
            FN = 1  # GT not matched

    return TP, FP, FN

def compute_ap(recall, precision):
    
    """
    Compute the Average Precision (AP) given precision and recall.
    Uses the 11-point interpolation method.

    Args:
        recall (Tensor): tensor of recall values at different confidence thresholds (N, ).
        precision (Tensor): tensor of precision values at the same thresholds (N, ).

    Return:
        ap(float): Average precision value
    """

    # Convert to numpy
    recall = recall.cpu().numpy()
    precision = precision.cpu().numpy()
    
    # 11-point interpolation (VOC 2007 style)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if (recall >= t).any() else 0
        ap += p / 11
    
    return ap

def compute_map(predictions, ground_truths, iou_threshold=0.5, score_threshold=0.5):
    """
    Compute mAP using count_tp_fp_fn (only 1 GT per image).

    Args:
        predictions (list): Each element is a dict with keys:
            - 'bbox': tensor (number_of_predictions, 4) xywh
            - 'scores': tensor (number_of_predictions,)
            - 'labels': tensor (number_of_predictions,)
        ground_truths (list): Each element is a dict with keys:
            - 'bbox': tensor (1, 4) xywh   # only 1 GT per image
            - 'labels': tensor (1,)         # only 1 GT class
    Returns:
        mAP (float): mean Average Precision 
    """
    aps = []
    classes = torch.unique(
        torch.cat(
            [ground_truth["labels"].unsqueeze(0) for ground_truth in ground_truths]
        )
    ).tolist()

    for cls in classes:

        class_scores = []
        class_matches = []
        num_of_groundThruth_boxes = 0

        for prediction, ground_truth in zip(predictions, ground_truths):
            
            # Extract target box and class  for this sample
            target_box = ground_truth['bbox']
            target_class = ground_truth['labels']

            # Calculate the number of images for current class
            num_of_groundThruth_boxes += int(target_class == cls)

            # Extract predicted boxes/scores/classes
            preditcted_boxes = prediction['bbox']
            preditcted_scores = prediction['scores']
            preditcted_classes = prediction['labels']

            # Calculate TP, FP, FN
            TP, FP, FN = count_tp_fp_fn(
                preditcted_boxes, preditcted_scores, preditcted_classes,
                target_box, target_class,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold
            )

            # Save scores + matches for PR curve
            for score in preditcted_scores:
                class_scores.append(score.item())
                class_matches.append(1 if TP else 0)  # 1 if correct, else 0

        # Sort predictions of this class by confidence
        if len(class_scores) == 0:
            continue

        sorted_idx = np.argsort(class_scores)[::-1].copy()
        class_matches = torch.tensor(class_matches, dtype=torch.float32)[sorted_idx]

        TP_cum = torch.cumsum(class_matches, dim=0)
        FP_cum = torch.cumsum(1 - class_matches, dim=0)

        recall = TP_cum / (num_of_groundThruth_boxes + 1e-6)
        precision = TP_cum / (TP_cum + FP_cum + 1e-6)

        ap = compute_ap(recall, precision)
        aps.append(ap)

    return sum(aps) / len(aps) if aps else 0.0
