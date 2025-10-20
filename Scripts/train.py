import argparse
import torch
from torch.utils.data import DataLoader
from Utils.dataset import SyentheticDataset
from Utils.model import Model
from Utils.loss import Loss
from Utils.metrics import *
import os
import zipfile


def main():
    # Parse script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)

    args = parser.parse_args()

    train_set_images_path = f"{args.dataset_path}/images/train"
    train_set_labels_path = f"{args.dataset_path}/labels/train"
    val_set_images_path = f"{args.dataset_path}/images/val"
    val_set_labels_path = f"{args.dataset_path}/labels/val"

    CLASSES_NUM = 43
    BOUNDING_BOXES_NUM = 2
    BATCH_SIZE = 16
    EPOCHES = 10

    # Extract datasets
    train_dataset = SyentheticDataset(train_set_images_path, train_set_labels_path)
    val_dataset = SyentheticDataset(val_set_images_path, val_set_labels_path)

    # Create datasets loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(num_of_classes=CLASSES_NUM, num_of_boxes=BOUNDING_BOXES_NUM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = Loss(S=20).to(device)

    for epoch in range(EPOCHES):
        # Set model to training mode
        model.train()

        # Training
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            # Move image and target tensors to current device
            imgs = imgs.to(device)
            targets["class"] = targets["class"].to(device)
            targets["bbox"] = targets["bbox"].to(device)

            # Forward prob
            preds = model(imgs)

            # Compute loss
            batch_loss = criterion(preds, targets)  # this returns a scalar tensor
            optimizer.zero_grad()

            # Update weights
            batch_loss.backward()
            optimizer.step()

            # Print epoch, batch, and loss
            print(
                f"Epoch [{epoch+1}/{EPOCHES}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {batch_loss.item():.4f}"
            )

        # Validation
        model.eval()
        with torch.no_grad():
            all_preds, all_targets = [], []
            preds = model(imgs)
            cells_num = preds.shape[1]

            for sample_num in range(BATCH_SIZE):
                # Extract predicted boxes coordinates, ojectness score and class probability
                predicted_sample = {
                    "bbox": torch.tensor([], device=device),
                    "scores": torch.tensor([], device=device),
                    "labels": torch.tensor([], device=device),
                }
                for cell_x in range(cells_num):
                    for cell_y in range(cells_num):

                        boxes = preds[
                            sample_num, cell_x, cell_y, : 5 * BOUNDING_BOXES_NUM
                        ].view(BOUNDING_BOXES_NUM, 5)

                        class_probability, class_idx = torch.max(
                            preds[sample_num, cell_x, cell_y, 5 * BOUNDING_BOXES_NUM :], dim=0
                        )

                        for box in range(BOUNDING_BOXES_NUM):
                            x, y, w, h, object_confindace = boxes[box]
                            object_existance_score = (
                                object_confindace * class_probability
                            )

                            if object_existance_score > 0.5:
                                predicted_box = torch.stack([x, y, w, h]).view(1, 4)
                                predicted_sample["bbox"] = torch.cat(
                                    [predicted_sample["bbox"], predicted_box]
                                )
                                predicted_sample["scores"] = torch.cat(
                                    [
                                        predicted_sample["scores"],
                                        object_existance_score.view(1),
                                    ]
                                )
                                predicted_sample["labels"] = torch.cat(
                                    [predicted_sample["labels"], class_idx.view(1)]
                                )

                all_preds.append(predicted_sample)

            # Add current batch targets boxes and classes
            target_sample = {
                "bbox": targets["bbox"][sample_num].view(1, 4),
                "labels": targets["class"][sample_num].unsqueeze(0),
            }
            all_targets.append(target_sample)

            print("mAP:", compute_map(all_preds, all_targets))

if __name__ == "__main__" :
    main()