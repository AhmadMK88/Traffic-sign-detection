import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
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
    parser.add_argument("--batch_size", type=str, required=True)
    parser.add_argument("--epoches", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=False)

    args = parser.parse_args()

    train_set_images_path = f"{args.dataset_path}/images/train"
    train_set_labels_path = f"{args.dataset_path}/labels/train"
    val_set_images_path = f"{args.dataset_path}/images/val"
    val_set_labels_path = f"{args.dataset_path}/labels/val"

    CLASSES_NUM = 43
    BOUNDING_BOXES_NUM = 2

    batch_size = int(args.batch_size)
    epoches = int(args.epoches)

    # Extract datasets
    train_dataset = SyentheticDataset(train_set_images_path, train_set_labels_path)
    val_dataset = SyentheticDataset(val_set_images_path, val_set_labels_path)

    # Create datasets loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, pin_memory=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(num_of_classes=CLASSES_NUM, num_of_boxes=BOUNDING_BOXES_NUM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = Loss(S=20).to(device)
    start_epoch = 0

    if args.model_config_path is not None :
        checkpoint = torch.load(args.model_config_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] 

    for epoch in range(start_epoch, start_epoch + epoches):
        print('===================================')
        print(f"Epoch {epoch+1} Started:")

        # ---- Training ----
        train_loss = 0.0
        train_accuracy = 0.0

        # Set model to training mode
        model.train()

        # Create training progress bar for the epoch
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc='Training: ',
            ncols=100,
            leave=False
        )

        for batch_idx, (images, targets) in progress_bar:

            # Move image and target tensors to current device
            images = images.to(device)
            targets["class"] = targets["class"].to(device)
            targets["bbox"] = targets["bbox"].to(device)

            # Forward prob
            predictions = model(images)

            # Compute loss
            batch_loss = criterion(predictions, targets)  
            optimizer.zero_grad()

            # Compute accuracy
            predicted_classes = torch.argmax(predictions[..., 5 * BOUNDING_BOXES_NUM:], dim=-1)  
            true_classes = targets["class"].view(-1, 1, 1).expand_as(predicted_classes)
            correct = (predicted_classes == true_classes).sum().item()
            total = predicted_classes.numel()
            batch_accuracy = correct / total

            # Update weights
            batch_loss.backward()
            optimizer.step()

            # Track running loss and acc
            train_loss += batch_loss.item()
            train_accuracy += batch_accuracy

            # Update progress bar text
            progress_bar.set_postfix({
                "Train loss": f"{batch_loss.item():.4f}",
                "Train accuracy": f"{batch_accuracy*100:.2f}%"
            })

        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_accuracy / len(train_loader)

        progress_bar.close()
        print(
            f"Avg train loss: {avg_train_loss:.4f} | Avg train accuracy: {avg_train_accuracy*100:.2f}%"
        )

        # ---- Validation ----
        validation_mAP = 0.0
        validation_accuracy = 0.0

        # Set model to evaluation mode
        model.eval()

        # Create evaluation progress bar for the epoch
        progress_bar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc='Validation: ',
            ncols=100,
            leave=False
        )
    
        with torch.no_grad():
            for batch_idx, (images, targets) in progress_bar:
                all_predictions, all_targets = [], []
                
                # Move image and target tensors to current device
                images = images.to(device)
                targets["class"] = targets["class"].to(device)
                targets["bbox"] = targets["bbox"].to(device)

                predictions = model(images)
                cells_num = predictions.shape[1]
                current_batch_size = images.size(0)

                for sample_num in range(current_batch_size):
                    # Extract predicted boxes coordinates, ojectness score and class probability
                    predicted_sample = {
                        "bbox": torch.tensor([], device=device),
                        "scores": torch.tensor([], device=device),
                        "labels": torch.tensor([], device=device),
                    }
                    for cell_x in range(cells_num):
                        for cell_y in range(cells_num):

                            boxes = predictions[
                                sample_num, cell_x, cell_y, : 5 * BOUNDING_BOXES_NUM
                            ].view(BOUNDING_BOXES_NUM, 5)

                            class_probability, class_idx = torch.max(
                                predictions[sample_num, cell_x, cell_y, 5 * BOUNDING_BOXES_NUM :], dim=0
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

                    all_predictions.append(predicted_sample)

                    # Add current batch targets boxes and classes
                    target_sample = {
                        "bbox": targets["bbox"][sample_num].view(1, 4),
                        "labels": targets["class"][sample_num].unsqueeze(0),
                    }
                    all_targets.append(target_sample)

                # Compute mAP
                batch_mAP = compute_map(all_predictions, all_targets)

                # Compute validation accuracy
                predicted_classes = torch.argmax(predictions[..., 5 * BOUNDING_BOXES_NUM:], dim=-1)
                true_classes = targets["class"].view(-1, 1, 1).expand_as(predicted_classes)
                val_correct = (predicted_classes == true_classes).sum().item()
                val_total = predicted_classes.numel()
                batch_accuracy = val_correct / val_total

                # Track validation mAP and acc
                validation_mAP += batch_mAP
                validation_accuracy += batch_accuracy

                # Update progress bar text
                progress_bar.set_postfix({
                    "Validation mAP": f"{batch_mAP:.4f}",
                    "Validation accuracy": f"{validation_accuracy*100:.2f}%"
                })
            avg_validation_mAP = validation_mAP / len(val_loader)
            avg_validation_accuracy = validation_accuracy / len(val_loader)
            
            progress_bar.close()
            print(
            f"Avg validation mAP: {avg_validation_mAP:.4f} | Avg valiadtion accuracy: {avg_validation_accuracy*100:.2f}%"
            )

        print(f"Epoch {epoch+1} Finished:")
        print('=====================================')
    
    # Check if a folder for weights exist
    if os.path.isdir("weights") == False:
        os.mkdir("weights")

    checkpoint_path = os.path.join("weights", f"{epoch+1} epochs trained model")
    torch.save({
        "epoch": epoch+1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)

if __name__ == "__main__" :
    main()
