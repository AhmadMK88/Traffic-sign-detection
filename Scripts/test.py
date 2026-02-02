import argparse
import torch
import json
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from Utils.dataset import SyentheticDataset
from Utils.model import Model
from Utils.metrics import compute_map

def main():
    """
    Test the trained model on a labeled dataset (validation or test split).
    Computes mAP, accuracy, and generates a performance report.
    """
    # Parse script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_report", type=str, default="test_report.json")
    args = parser.parse_args()

    # Model hyperparameters
    CLASSES_NUM = 43
    BOUNDING_BOXES_NUM = 2

    # Dataset paths (using test split)
    test_images_path = f"{args.dataset_path}/images/test"
    test_labels_path = f"{args.dataset_path}/labels/test"

    # Load datasettest
    test_dataset = SyentheticDataset(test_images_path, test_labels_path)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False
    )    

    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(num_of_classes=CLASSES_NUM, num_of_boxes=BOUNDING_BOXES_NUM).to(device)

    # Load model checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint_epoch = checkpoint.get("epoch", "unknown")

    # Evaluation mode
    model.eval()

    # Metrics tracking
    total_mAP = 0.0
    total_accuracy = 0.0
    num_batches = 0

    print(f"\nTesting on {len(test_dataset)} samples...")
    progress_bar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc='Testing: ',
        ncols=100,
        leave=True
    )

    with torch.no_grad():
        for batch_idx, (images, targets) in progress_bar:
            all_predictions, all_targets = [], []

            # Move to device
            images = images.to(device)
            targets["class"] = targets["class"].to(device)
            targets["bbox"] = targets["bbox"].to(device)

            # Forward pass
            predictions = model(images)
            cells_num = predictions.shape[1]
            current_batch_size = images.size(0)

            # Extract predictions per sample
            for sample_num in range(current_batch_size):
                predicted_sample = {
                    "bbox": torch.tensor([], device=device),
                    "scores": torch.tensor([], device=device),
                    "labels": torch.tensor([], device=device),
                }

                # Iterate over grid cells
                for cell_x in range(cells_num):
                    for cell_y in range(cells_num):
                        # Extract boxes and class probabilities for this cell
                        boxes = predictions[
                            sample_num, cell_x, cell_y, : 5 * BOUNDING_BOXES_NUM
                        ].view(BOUNDING_BOXES_NUM, 5)

                        class_probability, class_idx = torch.max(
                            predictions[sample_num, cell_x, cell_y, 5 * BOUNDING_BOXES_NUM :], dim=0
                        )

                        # Check each bounding box in the cell
                        for box in range(BOUNDING_BOXES_NUM):
                            x, y, w, h, object_confidence = boxes[box]
                            object_score = object_confidence * class_probability

                            # Keep high-confidence predictions
                            if object_score > 0.5:
                                predicted_box = torch.stack([x, y, w, h]).view(1, 4)
                                predicted_sample["bbox"] = torch.cat(
                                    [predicted_sample["bbox"], predicted_box]
                                )
                                predicted_sample["scores"] = torch.cat(
                                    [
                                        predicted_sample["scores"],
                                        object_score.view(1),
                                    ]
                                )
                                predicted_sample["labels"] = torch.cat(
                                    [predicted_sample["labels"], class_idx.view(1)]
                                )

                all_predictions.append(predicted_sample)

                # Add target for this sample
                target_sample = {
                    "bbox": targets["bbox"][sample_num].view(1, 4),
                    "labels": targets["class"][sample_num].unsqueeze(0),
                }
                all_targets.append(target_sample)

            # Compute mAP for batch
            batch_mAP = compute_map(all_predictions, all_targets)
            total_mAP += batch_mAP

            # Compute accuracy for batch
            object_mask = predictions[..., 4] > 0.5
            predicted_classes = torch.argmax(predictions[..., 5 * BOUNDING_BOXES_NUM:], dim=-1)
            true_classes = targets["class"].view(-1, 1, 1).expand_as(predicted_classes)

            correct = ((predicted_classes == true_classes) & object_mask).sum().item()
            total = object_mask.sum().item() + 1e-6
            batch_accuracy = correct / total

            total_accuracy += batch_accuracy
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                "mAP": f"{batch_mAP:.4f}",
                "Accuracy": f"{batch_accuracy*100:.2f}%"
            })

    progress_bar.close()

    # Compute averages
    avg_mAP = total_mAP / num_batches if num_batches > 0 else 0.0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0

    # Print results
    print("\n" + "="*60)
    print(f"TEST RESULTS on test split")
    print("="*60)
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Number of samples: {len(test_dataset)}")
    print(f"Average mAP: {avg_mAP:.4f}")
    print(f"Average Accuracy: {avg_accuracy*100:.2f}%")
    print("="*60 + "\n")

    # Save report
    report = {
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_epoch": checkpoint_epoch,
        "dataset_split": "test",
        "num_samples": len(test_dataset),
        "batch_size": args.batch_size,
        "num_classes": CLASSES_NUM,
        "metrics": {
            "mAP": float(avg_mAP),
            "accuracy": float(avg_accuracy),
            "accuracy_percent": float(avg_accuracy * 100)
        }
    }

    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
