import argparse
import cv2
import numpy as np
import torch
import json
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from Utils.dataset import SyentheticDataset
from Utils.model import Model
from Utils.metrics import compute_map


def draw_predictions(image, predictions, targets, save_path=None):

    image = image.copy()
    h, w = image.shape[:2]

    # Draw ground truth (GREEN)
    for box, label in zip(targets["bbox"], targets["labels"]):

        x, y, bw, bh = box

        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(
            image,
            f"GT:{label.item()}",
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1
        )

    # Draw predictions (RED)
    for box, score, label in zip(
        predictions["bbox"],
        predictions["scores"],
        predictions["labels"]
    ):

        x, y, bw, bh = box

        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)

        cv2.putText(
            image,
            f"{label.item()}:{score:.2f}",
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,255),
            1
        )

    if save_path:
        cv2.imwrite(save_path, image)

    return image


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_report", type=str, default="test_report.json")

    args = parser.parse_args()

    CLASSES_NUM = 43
    BOUNDING_BOXES_NUM = 2

    test_images_path = f"{args.dataset_path}/images/test"
    test_labels_path = f"{args.dataset_path}/labels/test"

    os.makedirs("visual_results", exist_ok=True)

    test_dataset = SyentheticDataset(test_images_path, test_labels_path)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False
    )

    # Load original image paths
    image_files = sorted(os.listdir(test_images_path))
    image_paths = [os.path.join(test_images_path, f) for f in image_files]

    global_index = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(
        num_of_classes=CLASSES_NUM,
        num_of_boxes=BOUNDING_BOXES_NUM
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint_epoch = checkpoint.get("epoch", "unknown")

    model.eval()

    total_mAP = 0.0
    total_accuracy = 0.0
    num_batches = 0

    print(f"\nTesting on {len(test_dataset)} samples...")

    progress_bar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc="Testing",
        ncols=100,
        leave=True
    )

    with torch.no_grad():

        for batch_idx, (images, targets) in progress_bar:

            all_predictions = []
            all_targets = []

            images = images.to(device)
            targets["class"] = targets["class"].to(device)
            targets["bbox"] = targets["bbox"].to(device)

            predictions = model(images)

            cells_num = predictions.shape[1]
            batch_size = images.size(0)

            for sample_num in range(batch_size):

                predicted_sample = {
                    "bbox": torch.tensor([], device=device),
                    "scores": torch.tensor([], device=device),
                    "labels": torch.tensor([], device=device),
                }

                for cell_x in range(cells_num):
                    for cell_y in range(cells_num):

                        boxes = predictions[
                            sample_num,
                            cell_x,
                            cell_y,
                            :5 * BOUNDING_BOXES_NUM
                        ].view(BOUNDING_BOXES_NUM, 5)

                        class_probability, class_idx = torch.max(
                            predictions[
                                sample_num,
                                cell_x,
                                cell_y,
                                5 * BOUNDING_BOXES_NUM:
                            ],
                            dim=0
                        )

                        for box in range(BOUNDING_BOXES_NUM):

                            x, y, w, h, object_confidence = boxes[box]

                            object_score = object_confidence * class_probability

                            if object_score > 0.5:

                                predicted_box = torch.stack(
                                    [x, y, w, h]
                                ).view(1, 4)

                                predicted_sample["bbox"] = torch.cat(
                                    [predicted_sample["bbox"], predicted_box]
                                )

                                predicted_sample["scores"] = torch.cat(
                                    [
                                        predicted_sample["scores"],
                                        object_score.view(1)
                                    ]
                                )

                                predicted_sample["labels"] = torch.cat(
                                    [
                                        predicted_sample["labels"],
                                        class_idx.view(1)
                                    ]
                                )

                all_predictions.append(predicted_sample)

                target_sample = {
                    "bbox": targets["bbox"][sample_num].view(1, 4),
                    "labels": targets["class"][sample_num].unsqueeze(0),
                }

                all_targets.append(target_sample)

                # Load original image from disk
                img_np = cv2.imread(image_paths[global_index])
                global_index += 1

                draw_predictions(
                    img_np,
                    predicted_sample,
                    target_sample,
                    save_path=f"visual_results/sample_{batch_idx}_{sample_num}.jpg"
                )

            batch_mAP = compute_map(all_predictions, all_targets)
            total_mAP += batch_mAP

            object_mask = predictions[..., 4] > 0.5

            predicted_classes = torch.argmax(
                predictions[..., 5 * BOUNDING_BOXES_NUM:],
                dim=-1
            )

            true_classes = targets["class"].view(-1, 1, 1).expand_as(predicted_classes)

            correct = ((predicted_classes == true_classes) & object_mask).sum().item()
            total = object_mask.sum().item() + 1e-6

            batch_accuracy = correct / total

            total_accuracy += batch_accuracy
            num_batches += 1

            progress_bar.set_postfix({
                "mAP": f"{batch_mAP:.4f}",
                "Accuracy": f"{batch_accuracy*100:.2f}%"
            })

    progress_bar.close()

    avg_mAP = total_mAP / num_batches if num_batches > 0 else 0.0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Samples: {len(test_dataset)}")
    print(f"Average mAP: {avg_mAP:.4f}")
    print(f"Average Accuracy: {avg_accuracy*100:.2f}%")
    print("="*60 + "\n")

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