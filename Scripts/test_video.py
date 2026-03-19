import argparse
import cv2
import torch
import numpy as np
from Utils.model import Model

def preprocess_frame(frame: np.ndarray, img_size: int) -> torch.tensor:
    '''
    Preprocess video frame before passing to model
    Args:
        - frame (np.ndarray): Input frame in BGR format (H, W, C) from OpenCV
        - img_size (int): Target size to resize the frame (img_size x img_size)

    Returns:
        - torch.Tensor: Preprocessed image tensor of shape (1, 3, img_size, img_size) normalized to range [0, 1] 
    '''
    
    frame_resized = cv2.resize(frame, (img_size, img_size))

    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_rgb.astype(np.float32) / 255.0

    frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1)

    return frame_tensor.unsqueeze(0)


def draw_predictions(frame: np.ndarray, predictions: dict) -> np.ndarray:
    '''
    Draw predicted bounding boxes and labels on a video frame.

    Args:
        - frame (np.ndarray): Original image frame in BGR format (H, W, C)
        - predictions (dict): Dictionary containing:
            * "bbox" (torch.Tensor): Bounding boxes (N, 4) in YOLO format (x_center, y_center, w, h)
            * "scores" (torch.Tensor): Confidence scores (N,)
            * "labels" (torch.Tensor): Class indices (N,)

    Returns:
        - np.ndarray: Frame with drawn bounding boxes and labels
    '''

    h, w = frame.shape[:2]

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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)

        cv2.putText(
            frame,
            f"{label.item()}:{score:.2f}",
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,255),
            1
        )

    return frame


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output.mp4")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--conf_threshold", type=float, default=0.5)

    args = parser.parse_args()

    CLASSES_NUM = 43
    BOUNDING_BOXES_NUM = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = Model(
        num_of_classes=CLASSES_NUM,
        num_of_boxes=BOUNDING_BOXES_NUM
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Open video
    cap = cv2.VideoCapture(args.video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        args.output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    print(f"Processing video: {args.video_path}")

    with torch.no_grad():
        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            input_tensor = preprocess_frame(frame, args.img_size).to(device)

            # Forward pass
            predictions = model(input_tensor)

            cells_num = predictions.shape[1]

            predicted_sample = {
                "bbox": [],
                "scores": [],
                "labels": [],
            }

            # Decode predictions
            for cell_x in range(cells_num):
                for cell_y in range(cells_num):

                    boxes = predictions[
                        0, cell_x, cell_y, :5 * BOUNDING_BOXES_NUM
                    ].view(BOUNDING_BOXES_NUM, 5)

                    class_prob, class_idx = torch.max(
                        predictions[
                            0,
                            cell_x,
                            cell_y,
                            5 * BOUNDING_BOXES_NUM:
                        ],
                        dim=0
                    )

                    for b in range(BOUNDING_BOXES_NUM):

                        x, y, w, h, conf = boxes[b]
                        score = conf * class_prob

                        if score > args.conf_threshold:
                            predicted_sample["bbox"].append(
                                [x.item(), y.item(), w.item(), h.item()]
                            )
                            predicted_sample["scores"].append(score.item())
                            predicted_sample["labels"].append(class_idx)

            # Convert to tensors
            if len(predicted_sample["bbox"]) > 0:
                predicted_sample["bbox"] = torch.tensor(predicted_sample["bbox"])
                predicted_sample["scores"] = torch.tensor(predicted_sample["scores"])
                predicted_sample["labels"] = torch.stack(predicted_sample["labels"])
            else:
                predicted_sample["bbox"] = torch.empty((0,4))
                predicted_sample["scores"] = torch.empty((0,))
                predicted_sample["labels"] = torch.empty((0,), dtype=torch.long)

            # Draw
            frame = draw_predictions(frame, predicted_sample)

            # Write frame
            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Saved output video to {args.output_path}")


if __name__ == "__main__":
    main()