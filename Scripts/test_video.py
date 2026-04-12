import os
import cv2
import torch
import numpy as np
import argparse
from Utils.model import Model

names = [
  "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
  "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
  "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
  "No passing", "No passing for vehicles over 3.5 metric tons",
  "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
  "No vehicles", "Vehicles over 3.5 metric tons prohibited",
  "No entry", "General caution", "Dangerous curve to the left",
  "Dangerous curve to the right", "Double curve", "Bumpy road",
  "Slippery road", "Road narrows on the right", "Road work",
  "Traffic signals", "Pedestrians", "Children crossing",
  "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
  "End of all speed and passing limits", "Turn right ahead",
  "Turn left ahead", "Ahead only", "Go straight or right",
  "Go straight or left", "Keep right", "Keep left",
  "Roundabout mandatory", "End of no passing",
  "End of no passing for vehicles over 3.5 metric tons"]

def extract_frames(video_path, frames_dir):
    """
    Extract and store frames from given video
    
    Args:
        video_path (string): the path of video to extract frames from
        frames_dir (string): the path of the folder where extracted frames are stored
    """
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(frames_dir, f"frame_{count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()
    print(f"Extracted {count} frames.")


def load_model(checkpoint_path, device):
    """
    Load model with weights from a checkpoint
    
    Args:
        checkpoint_path (string): the path of saved model checkpoint file
        device (string): the device where the model is loaded to
    Returns:
        model (torch.nn.Module): model with loaded weights
    """
    model = Model(num_of_classes=43, num_of_boxes=2).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def preprocess_frame(frame, img_size):
    """
    Preprocess video frame before passing to model
    
    Args:
        frame (np.ndarray): single video frame
        img_size (int): target size for resizing frame
    Returns:
        frame_tensor (torch.Tensor): ready frame for feeding to model
    """
    frame_resized = cv2.resize(frame, (img_size, img_size))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_rgb.astype(np.float32) / 255.0

    frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1)
    return frame_tensor.unsqueeze(0)


def decode_predictions(predictions, conf_threshold, B):
    """
    Decode raw model output to actual prediction
    
    Args:
        predictions (torch.Tensor): model output for one sample
        conf_threshold (float): minimum score for correct detection
        B (int): number of bounding boxes
    Returns:
        result (dict): formated model result
    """
    cells_num = predictions.shape[1]

    result = {"bbox": [], "scores": [], "labels": []}

    for cell_y in range(cells_num):
        for cell_x in range(cells_num):

            boxes = predictions[0, cell_y, cell_x, :5 * B].view(B, 5)

            class_prob, class_idx = torch.max(
                predictions[0, cell_y, cell_x, 5 * B:], dim=0
            )

            for b in range(B):
                x, y, w, h, conf = boxes[b]
                score = conf * class_prob

                if score > conf_threshold:
                    result["bbox"].append([x.item(), y.item(), w.item(), h.item()])
                    result["scores"].append(score.item())
                    result["labels"].append(class_idx)

    if len(result["bbox"]) > 0:
        result["bbox"] = torch.tensor(result["bbox"])
        result["scores"] = torch.tensor(result["scores"])
        result["labels"] = torch.stack(result["labels"])
    else:
        result["bbox"] = torch.empty((0, 4))
        result["scores"] = torch.empty((0,))
        result["labels"] = torch.empty((0,), dtype=torch.long)

    return result


def draw_predictions(frame, predictions):
    """
    Visualize prediction result on frame 
    
    Args:
        frame (np.ndarray): original image where boxes will be drawn
        predictions (dict): model predictions of this frame
    Returns:
        frame (np.ndarray): fram with predictions drawn on it 
    """
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

        # Clamp
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{names[label.item()]}:{score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            3
        )
        cv2.putText(
            frame,
            f"{names[label.item()]}:{score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    return frame


def process_frames(frames_dir, output_dir, model, device, img_size, conf_threshold):
    """
    Process a folder of frames, run inference, and save annotated results.

    Args:
        - frames_dir (str): Path to directory containing input frames 
        - output_dir (str): Path to directory where annotated frames will be saved
        - model (torch.nn.Module): Trained detection model
        - device (str or torch.device): Device to run inference on 
        - img_size (int): Target image size for model input 
        - conf_threshold (float): Confidence threshold for filtering predictions
    """
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted([
        f for f in os.listdir(frames_dir) if f.endswith('.jpg')
    ])

    for frame_file in frame_files:
        img_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(img_path)

        input_tensor = preprocess_frame(frame, img_size).to(device)

        with torch.no_grad():
            predictions = model(input_tensor)

        decoded = decode_predictions(predictions, conf_threshold, B=2)
        annotated = draw_predictions(frame, decoded)

        out_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(out_path, annotated)

    print(f"Processed {len(frame_files)} frames.")


def frames_to_video(frames_dir, output_path, fps=30):
    """
    Convert a sequence of image frames into a video using ffmpeg.

    Args:
        - frames_dir (str): Path to directory containing input frames 
        - output_path (str): Path where the output video will be saved 
        - fps (int): Frames per second for the output video 
    """
    command = f"""
    ffmpeg -y -framerate {fps} \
    -i {frames_dir}/frame_%05d.jpg \
    -c:v libx264 -pix_fmt yuv420p {output_path}
    """
    os.system(command)
    print(f"Video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output.mp4")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--conf_threshold", type=float, default=0.2)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames_dir = "frames_temp"
    annotated_dir = "annotated_frames_temp"

    # 1. Extract frames
    extract_frames(args.video_path, frames_dir)

    # 2. Load model
    model = load_model(args.checkpoint_path, device)

    # 3. Process frames
    process_frames(
        frames_dir,
        annotated_dir,
        model,
        device,
        args.img_size,
        args.conf_threshold
    )

    # 4. Rebuild video
    frames_to_video(annotated_dir, args.output_path, fps=30)


if __name__ == "__main__":
    main()