import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random

def main():

    classes_names = [
    '20', '30', '50', '60', '70', '80', 'End of speed limit', '100', '120', 'No passing',
    'No passing for vehicles over 3,5 tonnes', 'Priority', 'Priority road', 'Yield', 'Stop',
    'Road closed', 'Vehicles over 3,5 tonnes prohibited', 'Do not enter', 'General danger',
    'Left curve', 'Right curve', 'Double curve', 'Uneven road surface', 'Slippery when wet or dirty',
    'Road narrows', 'Roadworks', 'Traffic signals ahead', 'Pedestrian crossing', 'Watch for children',
    'Bicycle crossing', 'Ice / snow', 'Wild animal crossing', 'End of all restrictions', 'Turn right ahead',
    'Turn left ahead', 'Ahead only', 'Ahead or turn right only', 'Ahead or turn left only', 'Pass by on right',
    'Pass by on left', 'Roundabout', 'End of no passing zone', 'End of no passing zone for trucks'
    ]

    class_to_name = {i: name for i, name in enumerate(classes_names)}

    parser = argparse.ArgumentParser(description="Visualize a radom example in dataset")
    parser.add_argument('--image_path', type=str, required=True, help='example image path')
    parser.add_argument('--label_path', type=str, required=True, help='example label path')
    args = parser.parse_args()

    # Pick a random image
    image_number = random.randint(0, len(os.listdir(args.image_path)) - 1)

    # Get image and label full path
    image_path = os.path.join(args.image_path, f'image #{image_number}.jpg')
    label_path = os.path.join(args.label_path, f'label #{image_number}.txt')

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image.shape[:2]

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Read labels and draw boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)

            # Convert from YOLO format to pixel coordinates
            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height

            x_min = x_center - width / 2
            y_min = y_center - height / 2

            # Draw bounding box
            rect = patches.Rectangle((x_min, y_min), width, height,linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            # Add class label above box
            class_id_int = int(class_id)
            class_name = class_to_name[class_id_int]
            ax.text(x_min, y_min - 5, class_name, fontsize=10,color='white', bbox=dict(facecolor='red'))

    # Hide axes and show result
    plt.axis('off')
    plt.title(f"Image {image_number}")
    plt.show()

if __name__ == "__main__":
    main()