# Import needed packages
import albumentations
import argparse
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from roboflow import Roboflow
import subprocess

def download_signs(kaggle_username, kaggle_key):
    '''
    Downloads meta sign images with labeled backgrounds 
    from the specified kaggle datasets for use in synthetic data generation and in validation.

    Args:
        kaggle_username (str): Kaggle username used to authenticate the request.
        kaggle_key (str): Kaggle key used to authenticate the request
    '''

    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key

    datasets = [
        "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign",
        "safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb"
    ]

    for dataset in datasets:
        folder_name = dataset.split('/')[-1]  
        os.makedirs(folder_name, exist_ok=True) 
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", dataset,
            "-p", folder_name,  
            "--unzip"        
        ])

def save_example(image, label, image_path, label_path):
    '''
    Saves example image and label in given paths

    Args:
        image (numpy.ndarray) : np array representing image to be saved.              
        label (tuple): Sign bounding box attirbutes.
        image_path (str): path to save image
        label_path (str): path to save label
    '''
    
    # Extract example index and write image and label name
    example_index = len(os.listdir(image_path))
    image_filename = f'image #{example_index}.jpg'
    label_filename = f'label #{example_index}.txt'
    
    # Extract sign bounding box attributes
    sign_class, x_center, y_center, width, height = label
    
    # Save image and labels
    cv2.imwrite(os.path.join(image_path, image_filename), image)

    # Save YOLO label
    with open(os.path.join(label_path, label_filename), "w") as f:
        f.write(f"{sign_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def resize_example(image, label, size = (640, 640)):
    '''
    Resize example and label according to passed size 
    
    Args:
        image (numpy.ndarray) : np array representing image to be resized.
        label (tuple): Sign bounding box attirbutes.
        size (tuple): new example target size.
    
    Return:
        result(tuple): resized image and tuple
    '''
    # Obtain image dimensoins
    image_height, image_width = image.shape[:2]

    # Resize full synthetic image to 640x640
    resize_scale_x = size[1] / image_width
    resize_scale_y = size[0] / image_height

    x_center_px, y_center_px, bbox_width_px, bbox_height_px = label

    # Resize image
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    # Resacale and normalise
    x_center = (x_center_px * resize_scale_x) / size[1]
    y_center = (y_center_px * resize_scale_y) / size[0]
    width = (bbox_width_px * resize_scale_x) / size[1]
    height = (bbox_height_px * resize_scale_y) / size[0]

    return resized_image, (x_center, y_center, width, height)

def main():

    # Parse script arguments
    parser = argparse.ArgumentParser(description="Download background images from Roboflow")

    parser.add_argument('--roboflow_api_key', type=str, required=True, help='Roboflow API key')
    parser.add_argument('--kaggle_username', type=str, required=True, help='kaggle username')
    parser.add_argument('--kaggle_key', type=str, required=True, help='kaggle key')

    args = parser.parse_args()

    # Download backgrounds dataset from roboflow
    roboflow = Roboflow(args.roboflow_api_key)
    dataset = (
        roboflow
        .workspace("iitintern")
        .project("rdd2022-7ybsh")
        .version(3)
        .download("yolov5")
    )   

    # Download gtsdb and gtsrb datasets from kaggle
    os.environ['KAGGLE_USERNAME'] = args.kaggle_username
    os.environ['KAGGLE_KEY'] = args.kaggle_key

    datasets = [
        "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign",
        "safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb"
    ]

    for dataset in datasets:
        folder_name = dataset.split('/')[-1]  
        os.makedirs(folder_name, exist_ok=True) 
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", dataset,
            "-p", folder_name,  
            "--unzip"        
        ])

    # Create full yolo dataset folder
    os.makedirs("Dataset", exist_ok=True)

    os.makedirs("Dataset/images", exist_ok=True)
    os.makedirs("Dataset/images/train", exist_ok=True)
    os.makedirs("Dataset/images/val", exist_ok=True)
    os.makedirs("Dataset/images/test", exist_ok=True)

    os.makedirs("Dataset/labels", exist_ok=True)
    os.makedirs("Dataset/labels/train", exist_ok=True)
    os.makedirs("Dataset/labels/val", exist_ok=True)
    os.makedirs("Dataset/labels/test", exist_ok=True)

    # Store current directory
    current_directory = os.getcwd()

    # Create synthetic images for train-set and validation-set
    # this is done by applying transformations to sign then pasting sign to background image
    # after selecting random upperleft corner coordinates for pasting the image
    # Signs are taken from (gtsrb-german-traffic-sign) dataset and background
    # are taken from roboflow

    # Read meta sign images
    df = pd.read_csv(
        os.path.join(
            current_directory,
            "gtsrb-german-traffic-sign/Meta.csv",
        )
    )

    # Store signs path and backgrounds path
    signs_path = os.path.join(current_directory, 'gtsrb-german-traffic-sign')
    backgrounds_path = os.path.join(current_directory, 'RDD2022-3/train/images')

    backgrounds = [ background for background in os.listdir(backgrounds_path)]
    random.shuffle(backgrounds)

    image_output_path = None
    label_output_path = None

    for background_index in range(600):
        for sign_class, sign in zip(df.ClassId, df.Path):

            if background_index < 480:
                image_output_path = os.path.join(current_directory, 'Dataset/images/train')
                label_output_path = os.path.join(current_directory, 'Dataset/labels/train')
            else:
                image_output_path = os.path.join(current_directory, 'Dataset/images/val')
                label_output_path = os.path.join(current_directory, 'Dataset/labels/val')

            # Get a random scaling factor
            scale = random.uniform(0.5, 1.5)

            # Augmentaion pipline
            transform = albumentations.Compose(
                [
                    # Apply gaussian blur to sign
                    albumentations.GaussianBlur(blur_limit=3, p=0.5),
                    # change brightness and contrast of sign
                    albumentations.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.5
                    ),
                    # Apply perspective transformation to sign
                    albumentations.Perspective(
                        scale=(0.02, 0.1), fit_output=True, keep_size=True, p=0.5
                    ),
                    # Resize sign
                    albumentations.Resize(int(100 * scale), int(100 * scale)),
                    # Apply rotation to sign
                    albumentations.Affine(rotate=(-25, 25), fit_output=True),
                ]
            )

            # Write background and sign paths
            background_path = os.path.join(backgrounds_path, backgrounds[background_index])
            sign_path = os.path.join(signs_path, sign)

            # Load background and sign images
            background_image = cv2.imread(background_path)
            sign_image = cv2.imread(sign_path, cv2.IMREAD_UNCHANGED)

            # Apply augmentation to sign
            augmented_sign = transform(image=sign_image)
            sign_image = augmented_sign["image"]

            sign_height, sign_width = sign_image.shape[:2]
            background_height, background_width = background_image.shape[:2]

            # Generate random upperleft corner to paste sign
            upperleft_x = random.randint(0, background_width - sign_width)
            upperleft_y = random.randint(0, background_height - sign_height)

            # Paste sign onto background
            alpha_sign = sign_image[:, :, 3] / 255.0
            alpha_background = 1.0 - alpha_sign
            for c in range(3):
                background_image[
                    upperleft_y : upperleft_y + sign_height,
                    upperleft_x : upperleft_x + sign_width,
                    c,
                ] = (
                    alpha_sign * sign_image[:, :, c]
                    + alpha_background
                    * background_image[
                        upperleft_y : upperleft_y + sign_height,
                        upperleft_x : upperleft_x + sign_width,
                        c,
                    ]
                )

            # Scale bounding box coordinates accordingly
            x_center_px = upperleft_x + sign_width / 2
            y_center_px = upperleft_y + sign_height / 2
            bbox_width_px = sign_width
            bbox_height_px = sign_height

            label = (x_center_px, y_center_px, bbox_width_px, bbox_height_px)

            image, label = resize_example(background_image, label, size = (640, 640))
            label = (sign_class, ) + label

            save_example(image, label, image_output_path, label_output_path)

    # Create test-dataset
    images_path = os.path.join(
        current_directory,
        "german-traffic-sign-detection-benchmark-gtsdb/TrainIJCNN2013/TrainIJCNN2013",
    )

    test_image_output_path = os.path.join(current_directory, 'Dataset/images/test')
    test_label_output_path = os.path.join(current_directory, 'Dataset/labels/test')

    with open(os.path.join(images_path, 'gt.txt'), 'r') as file:
        for line in file:
            image_name, left, top, right, bottom, sign_class = line.strip().split(';')
            left, top, right, bottom = map(float, [left, top, right, bottom])

            # Load image
            image_path = os.path.join(images_path, image_name)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            # Convert bounding box coordinates to yolo coordinates
            bbox_width_px = right - left
            bbox_height_px = bottom - top
            x_center_px = left + bbox_width_px / 2
            y_center_px = top + bbox_height_px / 2

            label = (x_center_px, y_center_px, bbox_width_px, bbox_height_px)

            image, label = resize_example(image, label, size = (640, 640))
            label = (sign_class, ) + label

            save_example(image, label, test_image_output_path, test_label_output_path)

if __name__ == "__main__":
    main()
