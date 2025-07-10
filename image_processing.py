import cv2
import numpy as np
import os
import random

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def flip_image(image, flip_code):
    flipped = cv2.flip(image, flip_code)
    return flipped

def scale_image(image, scale_factor):
    (h, w) = image.shape[:2]
    new_size = (int(w * scale_factor), int(h * scale_factor))
    scaled = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return scaled

def adjust_contrast(image, alpha):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return adjusted

def augment_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Unable to read image '{image_path}', skipping...")
            continue

        #Rotation
        angle = random.uniform(-30, 30)
        rotated_image = rotate_image(image, angle)
        output_path = os.path.join(output_folder, f"rotated_{image_file}")
        cv2.imwrite(output_path, rotated_image)

        #Flip
        flip_code = random.choice([0, 1, -1])
        flipped_image = flip_image(image, flip_code)
        output_path = os.path.join(output_folder, f"flipped_{image_file}")
        cv2.imwrite(output_path, flipped_image)

        #Scaling
        scale_factor = random.uniform(0.8, 1.2)
        scaled_image = scale_image(image, scale_factor)
        output_path = os.path.join(output_folder, f"scaled_{image_file}")
        cv2.imwrite(output_path, scaled_image)

        #Contrast
        alpha = random.uniform(0.8, 1.5)
        contrast_image = adjust_contrast(image, alpha)
        output_path = os.path.join(output_folder, f"contrast_{image_file}")
        cv2.imwrite(output_path, contrast_image)

        print(f"Processed and augmented: {image_file}")

input_folder = "E:/val_tip_burn/222"
output_folder = "E:/val_tip_burn/123"
augment_images(input_folder, output_folder)
