import cv2
import numpy as np
import os
from tqdm import tqdm

def process_images_in_folder(folder_path):
    """
    Calculates the overall mean and standard deviation of RGB channels 
    across all images in a folder.

    Args:
        folder_path: Path to the folder containing the images.

    Returns:
        A dictionary containing the overall mean and standard deviation for each channel (R, G, B),
        or None if no images are found or there's an error.
        Example: {'mean': (r_mean, g_mean, b_mean), 'std': (r_std, g_std, b_std)}
    """

    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        print(f"No image files found in {folder_path}")
        return None

    all_pixels = []  # List to store all pixel values from all images

    for image_file in tqdm(image_files, total=len(image_files)):
        image_path = os.path.join(folder_path, image_file)
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image at {image_path}")
                continue  # Skip to the next image

            # Reshape the image to (number_of_pixels, 3) to get all RGB values
            pixels = img.reshape(-1, 3)
            all_pixels.extend(pixels)

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    if not all_pixels:  # Check if any pixels were actually collected
        print("No image data processed. Check image files and paths.")
        return None

    all_pixels = np.array(all_pixels) #convert to numpy array for efficiency
    overall_mean = np.mean(all_pixels, axis=0)
    overall_std = np.std(all_pixels, axis=0)

    return {'mean': overall_mean, 'std': overall_std}

if __name__ == "__main__":
    folder_path = "/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/semi_supervise/processed_format/real_testing"  # Replace with your folder path
    overall_stats = process_images_in_folder(folder_path)

    if overall_stats:
        print("Overall Statistics for all images:")
        print(f"  Mean (B, G, R): {overall_stats['mean'] / 255.}")
        print(f"  Std Dev (B, G, R): {overall_stats['std'] / 255.}")
    else:
        print("Could not calculate overall statistics.")