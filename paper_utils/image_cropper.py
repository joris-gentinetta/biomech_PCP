import os

import cv2


def process_images(input_dir, output_dir, square_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)

            # Load the image in BGR format
            img_bgr = cv2.imread(img_path)

            if img_bgr is None:
                print(f"Failed to load image {img_path}")
                continue

            # Convert BGR to RGB
            # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = img_bgr

            # Get dimensions
            height, width, _ = img_rgb.shape

            # Ensure square_size is within image dimensions
            if width < square_size or height < square_size:
                # Calculate scaling factors
                scale_x = square_size / width
                scale_y = square_size / height
                scale = max(scale_x, scale_y)

                # Compute new dimensions
                new_width = int(width * scale)
                new_height = int(height * scale)

                # Resize image to ensure it is at least square_size x square_size
                img_resized = cv2.resize(
                    img_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR
                )
            else:
                img_resized = img_rgb

            # Get new dimensions
            new_height, new_width, _ = img_resized.shape

            # Calculate the cropping box
            left = (new_width - square_size) // 2
            top = (new_height - square_size) // 2
            right = left + square_size
            bottom = top + square_size

            # Crop the image
            img_cropped = img_resized[top:bottom, left:right]

            # Save the cropped image to the new directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    input_directory = "/Users/jg/Desktop/upper_limb/paper_data/movements"
    output_directory = "/Users/jg/Desktop/upper_limb/paper_data/movements_cropped"
    size_of_square = 530  # Set the size of the square

    process_images(input_directory, output_directory, size_of_square)
