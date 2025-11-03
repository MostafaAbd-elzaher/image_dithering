import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Algorithm 1: Simple (Naïve) Threshold Dithering
# ----------------------------------------------------------------------------


def simple_threshold_dithering(image_array, threshold=128):


    # Apply threshold: if pixel >= threshold, set to 255 (white), else 0 (black)
    thresholded_image = np.where(image_array >= threshold, 255, 0)

    return thresholded_image.astype(np.uint8)

# ----------------------------------------------------------------------------
# Algorithm 2: Adaptive Threshold Dithering (Brightness Version)
# ----------------------------------------------------------------------------


def adaptive_threshold_dithering(image_array):


    # [cite_start]Compute the average intensity over the image [cite: 42]
    average_intensity = np.mean(image_array)

    print(f"Using Adaptive Threshold (Average Intensity): {average_intensity:.2f}")

    # Use the computed average as the threshold
    thresholded_image = np.where(image_array >= average_intensity, 255, 0)

    return thresholded_image.astype(np.uint8)

# ----------------------------------------------------------------------------
# Algorithm 3: Floyd-Steinberg Error Diffusion Dithering
# ----------------------------------------------------------------------------


def floyd_steinberg_dithering(image_array):


    # Create a copy of the image for processing.
    # Convert to float32 to allow for fractional error propagation.
    dithered_image = image_array.astype(np.float32)

    height, width = dithered_image.shape

    # Iterate through each pixel (top-to-bottom, left-to-right)
    for y in range(height):
        for x in range(width):
            # Get the current pixel's value (which may include diffused error)
            old_pixel = dithered_image[y, x]
            # [cite_start]1. Threshold the pixel [cite: 51]

            new_pixel = 255.0 if old_pixel >= 128.0 else 0.0
            dithered_image[y, x] = new_pixel

            # [cite_start]2. Compute the quantization error [cite: 53]
            quant_error = old_pixel - new_pixel

            # [cite_start]3. Propagate the error to neighbors [cite: 54, 62]
            # Using the Floyd-Steinberg mask weights (7/16, 3/16, 5/16, 1/16)

            # (x+1, y) -> Right
            if (x + 1) < width:
                dithered_image[y, x + 1] += quant_error * (7.0 / 16.0)

            # (x-1, y+1) -> Bottom-Left
            if (x - 1) >= 0 and (y + 1) < height:
                dithered_image[y + 1, x - 1] += quant_error * (3.0 / 16.0)

            # (x, y+1) -> Bottom
            if (y + 1) < height:
                dithered_image[y + 1, x] += quant_error * (5.0 / 16.0)

            # (x+1, y+1) -> Bottom-Right
            if (x + 1) < width and (y + 1) < height:
                dithered_image[y + 1, x + 1] += quant_error * (1.0 / 16.0)

    # Clip values to ensure they are within the 0-255 range and convert back to uint8
    dithered_image = np.clip(dithered_image, 0, 255)

    return dithered_image.astype(np.uint8)

# ----------------------------------------------------------------------------
# Main execution block
# ----------------------------------------------------------------------------


def main():

    # --- الإعداد: قم بتغيير هذا المسار إلى مسار صورتك ---
    IMAGE_PATH = 'long-exposure-shot-pier-beach-california.jpg'
    # ----------------------------------------------------

    try:
        # 1. تحميل الصورة وتحويلها إلى Grayscale
        # 'L' mode converts the image to 8-bit grayscale

        print(f"Loading image from: {IMAGE_PATH}")
        img_pil = Image.open(IMAGE_PATH).convert('L')
        img_array = np.array(img_pil)

        # --- تطبيق الخوارزميات ---
        print("Applying Naïve Threshold (T=128)...")
        naive_thresh_img = simple_threshold_dithering(img_array.copy())

        print("Applying Adaptive Threshold (T=Average)...")
        adaptive_thresh_img = adaptive_threshold_dithering(img_array.copy())

        print("Applying Floyd-Steinberg Error Diffusion...")
        fs_dither_img = floyd_steinberg_dithering(img_array.copy())

        print("Processing complete.")

        # --- عرض النتائج ---
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.title('Original Grayscale Image')
        plt.imshow(img_array, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title('Naïve Threshold (T=128)')
        plt.imshow(naive_thresh_img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title('Adaptive Threshold (T=Average)')
        plt.imshow(adaptive_thresh_img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title('Floyd-Steinberg Dithering')
        plt.imshow(fs_dither_img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')

        plt.tight_layout()
        plt.suptitle('Image Dithering Algorithm Comparison', fontsize=16)
        plt.subplots_adjust(top=0.85)
        plt.show()

        # --- حفظ النتائج ---
        Image.fromarray(naive_thresh_img).save('output_naive_threshold.png')
        Image.fromarray(adaptive_thresh_img).save('output_adaptive_threshold.png')
        Image.fromarray(fs_dither_img).save('output_floyd_steinberg.png')
        print("Output images saved successfully.")

    except FileNotFoundError:
        print("--- Error ---")
        print(f"file is not found: '{IMAGE_PATH}'")
        print("Please make sure to place a grayscale image with this name in the same folder where the code is running.")
    except Exception as e:
        print(f"error has been occurred: {e}")


if __name__ == "__main__":
    main()
