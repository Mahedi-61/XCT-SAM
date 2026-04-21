import os
from PIL import Image

def convert_tif_to_png(input_dir, output_dir):
    """
    Converts all .tif or .tiff images in input_dir to .png format in output_dir.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".tif", ".tiff")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir, os.path.splitext(filename)[0] + ".png"
            )

            # Open and convert the image
            with Image.open(input_path) as img:
                img = img.convert("RGBA")  # ensure full color/alpha support
                img.save(output_path, "PNG")

            print(f"Converted: {filename} → {os.path.basename(output_path)}")

    print("✅ Conversion complete.")

if __name__ == "__main__":
    # Example usage
    input_directory = "input_tif"
    output_directory = "input"

    convert_tif_to_png(input_directory, output_directory)

