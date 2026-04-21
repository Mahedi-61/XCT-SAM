import os
import csv

def generate_image_label_csv(input_dir, label_dir, output_csv):
    """
    Generate a CSV with columns [image, label] by pairing files in input_dir and label_dir.
    """


    # Collect and sort file lists
    input_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".png"))
    label_files = sorted(f for f in os.listdir(label_dir) if f.lower().endswith(".png"))

    # Sanity check: Ensure matching filenames exist
    if len(input_files) != len(label_files):
        print("Warning: Different number of images and labels. Some may be missing.")

    # Write the CSV
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "label"])

        for img_file in input_files:
            label_path = os.path.join(label_dir, img_file)
            if os.path.exists(label_path):
                writer.writerow([
                    os.path.join(input_dir, img_file),
                    label_path
                ])
            else:
                print(f"No matching label found for {img_file}")

    print(f"CSV file created: {output_csv}")

if __name__ == "__main__":
    input_dir = "test6/input"
    label_dir = "test6/class_1"
    output_csv = "test6_class_1.csv"
    generate_image_label_csv(input_dir, label_dir, output_csv)

