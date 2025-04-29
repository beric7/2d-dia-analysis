import argparse
from PIL import Image
import pyheif
import os


def convert_heic_to_jpg(heic_file_path, jpg_file_path):
    # Read HEIC file
    heif_file = pyheif.read(heic_file_path)
    # Convert HEIF/HEIC to a Pillow image
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    # Save as JPG
    image.save(jpg_file_path, "JPEG")




def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert HEIC to JPG.')
    parser.add_argument('--path', type=str, help='Path to the input HEIC folder')
    args = parser.parse_args()
    path = args.path
    # Call the conversion function
    # Example usage
    for heic in os.listdir(path):
        heic_ = heic.split('.')[0]
        heic_file = path + heic
        jpg_file = path + heic_ + '.jpg'
        convert_heic_to_jpg(heic_file, jpg_file)
        print(f"Converted {heic_file} to {jpg_file}")

if __name__ == "__main__":
    main()
