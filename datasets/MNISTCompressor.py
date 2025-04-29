from pathlib import Path
import argparse
from PIL import Image

def resize_folder(folder_path: Path, output_width: int, output_height: int) -> None:
    """Given input folder 'folder_path' with images of any file type, output compressed images to folder_path/MNIST_out"""
    out_dir = folder_path / "MNIST_out"
    out_dir.mkdir(exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}

    for p in folder_path.iterdir():
        if p.is_dir() or p.suffix.lower() not in extensions:
            continue 

        with Image.open(p) as img:
            resized = img.resize((output_width, output_height), Image.LANCZOS)
            resized.save(out_dir / p.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resize images to a specified dimension. Requires you to first use 'pip install Pillow'"
    )
    parser.add_argument("file_location", type=Path,
                        help="Directory that already contains the source images.")
    parser.add_argument("x_output", type=int, help="Output width in pixels.")
    parser.add_argument("y_output", type=int, help="Output height in pixels.")
    args = parser.parse_args()

    if not args.file_location.is_dir():
        raise NotADirectoryError(f"{args.file_location} is not a directory")

    resize_folder(args.file_location, args.x_output, args.y_output)
    print("Finished image resizing. Images can be found in: ", args.file_location / "MNIST_out")