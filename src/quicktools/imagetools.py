"""Image utilities: format conversion, resizing, editing, and image-to-PDF, powered by Pillow.

Supports conversion between PNG, JPEG/JPG, BMP, GIF, WEBP, TIFF, and more —
the output format is inferred automatically from the output file's extension.
"""
from PIL import Image, ImageOps, ImageDraw, ImageFont


def convert_image(input_path: str, output_path: str) -> None:
    """Convert an image from one format to another (e.g. PNG -> JPEG, JPEG -> WEBP).
    The target format is inferred from output_path's file extension."""
    img = Image.open(input_path)
    # JPEG doesn't support transparency (RGBA); flatten onto white if needed.
    if output_path.lower().endswith((".jpg", ".jpeg")) and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(output_path)


def resize_image(input_path: str, output_path: str, width: int, height: int) -> None:
    """Resize an image to the given width and height."""
    img = Image.open(input_path)
    resized = img.resize((width, height))
    resized.save(output_path)


def rotate_image(input_path: str, output_path: str, degrees: float) -> None:
    """Rotate an image by the given number of degrees (counter-clockwise), expanding the canvas as needed."""
    img = Image.open(input_path)
    rotated = img.rotate(degrees, expand=True)
    rotated.save(output_path)


def flip_image(input_path: str, output_path: str, direction: str = "horizontal") -> None:
    """Flip an image. direction is 'horizontal' or 'vertical'."""
    img = Image.open(input_path)
    if direction == "horizontal":
        flipped = ImageOps.mirror(img)
    elif direction == "vertical":
        flipped = ImageOps.flip(img)
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")
    flipped.save(output_path)


def grayscale_image(input_path: str, output_path: str) -> None:
    """Convert an image to grayscale."""
    img = Image.open(input_path).convert("L")
    img.save(output_path)


def crop_image(input_path: str, output_path: str, left: int, top: int, right: int, bottom: int) -> None:
    """Crop an image to the box defined by (left, top, right, bottom) pixel coordinates."""
    img = Image.open(input_path)
    cropped = img.crop((left, top, right, bottom))
    cropped.save(output_path)


def get_image_info(path: str) -> dict:
    """Return basic information about an image: format, dimensions, and color mode."""
    img = Image.open(path)
    return {"format": img.format, "size": img.size, "mode": img.mode}


def create_thumbnail(input_path: str, output_path: str, max_size: int = 128) -> None:
    """Create a thumbnail no larger than max_size x max_size, preserving aspect ratio."""
    img = Image.open(input_path)
    img.thumbnail((max_size, max_size))
    img.save(output_path)


def compress_image(input_path: str, output_path: str, quality: int = 70) -> None:
    """Save a (typically JPEG) image at a reduced quality level to shrink file size."""
    img = Image.open(input_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(output_path, quality=quality, optimize=True)


def merge_images_horizontally(paths: list[str], output_path: str) -> None:
    """Combine multiple images side by side into a single image."""
    images = [Image.open(p) for p in paths]
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    combined = Image.new("RGB", (total_width, max_height), "white")
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    combined.save(output_path)


def merge_images_vertically(paths: list[str], output_path: str) -> None:
    """Stack multiple images on top of each other into a single image."""
    images = [Image.open(p) for p in paths]
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    combined = Image.new("RGB", (max_width, total_height), "white")
    y_offset = 0
    for img in images:
        combined.paste(img, (0, y_offset))
        y_offset += img.height
    combined.save(output_path)


def add_text_watermark(input_path: str, output_path: str, text: str) -> None:
    """Overlay a text watermark in the bottom-right corner of an image."""
    img = Image.open(input_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    position = (img.width - text_w - 10, img.height - text_h - 10)
    draw.text(position, text, fill=(255, 255, 255, 180), font=font)
    watermarked = Image.alpha_composite(img, overlay).convert("RGB")
    watermarked.save(output_path)


def images_to_pdf(image_paths: list[str], output_pdf_path: str) -> None:
    """Combine one or more images into a single multi-page PDF file."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    if not images:
        raise ValueError("At least one image path is required")
    first, rest = images[0], images[1:]
    first.save(output_pdf_path, save_all=True, append_images=rest)

def remove_background(input_path: str, output_path: str) -> None:
    """Removes the background from an image using AI (rembg)."""
    try:
        from rembg import remove
        from PIL import Image
    except ImportError:
        raise ImportError("Background removal requires rembg and Pillow. Run: pip install rembg pillow")
    
    input_image = Image.open(input_path)
    output_image = remove(input_image)
    output_image.save(output_path, "PNG")

def compress_image(input_path: str, output_path: str, quality: int = 65) -> None:
    """Compresses an image to drastically reduce file size while maintaining acceptable quality."""
    from PIL import Image
    img = Image.open(input_path)
    # Convert RGBA (PNGs with transparency) to RGB so they can be saved as JPEGs
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(output_path, "JPEG", optimize=True, quality=quality)