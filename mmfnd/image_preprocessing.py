from __future__ import annotations

from io import BytesIO
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


def _flatten_transparency(
    image: Image.Image,
    background_rgb: tuple[int, int, int],
) -> Image.Image:
    """Convert every source mode to RGB without palette transparency warnings."""
    image = ImageOps.exif_transpose(image)
    has_transparency = (
        image.mode in {"P", "PA", "LA", "RGBA"}
        or "transparency" in image.info
    )
    if not has_transparency:
        return image.convert("RGB")

    # P-mode transparency may be stored as bytes. Converting to RGBA first is
    # the Pillow-supported path and avoids the palette transparency warning.
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (*background_rgb, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def _random_semantic_crop(
    image: Image.Image,
    minimum_scale: float,
    maximum_scale: float,
) -> Image.Image:
    """Mild aspect-ratio-preserving crop using the configured area range."""
    area_scale = random.uniform(minimum_scale, maximum_scale)
    if area_scale >= 0.999999:
        return image
    width, height = image.size
    side_scale = area_scale ** 0.5
    crop_width = max(1, round(width * side_scale))
    crop_height = max(1, round(height * side_scale))
    left = random.randint(0, max(0, width - crop_width))
    top = random.randint(0, max(0, height - crop_height))
    return image.crop((left, top, left + crop_width, top + crop_height))


def _jpeg_recompress(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def load_preprocessed_image(
    path: Path,
    preprocessing: dict,
    train: bool,
) -> Image.Image:
    """Apply the shared CUTE-FND image policy to one source image.

    Raw files remain unchanged. All splits receive EXIF correction,
    transparency flattening and RGB conversion. Only training samples receive
    conservative crop/brightness/contrast/JPEG augmentation. Resize, rescale
    and normalization are deliberately left to the pretrained SigLIP processor
    so all datasets use the checkpoint's exact 224x224 and mean/std convention.
    """
    color_mode = str(preprocessing.get("color_mode", "RGB")).upper()
    if color_mode != "RGB":
        raise ValueError("CUTE-FND vision encoder requires color_mode=RGB")
    background_name = str(
        preprocessing.get("transparent_background", "white")
    ).lower()
    if background_name != "white":
        raise ValueError("transparent_background currently supports white only")
    with Image.open(path) as source:
        image = _flatten_transparency(source, (255, 255, 255)).copy()

    augmentation = preprocessing.get("train_augmentation", {})
    if not train or not bool(augmentation.get("enabled", False)):
        return image

    crop_scale = augmentation.get("random_crop_scale", [0.85, 1.0])
    if not isinstance(crop_scale, (list, tuple)) or len(crop_scale) != 2:
        raise ValueError("random_crop_scale must be [min, max]")
    minimum_scale, maximum_scale = map(float, crop_scale)
    if not 0.0 < minimum_scale <= maximum_scale <= 1.0:
        raise ValueError("random_crop_scale must satisfy 0 < min <= max <= 1")
    image = _random_semantic_crop(image, minimum_scale, maximum_scale)

    brightness = float(augmentation.get("brightness", 0.0))
    contrast = float(augmentation.get("contrast", 0.0))
    if not 0.0 <= brightness < 1.0 or not 0.0 <= contrast < 1.0:
        raise ValueError("brightness and contrast must be in [0, 1)")
    if brightness > 0.0:
        image = ImageEnhance.Brightness(image).enhance(
            random.uniform(1.0 - brightness, 1.0 + brightness)
        )
    if contrast > 0.0:
        image = ImageEnhance.Contrast(image).enhance(
            random.uniform(1.0 - contrast, 1.0 + contrast)
        )

    jpeg = augmentation.get("jpeg_recompression", {})
    if not isinstance(jpeg, dict):
        raise ValueError("jpeg_recompression must be an object")
    probability = float(jpeg.get("probability", 0.0))
    quality = jpeg.get("quality", [80, 100])
    if not 0.0 <= probability <= 1.0:
        raise ValueError("jpeg_recompression.probability must be in [0, 1]")
    if not isinstance(quality, (list, tuple)) or len(quality) != 2:
        raise ValueError("jpeg_recompression.quality must be [min, max]")
    minimum_quality, maximum_quality = map(int, quality)
    if not 1 <= minimum_quality <= maximum_quality <= 100:
        raise ValueError(
            "jpeg_recompression.quality must satisfy 1 <= min <= max <= 100"
        )
    if random.random() < probability:
        image = _jpeg_recompress(
            image, random.randint(minimum_quality, maximum_quality)
        )
    return image
