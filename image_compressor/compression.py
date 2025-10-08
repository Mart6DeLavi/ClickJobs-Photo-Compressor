"""
Compression Module

Provides adaptive block-based image compression functions using PIL and OpenCV.
Includes block-level detail analysis and optional color quantization.
"""

import os
import io
import time
from typing import Tuple, Optional
from PIL import Image
import numpy as np
import cv2
from .logger_setup import setup_logger

log = setup_logger()

def block_detail(block: np.ndarray) -> float:
    """
    Estimate visual detail of an image block using standard deviation per RGB channel.

    Args:
        block (np.ndarray): H x W x 3 array representing an RGB block.

    Returns:
        float: Normalized detail score between 0.0 (flat) and 1.0 (very detailed).

    Example:
        >>> import numpy as np
        >>> block = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        >>> detail = block_detail(block)
    """
    if block.size == 0:
        return 0.0
    std_r = np.std(block[:, :, 0])
    std_g = np.std(block[:, :, 1])
    std_b = np.std(block[:, :, 2])
    return min(1.0, (std_r + std_g + std_b) / 150)

def compress_block(
        pil_block: Image.Image,
        quality: int,
        scale_factor: float = 1.0,
        quantize_colors: Optional[int] = None
) -> Image.Image:
    """
    Compress a single image block using WEBP with optional downscaling and color quantization.

    Args:
        pil_block (Image.Image): Pillow image block to compress.
        quality (int): Compression quality (0-100).
        scale_factor (float, optional): Scale factor for downscaling the block. Defaults to 1.0.
        quantize_colors (Optional[int], optional): Reduce the block to this number of colors if specified.

    Returns:
        Image.Image: Compressed Pillow image block.

    Notes:
        - Uses Image.LANCZOS for resizing to maintain quality.
        - Color quantization uses an adaptive palette if quantize_colors is set.
    """
    w, h = pil_block.size
    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))

    if scale_factor < 1.0:
        pil_block = pil_block.resize((new_w, new_h), Image.LANCZOS)

    if quantize_colors is not None:
        pil_block = pil_block.convert("P", palette=Image.ADAPTIVE, colors=quantize_colors).convert("RGB")

    buffer = io.BytesIO()
    pil_block.save(buffer, format="WEBP", quality=quality, method=6, lossless=False)
    buffer.seek(0)
    return Image.open(buffer)

def adaptive_compression(
        image_path: str,
        output_path: str,
        block_size: int = 16,
        scale_factor: float = 0.88,
        min_quality: int = 40,
        mid_quality: int = 65,
        max_quality: int = 90
) -> Tuple[int, int]:
    """
    Compress an entire image adaptively using block-level analysis.

    Args:
        image_path (str): Path to the input image file.
        output_path (str): Path to save the compressed image.
        block_size (int, optional): Size of each block in pixels. Defaults to 16.
        scale_factor (float, optional): Overall image downscale factor. Defaults to 0.88.
        min_quality (int, optional): Minimum quality for smooth areas. Defaults to 40.
        mid_quality (int, optional): Medium quality for moderately detailed areas. Defaults to 65.
        max_quality (int, optional): Maximum quality for highly detailed areas. Defaults to 90.

    Returns:
        Tuple[int, int]: Original file size and compressed file size in bytes.

    Notes:
        - Logs info about each image including size, compression ratio, average quality, and time taken.
        - Creates output directories automatically if they do not exist.
    """
    start_time = time.time()

    image = cv2.imread(image_path)
    if image is None:
        log.error(f"Failed to read {image_path}")
        return 0, 0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    w, h = pil_image.size
    pil_image = pil_image.resize((int(w * scale_factor), int(h * scale_factor)), Image.LANCZOS)

    compressed_img = Image.new("RGB", pil_image.size)
    total_quality = 0
    block_count = 0

    for y in range(0, pil_image.size[1], block_size):
        for x in range(0, pil_image.size[0], block_size):
            bw = min(block_size, pil_image.size[0] - x)
            bh = min(block_size, pil_image.size[1] - y)
            block = pil_image.crop((x, y, x + bw, y + bh))
            np_block = np.array(block)
            detail = block_detail(np_block)

            if detail < 0.15:
                quality = np.random.randint(min_quality, mid_quality)
                local_scale = 0.85
                quantize = 32
            elif detail < 0.5:
                quality = np.random.randint(mid_quality, max_quality)
                local_scale = 0.9
                quantize = None
            else:
                quality = np.random.randint(max_quality - 5, max_quality)
                local_scale = 1.0
                quantize = None

            compressed_block = compress_block(block, quality, local_scale, quantize)
            if compressed_block.size != (bw, bh):
                compressed_block = compressed_block.resize((bw, bh), Image.LANCZOS)
            compressed_img.paste(compressed_block, (x, y))

            total_quality += quality
            block_count += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    compressed_img.save(output_path, "WEBP", quality=max_quality, method=6, lossless=False)

    orig_size = os.path.getsize(image_path)
    new_size = os.path.getsize(output_path)
    ratio = 100 - (new_size / orig_size * 100)
    avg_quality = total_quality / block_count if block_count else 0
    elapsed = time.time() - start_time

    log.info(f"✅ {os.path.basename(image_path)}: {orig_size/1024:.1f}KB → {new_size/1024:.1f}KB (-{ratio:.1f}%), avg quality: {avg_quality:.1f}, time: {elapsed:.2f}s")

    return orig_size, new_size
