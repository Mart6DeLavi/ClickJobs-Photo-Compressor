"""
Adaptive Image Compression Tool with Async and Parallel Processing

This module provides functionality to compress images efficiently using
adaptive block-based compression, while leveraging multi-core CPUs and
asynchronous execution for speed. Logs are enhanced with colored levelnames
in the console, and full logs are saved to a file.

Features:
- Adaptive compression per block (smooth areas compressed more aggressively)
- Downscaling and optional color quantization per block
- Parallel processing across CPU cores using ProcessPoolExecutor
- Async orchestration with asyncio
- Colored console logs (INFO: green, WARNING: yellow, ERROR: red)
- File logging without colors
- Logs per image include size before/after, compression ratio, average block quality, and processing time

Dependencies:
- Python 3.8+
- Pillow
- OpenCV (cv2)
- numpy
"""

import os
import io
import time
import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import numpy as np
import cv2

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────

log = logging.getLogger("ImageCompressor")
log.setLevel(logging.INFO)

class LevelColorFormatter(logging.Formatter):
    """
    Formatter that colors only the levelname in console logs:
        DEBUG    -> gray
        INFO     -> green
        WARNING  -> yellow
        ERROR    -> red
        CRITICAL -> magenta
    Other text remains standard terminal color.
    """
    COLORS = {
        'DEBUG': '\033[90m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[95m'
    }
    RESET = '\033[0m'

    def format(self, record):
        original_levelname = record.levelname
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        formatted = super().format(record)
        record.levelname = original_levelname
        return formatted

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(LevelColorFormatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
))

# File handler
file_handler = logging.FileHandler("compressor.log", mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))

log.addHandler(console_handler)
log.addHandler(file_handler)

# ──────────────────────────────────────────────
# Core compression logic
# ──────────────────────────────────────────────

def block_detail(block: np.ndarray) -> float:
    """
    Estimate visual detail of an image block.

    Args:
        block (np.ndarray): H x W x 3 array representing an RGB block.

    Returns:
        float: Normalized detail score (0.0 - 1.0)
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
        quantize_colors: int = None
) -> Image.Image:
    """
    Compress a single image block.

    Args:
        pil_block (Image.Image): Pillow image block
        quality (int): WEBP compression quality (0-100)
        scale_factor (float): Local resizing factor
        quantize_colors (int | None): Reduce block to N colors if provided

    Returns:
        Image.Image: Compressed Pillow block
    """
    w, h = pil_block.size
    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))

    if scale_factor < 1.0:
        pil_block = pil_block.resize((new_w, new_h), Image.LANCZOS)

    if quantize_colors is not None:
        pil_block = (
            pil_block.convert("P", palette=Image.ADAPTIVE, colors=quantize_colors)
            .convert("RGB")
        )

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
        max_quality: int = 90,
):
    """
    Compress a single image using adaptive block-level compression.

    Args:
        image_path (str): Source image path
        output_path (str): Destination path for compressed image
        block_size (int): Block size in pixels
        scale_factor (float): Downscale factor for full image
        min_quality (int): Minimum block quality for smooth areas
        mid_quality (int): Mid-level quality for medium blocks
        max_quality (int): Maximum quality for detailed blocks

    Returns:
        tuple[int, int]: Original file size and compressed file size (bytes)
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

    log.info(
        f"✅ {os.path.basename(image_path)}: {orig_size/1024:.1f}KB → "
        f"{new_size/1024:.1f}KB (-{ratio:.1f}%), avg quality: {avg_quality:.1f}, time: {elapsed:.2f}s"
    )

    return orig_size, new_size


# ──────────────────────────────────────────────
# Async orchestration
# ──────────────────────────────────────────────

async def process_image(executor, input_path, output_path, block_size):
    """
    Run adaptive_compression() asynchronously in a process pool.

    Args:
        executor (ProcessPoolExecutor): Executor for parallel CPU-bound tasks
        input_path (str): Source image path
        output_path (str): Compressed image path
        block_size (int): Block size for compression

    Returns:
        tuple[int, int]: Original and compressed file size
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, adaptive_compression, input_path, output_path, block_size)


async def compress_folder_async(input_folder="images", output_folder="compressed", block_size=16):
    """
    Compress all images in a folder asynchronously with multiple processes.

    Args:
        input_folder (str): Directory containing source images
        output_folder (str): Directory for compressed images
        block_size (int): Block size for compression
    """
    image_tasks = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, rel_path)
                output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".webp")
                image_tasks.append((input_path, output_path))

    max_workers = os.cpu_count() or 8
    log.info(f"🧠 Using {max_workers} parallel processes for compression")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [process_image(executor, inp, out, block_size) for inp, out in image_tasks]
        await asyncio.gather(*tasks)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(compress_folder_async("images", "compressed", block_size=16))
