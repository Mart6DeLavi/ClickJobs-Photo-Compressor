"""
Async Compressor Module

Provides asynchronous folder compression using ProcessPoolExecutor for CPU-bound tasks.
"""

import os
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple
from .compression import adaptive_compression
from .logger_setup import setup_logger

log = setup_logger()

async def process_image(
        executor: ProcessPoolExecutor,
        input_path: str,
        output_path: str,
        block_size: int
) -> Tuple[int, int]:
    """
    Run adaptive_compression asynchronously in a separate process.

    Args:
        executor (ProcessPoolExecutor): Executor for CPU-bound tasks.
        input_path (str): Path to the input image.
        output_path (str): Path to save the compressed image.
        block_size (int): Block size for compression.

    Returns:
        Tuple[int, int]: Original and compressed file sizes in bytes.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, adaptive_compression, input_path, output_path, block_size)

async def compress_folder_async(
        input_folder: str = "images",
        output_folder: str = "compressed",
        block_size: int = 16
):
    """
    Compress all images in a folder asynchronously using multiple CPU processes.

    Args:
        input_folder (str): Directory containing images to compress.
        output_folder (str): Directory to save compressed images.
        block_size (int): Block size in pixels.

    Notes:
        - Uses all available CPU cores for parallel processing.
        - Logs progress for each image including size, compression ratio, average quality, and time taken.
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
