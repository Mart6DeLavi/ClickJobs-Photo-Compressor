"""
Asynchronous Image Compressor Module.

This module provides asynchronous folder compression using `ProcessPoolExecutor`
for CPU-bound tasks. It processes images in parallel and outputs logs in real-time.

Functions:
    - compress_folder_async: Compress all images in a folder asynchronously.
    - process_image: Compress a single image using a separate process.
    - logger_worker: Reads log messages from a queue and prints them immediately.
"""

import os
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple
from .compression import adaptive_compression
from .logger_setup import setup_logger

log = setup_logger()


async def process_image(
        executor: ProcessPoolExecutor,
        input_path: str,
        output_path: str,
        start_total: float,
        log_queue: asyncio.Queue
) -> None:
    """
    Compress a single image in a separate process and send the log to a queue.

    Args:
        executor (ProcessPoolExecutor): Executor for running CPU-bound compression.
        input_path (str): Path to the source image.
        output_path (str): Path to save the compressed image.
        start_total (float): Timestamp when the compression started (for total time calculation).
        log_queue (asyncio.Queue): Queue for sending log messages to the logger worker.

    Notes:
        - Uses `adaptive_compression` for the actual compression logic.
        - The log is sent asynchronously to avoid blocking the main compression loop.
    """
    loop = asyncio.get_running_loop()
    orig_size, new_size, elapsed = await loop.run_in_executor(
        executor,
        adaptive_compression,
        input_path,
        output_path,
        start_total
    )
    await log_queue.put((input_path, orig_size, new_size, elapsed, time.time() - start_total))


async def logger_worker(log_queue: asyncio.Queue) -> None:
    """
    Asynchronous logger worker that prints logs in real-time from the queue.

    Args:
        log_queue (asyncio.Queue): Queue containing log messages from `process_image`.

    Notes:
        - Exits when `None` is put into the queue.
        - Logs include original size, new size, compression ratio, processing time, and total elapsed time.
    """
    while True:
        msg = await log_queue.get()
        if msg is None:
            break
        input_path, orig_size, new_size, processing_time, total_time = msg
        name = os.path.basename(input_path)
        ratio = 100 - (new_size / orig_size * 100) if orig_size else 0
        log.info(
            f"{name:<45} | {orig_size/1024:7.1f}KB → {new_size/1024:7.1f}KB (-{ratio:5.1f}%) "
            f"| processing: {processing_time:5.2f}s | total: {total_time:5.2f}s"
        )
        log_queue.task_done()


async def compress_folder_async(
        input_folder: str = "images",
        output_folder: str = "compressed"
) -> None:
    """
    Compress all images in a folder asynchronously using multiple CPU processes.

    Args:
        input_folder (str, optional): Source folder containing images. Defaults to "images".
        output_folder (str, optional): Destination folder to save compressed images. Defaults to "compressed".

    Notes:
        - Supports JPG, JPEG, PNG, BMP, TIFF, and WEBP formats.
        - Automatically creates necessary directories in the output folder.
        - Uses all available CPU cores for parallel processing.
        - Logs progress with aligned columns for readability.
    """
    start_total = time.time()
    image_tasks = []
    log_queue = asyncio.Queue()

    # Start logger worker
    log_task = asyncio.create_task(logger_worker(log_queue))

    # Collect images
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
        tasks = [process_image(executor, inp, out, start_total, log_queue) for inp, out in image_tasks]
        await asyncio.gather(*tasks)

    # Stop logger worker
    await log_queue.put(None)
    await log_task
