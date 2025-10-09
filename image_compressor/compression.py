"""
Compression Module for Asynchronous Image Processing.

This module provides the core image compression functionality used by
the asynchronous image compressor. It includes an adaptive and optimized
compression function that:

- Reads large images efficiently using OpenCV's reduced color loading.
- Encodes images to WebP in memory using `cv2.imencode` (faster than imwrite).
- Minimizes memory copies between numpy arrays and OpenCV.
- Supports downscaling via `scale_factor` to reduce size further.
- Ensures compatibility with ARM-based Macs (M1/M2/M3) and leverages NEON vectorization.

The main function:
    - adaptive_compression: Compresses a single image efficiently.

Usage:
    from compression import adaptive_compression

    orig_size, new_size, elapsed = adaptive_compression(
        "images/photo.jpg",
        "compressed/photo.webp",
        scale_factor=0.8,
        quality=70
    )
"""

import os
import time
import cv2
import numpy as np
from typing import Tuple
from .logger_setup import setup_logger

log = setup_logger()


def adaptive_compression(
        image_path: str,
        output_path: str,
        start_total: float = None,
        scale_factor: float = 0.8,
        quality: int = 70
) -> Tuple[int, int, float]:
    """
    Adaptive and optimized image compression using OpenCV.

    This function compresses a single image to WebP format with optional downscaling.
    It is designed for high performance on both Intel and ARM architectures.

    Features:
        - Fast loading of large images using cv2.IMREAD_REDUCED_COLOR_*.
        - Optional downscaling to reduce memory usage and speed up processing.
        - In-memory WebP encoding with `cv2.imencode` (faster than writing via cv2.imwrite).
        - Automatic directory creation for output path.
        - Detailed logging on read/encode failures.

    Parameters:
        image_path (str): Absolute or relative path to the source image.
        output_path (str): Absolute or relative path where the compressed WebP image will be saved.
        start_total (float, optional): Timestamp marking start of total operation for logging purposes.
        scale_factor (float, optional): Factor to downscale the image before compression (0 < scale_factor <= 1.0).
                                        Default is 0.8 (reduce size to 80%).
        quality (int, optional): WebP compression quality (0-100). Higher means better quality but larger size.
                                 Default is 70.

    Returns:
        Tuple[int, int, float]: A tuple containing:
            - original_size_bytes (int): Size of the input image in bytes.
            - new_size_bytes (int): Size of the compressed WebP image in bytes.
            - elapsed_time_seconds (float): Time taken to compress this image.

    Raises:
        Logs errors instead of raising exceptions for:
            - Failure to read the image.
            - Failure to encode the image.
            - Any other unexpected exception during processing.

    Example:
        orig_size, new_size, elapsed = adaptive_compression(
            "images/photo.jpg",
            "compressed/photo.webp",
            scale_factor=0.75,
            quality=80
        )
        log.info(f"Compressed {orig_size} → {new_size} in {elapsed:.2f}s")
    """
    start_time = time.time()

    try:
        # Determine original file size
        orig_size = os.path.getsize(image_path)

        # Choose fast loading mode for large images (>5MB)
        read_flag = cv2.IMREAD_COLOR
        if orig_size > 5 * 1024 * 1024:
            read_flag = cv2.IMREAD_REDUCED_COLOR_2  # Load at half resolution

        # Load the image
        img = cv2.imread(image_path, read_flag)
        if img is None:
            log.error(f"Failed to read {image_path}")
            return 0, 0, 0.0

        # Apply optional downscaling
        if scale_factor < 1.0:
            h, w = img.shape[:2]
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Encode to WebP in memory
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        success, encoded_img = cv2.imencode(".webp", img, encode_params)
        if not success:
            log.error(f"Failed to encode {image_path}")
            return orig_size, 0, 0.0

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write encoded bytes directly to disk
        with open(output_path, "wb") as f:
            encoded_img.tofile(f)

        # Get new file size
        new_size = os.path.getsize(output_path)
        elapsed = time.time() - start_time

        return orig_size, new_size, elapsed

    except Exception as e:
        log.error(f"Error processing {image_path}: {e}")
        return 0, 0, 0.0
