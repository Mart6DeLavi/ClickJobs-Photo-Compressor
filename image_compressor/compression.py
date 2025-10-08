"""
Adaptive Compression Module.

Provides functions to compress images using OpenCV with block scaling and WebP format.

Functions:
    - adaptive_compression: Compress an image and return sizes and elapsed processing time.
"""

import os
import time
import cv2
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
    Compress an image adaptively using OpenCV and save as WebP.

    Args:
        image_path (str): Path to the source image.
        output_path (str): Path to save the compressed WebP image.
        start_total (float, optional): Timestamp when compression started, for total time calculation.
        scale_factor (float, optional): Downscale factor for resizing the image. Defaults to 0.8.
        quality (int, optional): Compression quality for WebP (0-100). Defaults to 70.

    Returns:
        Tuple[int, int, float]: Original file size (bytes), new file size (bytes), elapsed time (seconds).

    Notes:
        - Automatically creates directories if they do not exist.
        - Measures processing time for each image.
        - Logs an error if the image cannot be read.
    """
    start_time = time.time()
    cv2.setNumThreads(cv2.getNumberOfCPUs())

    img = cv2.imread(image_path)
    if img is None:
        log.error(f"Failed to read {image_path}")
        return 0, 0, 0.0

    h, w = img.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_resized, [cv2.IMWRITE_WEBP_QUALITY, quality])

    orig_size = os.path.getsize(image_path)
    new_size = os.path.getsize(output_path)
    elapsed = time.time() - start_time
    return orig_size, new_size, elapsed
