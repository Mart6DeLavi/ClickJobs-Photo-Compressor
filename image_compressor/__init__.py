"""
Image Compressor Package

Provides adaptive block-based image compression with async and parallel processing.
Includes colored console logging and file logging.
"""

from .compression import adaptive_compression
from .async_compressor import compress_folder_async

__all__ = [
    "adaptive_compression",
    "compress_folder_async",
]
