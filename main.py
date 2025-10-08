"""
Main entry-point for the image compressor package.
"""

import asyncio
from image_compressor import compress_folder_async

if __name__ == "__main__":
    asyncio.run(compress_folder_async("images", "compressed", block_size=16))
