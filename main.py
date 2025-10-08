"""
Main entry point for the asynchronous image compression script.

This module initializes the asynchronous compression of all images in a folder
and outputs them as WebP files in the target directory.

Example:
    $ python main.py
"""

import asyncio
from image_compressor import compress_folder_async

if __name__ == "__main__":
    asyncio.run(compress_folder_async("images", "compressed"))
