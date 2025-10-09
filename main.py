"""
Main entry point for the asynchronous image compression script.

This module initializes the asynchronous compression of all images in a folder
and outputs them as WebP files in the target directory.

Now it interactively asks the user for the folder path.
If no input is provided, it defaults to the 'images' directory.

Example:
    $ python main.py
"""

import os
import asyncio
from image_compressor import compress_folder_async


async def main():
    """Interactive entry point for image compression."""
    input_folder = input("📁 Enter the path to the folder with images (default: ./images): ").strip() or "images"
    output_folder = "compressed"

    if not os.path.exists(input_folder):
        print(f"❌ The folder '{input_folder}' does not exist.")
        return

    print(f"🚀 Starting compression from: {input_folder}")
    await compress_folder_async(input_folder, output_folder)


if __name__ == "__main__":
    asyncio.run(main())
