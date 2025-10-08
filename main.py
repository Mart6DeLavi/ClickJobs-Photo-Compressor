import os
import time
import io
from PIL import Image
import numpy as np
import cv2

def block_detail(block: np.ndarray) -> float:
    """Оцениваем детализацию блока: std отклонение по RGB"""
    if block.size == 0:
        return 0.0
    std_r = np.std(block[:, :, 0])
    std_g = np.std(block[:, :, 1])
    std_b = np.std(block[:, :, 2])
    return min(1.0, (std_r + std_g + std_b) / 150)

def compress_block(pil_block: Image.Image, quality: int, scale_factor: float = 1.0, quantize_colors: int = None) -> Image.Image:
    """Сжимаем блок с защитой от слишком маленького размера"""
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


def adaptive_compression(image_path: str, output_path: str, block_size=16, scale_factor=0.88,
                         min_quality=40, mid_quality=65, max_quality=90):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Не удалось прочитать {image_path}")
        return 0, 0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    # 🔹 Масштабирование всего изображения
    w, h = pil_image.size
    pil_image = pil_image.resize((int(w*scale_factor), int(h*scale_factor)), Image.LANCZOS)

    compressed_img = Image.new("RGB", pil_image.size)
    total_quality = 0
    block_count = 0

    for y in range(0, pil_image.size[1], block_size):
        for x in range(0, pil_image.size[0], block_size):
            bw = min(block_size, pil_image.size[0]-x)
            bh = min(block_size, pil_image.size[1]-y)
            block = pil_image.crop((x, y, x+bw, y+bh))
            np_block = np.array(block)
            detail = block_detail(np_block)

            # 🔹 Адаптивное качество и локальный scale
            if detail < 0.15:
                quality = np.random.randint(min_quality, mid_quality)      # гладкие области
                local_scale = 0.85
                quantize = 32
            elif detail < 0.5:
                quality = np.random.randint(mid_quality, max_quality)      # средние
                local_scale = 0.9
                quantize = None
            else:
                quality = np.random.randint(max_quality-5, max_quality)    # детализированные
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
    print(f"✅ {os.path.basename(image_path)}: {orig_size/1024:.1f}KB → {new_size/1024:.1f}KB (-{ratio:.1f}%), avg quality: {avg_quality:.1f})")
    return orig_size, new_size

def compress_folder(input_folder="images", output_folder="compressed", block_size=16):
    total_orig = 0
    total_new = 0
    total_files = 0
    start_time = time.time()

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff','.webp')):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, rel_path)
                output_path = os.path.join(output_dir, os.path.splitext(file)[0]+".webp")
                orig, new = adaptive_compression(input_path, output_path, block_size)
                total_orig += orig
                total_new += new
                total_files += 1

    elapsed = time.time() - start_time
    total_saved = 100 - (total_new / total_orig * 100) if total_orig > 0 else 0
    print("\n📊 Сводка оптимизации")
    print("──────────────────────────────")
    print(f"Всего файлов:        {total_files}")
    print(f"Исходный объём:      {total_orig/1024/1024:.2f} MB")
    print(f"После сжатия:        {total_new/1024/1024:.2f} MB")
    print(f"Общая экономия:      {total_saved:.1f}%")
    print(f"Общее время:         {elapsed:.2f} сек")

if __name__ == "__main__":
    compress_folder("images", "compressed", block_size=16)
