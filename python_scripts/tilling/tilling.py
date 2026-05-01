import os
import cv2

def tile_images(input_dir, output_dir, tile_size=640, overlap=128):
    os.makedirs(output_dir, exist_ok=True)
    stride = tile_size - overlap
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        filepath = os.path.join(input_dir, filename)
        img = cv2.imread(filepath)
        if img is None:
            continue
            
        height, width, _ = img.shape
        
        # Generowanie współrzędnych X
        x_starts, x = [], 0
        while x <= width - tile_size:
            x_starts.append(x)
            x += stride
        if x_starts and x_starts[-1] < width - tile_size:
            x_starts.append(width - tile_size)
        elif not x_starts and width >= tile_size:
            x_starts.append(0)
            
        # Generowanie współrzędnych Y
        y_starts, y = [], 0
        while y <= height - tile_size:
            y_starts.append(y)
            y += stride
        if y_starts and y_starts[-1] < height - tile_size:
            y_starts.append(height - tile_size)
        elif not y_starts and height >= tile_size:
            y_starts.append(0)

        # Wycinanie i zapis kafelków
        base_name, _ = os.path.splitext(filename)
        for i, y_start in enumerate(y_starts):
            for j, x_start in enumerate(x_starts):
                tile = img[y_start : y_start + tile_size, x_start : x_start + tile_size]
                
                output_filename = f"{base_name}_R{i:03d}_C{j:03d}.jpg"
                output_filepath = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_filepath, tile)

# URUCHOMIENIE
tile_images(input_dir='datasets/unlabeled_raw', output_dir='datasets/pseudo/images')
# tile_images(input_dir='datasets/test_sacred/whole_images', output_dir='datasets/test_sacred/images')