from PIL import Image, ImageDraw, ImageFont
import random

# Assign consistent colors to each class
def get_color_map(names):
    unique_names = list(set(names))
    random.seed(42)  # ensures consistent color mapping
    return {name: tuple(random.choices(range(50, 256), k=3)) for name in unique_names}

def draw_annotations(img_path, names, confs, boxes):
    # Load image
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Font
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        font = ImageFont.load_default()

    # Ensure tensors are on CPU and converted to list
    confs = confs.cpu().tolist()
    boxes = boxes.cpu().tolist()

    # Color map per class name
    color_map = get_color_map(names)

    # Draw boxes and labels
    for name, conf, box in zip(names, confs, boxes):
        x1, y1, x2, y2 = box
        label = f"{name} {conf:.2f}"
        color = color_map[name]

        # Draw thicker lines (multiple rectangles offset by 1px)
        for offset in range(5):  # adjust range for thicker lines
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color
            )

        # Draw label
        draw.text((x1, y1 - 30), label, fill=color, font=font)

    return img
