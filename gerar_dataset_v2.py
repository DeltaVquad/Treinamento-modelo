import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import albumentations as A

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_OBJS = os.path.join(BASE_DIR, "input_objs")
PATH_BGS = os.path.join(BASE_DIR, "input_bgs")
PATH_OUT = os.path.join(BASE_DIR, "output")

TOTAL_IMAGES = 500
MAX_OBJECTS_PER_IMAGE = 4
MIN_SCALE = 0.08
MAX_SCALE_LARGE_BG = 0.35
MAX_SCALE_SMALL_BG = 0.25

# ========== BACKGROUND AUGMENT ==========
bg_augment = A.Compose([
    A.RandomBrightnessContrast(0.25, 0.25, p=0.8),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.2, 0.4), p=0.4),
    A.MotionBlur(blur_limit=(3, 9), p=0.3),
    A.RandomShadow(
        shadow_roi=(0, 0.5, 1, 1),
        num_shadows_lower=1,
        num_shadows_upper=3,
        shadow_dimension=6,
        p=0.4
    ),
])

# ========== GEOMETRIA / COLISÃO ==========
def intersects(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (
        x1_max < x2_min or
        x1_min > x2_max or
        y1_max < y2_min or
        y1_min > y2_max
    )

def find_valid_position(bg_size, obj_size, placed_boxes, margin=15, max_tries=80):
    bg_w, bg_h = bg_size
    obj_w, obj_h = obj_size

    for _ in range(max_tries):
        x = random.randint(0, bg_w - obj_w)
        y = random.randint(0, bg_h - obj_h)

        new_box = (
            x - margin,
            y - margin,
            x + obj_w + margin,
            y + obj_h + margin
        )

        if not any(intersects(new_box, box) for box in placed_boxes):
            return (x, y), new_box

    return None, None

# ========== SOMBRA ==========
def add_shadow(bg, obj, pos):
    shadow = obj.copy().convert("RGBA")
    shadow = shadow.point(lambda p: 0 if p < 10 else 120)
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))

    dx = random.randint(15, 50)
    dy = random.randint(15, 50)

    bg.paste(shadow, (pos[0] + dx, pos[1] + dy), shadow)

# ========== OCLUSÃO ==========
def random_occlusion(img):
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(1, 2)):
        x1 = random.randint(0, img.width)
        y1 = random.randint(0, img.height)
        x2 = x1 + random.randint(30, 100)
        y2 = y1 + random.randint(30, 100)
        draw.rectangle(
            [x1, y1, x2, y2],
            fill=(0, 0, 0, random.randint(40, 100))
        )

# ================= MAIN =================
def generate():
    os.makedirs(PATH_OUT, exist_ok=True)

    objs = [f for f in os.listdir(PATH_OBJS) if f.lower().endswith(".png")]
    bgs = [f for f in os.listdir(PATH_BGS)]

    if not objs or not bgs:
        print("❌ Erro: verifique input_objs e input_bgs.")
        return

    print(f"🛠 Gerando {TOTAL_IMAGES} imagens sintéticas...")

    for i in range(TOTAL_IMAGES):
        # ---------- BACKGROUND ----------
        bg = Image.open(
            os.path.join(PATH_BGS, random.choice(bgs))
        ).convert("RGBA")

        bg_np = np.array(bg.convert("RGB"))
        bg_np = bg_augment(image=bg_np)["image"]
        bg = Image.fromarray(bg_np).convert("RGBA")

        placed_boxes = []
        num_objects = random.randint(1, MAX_OBJECTS_PER_IMAGE)

        # ---------- OBJECTS ----------
        for _ in range(num_objects):
            obj = Image.open(
                os.path.join(PATH_OBJS, random.choice(objs))
            ).convert("RGBA")

            max_scale = (
                MAX_SCALE_LARGE_BG
                if bg.width > 1000 else
                MAX_SCALE_SMALL_BG
            )
            scale = random.uniform(MIN_SCALE, max_scale)

            w = int(bg.width * scale)
            h = int(w * obj.height / obj.width)
            obj = obj.resize((w, h), Image.Resampling.LANCZOS)

            # compressão de perspectiva
            if random.random() < 0.4:
                obj = obj.resize(
                    (obj.width, int(obj.height * random.uniform(0.75, 0.9))),
                    Image.Resampling.BICUBIC
                )

            obj = obj.rotate(
                random.randint(0, 360),
                expand=True,
                resample=Image.BICUBIC
            )

            pos, bbox = find_valid_position(
                (bg.width, bg.height),
                (obj.width, obj.height),
                placed_boxes
            )

            if pos is None:
                continue

            if random.random() < 0.6:
                add_shadow(bg, obj, pos)

            bg.paste(obj, pos, obj)
            placed_boxes.append(bbox)

        # ---------- OCLUSÃO FINAL ----------
        if random.random() < 0.3:
            random_occlusion(bg)

        # ---------- SAVE ----------
        filename = f"scene_{i:04d}.jpg"
        bg.convert("RGB").save(
            os.path.join(PATH_OUT, filename),
            "JPEG",
            quality=random.randint(85, 95)
        )

    print(f"✅ Dataset final salvo em: {PATH_OUT}")

if __name__ == "__main__":
    generate()
