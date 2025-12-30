import os
import random
import numpy as np
from PIL import Image
import albumentations as A

# --- CONFIGURAÇÃO DE PASTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_OBJS = os.path.join(BASE_DIR, "input_objs")
PATH_BGS = os.path.join(BASE_DIR, "input_bgs")
PATH_OUT = os.path.join(BASE_DIR, "output")

TOTAL_IMAGES = 500 

# Pipeline preservando as cores originais (Azul/Preto)
# Foco apenas em realismo ambiental (Luz e Nitidez)
transform_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.GaussNoise(noise_scale_factor=0.02, p=0.3),
    A.Perspective(scale=(0.05, 0.1), p=0.4), # Simula inclinação do drone
])

def generate():
    os.makedirs(PATH_OUT, exist_ok=True)
    
    objs = [f for f in os.listdir(PATH_OBJS) if f.lower().endswith('.png')]
    bgs = [f for f in os.listdir(PATH_BGS)]

    if not objs or not bgs:
        print("❌ Erro: Verifique as pastas input_objs e input_bgs.")
        return

    print(f"🛠️  Gerando dataset para plataforma azul ({TOTAL_IMAGES} imagens)...")

    for i in range(TOTAL_IMAGES):
        bg = Image.open(os.path.join(PATH_BGS, random.choice(bgs))).convert("RGBA")
        obj = Image.open(os.path.join(PATH_OBJS, random.choice(objs))).convert("RGBA")

        # 1. Augmentation sem alterar matriz de cor (Hue)
        obj_np = np.array(obj)
        aug = transform_pipeline(image=obj_np[:, :, :3])["image"]
        obj = Image.fromarray(np.dstack((aug, obj_np[:, :, 3])))

        # 2. Escala Dinâmica (Simula altitude do drone)
        scale = random.uniform(0.1, 0.5) 
        w, h = (int(bg.width * scale), int(bg.height * scale))
        obj = obj.resize((w, h), Image.Resampling.LANCZOS)
        
        # 3. Rotação (A plataforma pode estar em qualquer orientação no chão)
        obj = obj.rotate(random.randint(0, 360), expand=True, resample=Image.BICUBIC)

        # 4. Colagem
        max_x, max_y = (bg.width - obj.width, bg.height - obj.height)
        if max_x > 0 and max_y > 0:
            pos_x, pos_y = (random.randint(0, max_x), random.randint(0, max_y))
            bg.paste(obj, (pos_x, pos_y), obj)

        # 5. Save para anotação no CVAT
        filename = f"plataforma_azul_{i:04d}.jpg"
        bg.convert("RGB").save(os.path.join(PATH_OUT, filename), "JPEG", quality=95)

    print(f"✅ Concluído! Imagens prontas para o CVAT em: {PATH_OUT}")

if __name__ == "__main__":
    generate()
