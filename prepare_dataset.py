import os
import zipfile
import shutil
import random
from pathlib import Path

# mude o caminho para o zip dataset exportar do CVAT
PATH_DATASET = "numeros_v3.zip"

def organize_dataset(zip_path, output_dir, train_ratio=0.8):
    # extração
    temp_extract = "temp_cvat_extract"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)

    # setup de caminhos
    output_path = Path(output_dir)
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # listar arquivos (assumindo que imagens e labels têm o mesmo nome)
    search_path = list(Path(temp_extract).rglob('obj_train_data'))
    
    if not search_path:
        raise FileNotFoundError("Pasta 'obj_train_data' não encontrada dentro do ZIP.")
        
    img_dir = search_path[0]
    
    images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    def move_files(files, split):
        for f in files:
            name = Path(f).stem
            # mover Imagem
            shutil.copy(img_dir / f, output_path / split / 'images' / f)
            # mover Label (.txt)
            label_file = f"{name}.txt"
            if (img_dir / label_file).exists():
                shutil.copy(img_dir / label_file, output_path / split / 'labels' / label_file)

    move_files(train_imgs, 'train')
    move_files(val_imgs, 'val')

    # Procura o arquivo obj.names em qualquer subpasta
    names_file = next(Path(temp_extract).rglob('obj.names'), None)
    
    if not names_file or not names_file.exists():
        raise RuntimeError("Arquivo de classes (obj.names) não encontrado no ZIP.")

    yaml_path = output_path / 'data.yaml'

    if not names_file.exists():
        raise RuntimeError("obj.names não encontrado")

    with open(names_file) as f:
        class_names = [line.strip() for line in f if line.strip()]

    yaml_lines = [
        f"path: {output_path.resolve()}",
        "train: train/images",
        "val: val/images",
        "",
        "names:"
    ]

    for i, name in enumerate(class_names):
        yaml_lines.append(f"  {i}: {name}")

    with open(output_path / "data.yaml", "w") as f:
        f.write("\n".join(yaml_lines))

    print("data.yaml criado com sucesso")

    # limpeza
    shutil.rmtree(temp_extract)
    print(f"Dataset organizado em: {output_dir}")

if __name__ == "__main__":
    organize_dataset(PATH_DATASET, 'dataset_para_treino')