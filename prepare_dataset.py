import zipfile
import shutil
import random
from pathlib import Path
from typing import List, Tuple

# Caminho do ZIP exportado do CVAT
PATH_DATASET = "manometro_v2.zip"


def extract_zip(zip_path: str, extract_to: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def find_obj_train_data(root: Path) -> Path:
    path = next(root.rglob("obj_train_data"), None)
    if path is None:
        raise FileNotFoundError("Pasta 'obj_train_data' não encontrada no ZIP.")
    return path


def create_split_dirs(base_path: Path, splits: List[str]) -> None:
    for split in splits:
        (base_path / split / "images").mkdir(parents=True, exist_ok=True)
        (base_path / split / "labels").mkdir(parents=True, exist_ok=True)


def split_dataset(
    images: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio deve ser < 1.0")

    random.seed(seed)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train = images[:n_train]
    val = images[n_train:n_train + n_val]
    test = images[n_train + n_val:]

    return train, val, test

def copy_files(files: List[str], img_dir: Path, output_path: Path, split: str) -> None:
    img_out = output_path / split / "images"
    label_out = output_path / split / "labels"

    for filename in files:
        name = Path(filename).stem

        src_img = img_dir / filename
        dst_img = img_out / filename

        # Copiar imagem
        shutil.copy2(src_img, dst_img)

        # Copiar label correspondente
        src_label = img_dir / f"{name}.txt"
        if src_label.exists():
            dst_label = label_out / f"{name}.txt"
            shutil.copy2(src_label, dst_label)

def generate_yaml(output_path: Path, class_names: List[str]) -> None:
    yaml_content = [
        f"path: {output_path.resolve()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        "names:"
    ]

    for idx, name in enumerate(class_names):
        yaml_content.append(f"  {idx}: {name}")

    with open(output_path / "data.yaml", "w") as f:
        f.write("\n".join(yaml_content))


def load_class_names(root: Path) -> List[str]:
    names_file = next(root.rglob("obj.names"), None)

    if names_file is None or not names_file.exists():
        raise RuntimeError("Arquivo 'obj.names' não encontrado no ZIP.")

    with open(names_file) as f:
        return [line.strip() for line in f if line.strip()]


def organize_dataset(
    zip_path: str,
    output_dir: str = "dataset_para_treino",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42
) -> None:

    temp_extract = Path("temp_cvat_extract")
    output_path = Path(output_dir)

    # 1️⃣ Extrair
    extract_zip(zip_path, temp_extract)

    # 2️⃣ Encontrar pasta de imagens
    img_dir = find_obj_train_data(temp_extract)

    # 3️⃣ Criar estrutura de pastas
    splits = ["train", "val", "test"]
    create_split_dirs(output_path, splits)

    # 4️⃣ Listar imagens
    images = [
        f.name for f in img_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    # 5️⃣ Split
    train_imgs, val_imgs, test_imgs = split_dataset(
        images,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )

    # 6️⃣ Copiar arquivos
    copy_files(train_imgs, img_dir, output_path, "train")
    copy_files(val_imgs, img_dir, output_path, "val")
    copy_files(test_imgs, img_dir, output_path, "test")

    # 7️⃣ Criar YAML
    class_names = load_class_names(temp_extract)
    generate_yaml(output_path, class_names)

    # 8️⃣ Limpeza
    shutil.rmtree(temp_extract)

    print("Dataset organizado com sucesso!")
    print(f"Train: {len(train_imgs)}")
    print(f"Val:   {len(val_imgs)}")
    print(f"Test:  {len(test_imgs)}")
    print(f"Diretório final: {output_path.resolve()}")


if __name__ == "__main__":
    organize_dataset(PATH_DATASET)