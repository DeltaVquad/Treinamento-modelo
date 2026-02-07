import os
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def run_robust_eda(data_yaml_path):
    # 1. Carregar configuração do dataset
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    classes = data_cfg['names']
    base_path = Path(data_yaml_path).parent
    
    stats = []
    
    # 2. Percorrer divisões (train, val, test)
    for split in ['train', 'val']:
        label_path = base_path / data_cfg[split].replace('images', 'labels')
        
        for label_file in label_path.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.split())
                    stats.append({
                        'split': split,
                        'class': classes[int(cls)],
                        'width': w,
                        'height': h,
                        'area': w * h
                    })

    df = pd.DataFrame(stats)

    # 3. Geração de Gráficos 
    plt.style.use('dark_background')
    accent_color = '#39d353'

    # Gráfico de Distribuição de Classes
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='class', hue='split', palette=[accent_color, '#1f77b4'])
    plt.title('Distribuição de Instâncias por Classe', color=accent_color)
    plt.savefig('eda_class_distribution.png')
    
    # Gráfico de Dispersão de Tamanho (Anchor Box Analysis)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='width', y='height', hue='class', alpha=0.5)
    plt.title('Análise de Geometria das Bounding Boxes', color=accent_color)
    plt.savefig('eda_boxes_geometry.png')

    print(f"📊 EDA Concluída. {len(df)} objetos analisados.")
    return df

if __name__ == "__main__":
    run_robust_eda('./dataset_para_treino/data.yaml')