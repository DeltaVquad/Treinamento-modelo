import os
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def run_enhanced_eda(data_yaml_path):
    # 1. estetica
    plt.style.use('dark_background')
    accent_color = '#39d353'
    
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    classes = data_cfg['names']
    base_path = Path(data_yaml_path).parent
    stats = []

    # 2. Extração de Coordenadas
    for split in ['train', 'val']:
        label_path = base_path / data_cfg[split].replace('images', 'labels')
        for label_file in label_path.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.split())
                    stats.append({'split': split, 'class': classes[int(cls)], 'x': x, 'y': y, 'w': w, 'h': h})

    df = pd.DataFrame(stats)

    # 3. Gráfico 1: Mapa de Calor Espacial (Distribuição x, y)
    plt.figure(figsize=(8, 8))
    # Usando hexbin para densidade de pontos
    hb = plt.hexbin(df['x'], df['y'], gridsize=20, cmap='viridis', mincnt=1)
    plt.colorbar(hb, label='Frequência de Objetos')
    plt.title('Heatmap de Localização das Bounding Boxes (FOV)', color=accent_color, pad=20)
    plt.xlabel('Eixo X (Normalizado)')
    plt.ylabel('Eixo Y (Normalizado)')
    plt.gca().invert_yaxis() # Inverter para bater com a origem (0,0) no topo-esquerda da imagem
    plt.savefig('eda_spatial_heatmap.png', bbox_inches='tight')

    # 4. Gráfico 2: Distribuição de Tamanho Relativo
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='w', y='h', cmap="Greens", fill=True, thresh=0.05)
    plt.title('Densidade de Tamanho Relativo (Width vs Height)', color=accent_color)
    plt.savefig('eda_size_density.png')

    print(f"✅ EDA V2 Concluída. Arquivos gerados: eda_spatial_heatmap.png e eda_size_density.png")
    return df

if __name__ == "__main__":
    run_enhanced_eda('./dataset_para_treino/data.yaml')