import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import imagesize
from tqdm import tqdm
from pathlib import Path
import warnings

# Configuração Visual para Relatórios Técnicos (Estilo Paper)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
warnings.filterwarnings("ignore")

class YOLOExplorer:
    def __init__(self, data_path, class_names=None):
        """
        Inicializa o explorador de dados YOLO.
        Args:
            data_path (str): Caminho para a pasta raiz (ex: './dataset/train') contendo 'images' e 'labels'.
            class_names (list): Lista de nomes das classes (ex: ['drone', 'pessoa']).
        """
        self.data_path = Path(data_path)
        self.images_path = self.data_path / 'images'
        self.labels_path = self.data_path / 'labels'
        self.class_names = class_names
        self.df = pd.DataFrame()
        
        if not self.images_path.exists() or not self.labels_path.exists():
            raise FileNotFoundError(f"Estrutura não encontrada em {data_path}. Esperado: /images e /labels")

    def build_dataframe(self):
        """
        Lê imagens e labels, calcula métricas e constrói um DataFrame mestre.
        """
        print(f"🚀 Iniciando varredura em: {self.data_path}...")
        
        # Extensões de imagem suportadas
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        img_files = []
        for ext in img_extensions:
            img_files.extend(list(self.images_path.rglob(ext)))
            
        data = []

        for img_file in tqdm(img_files, desc="Processando Dataset"):
            # 1. Obter dimensões da imagem (rápido)
            w_img, h_img = imagesize.get(img_file)
            
            # 2. Buscar label correspondente
            label_file = self.labels_path / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                # Imagem sem label (background image)
                data.append({
                    'filename': img_file.name,
                    'img_width': w_img,
                    'img_height': h_img,
                    'has_object': False,
                    'class_id': -1,
                    'class_name': 'background',
                    'bbox_area_norm': 0,
                    'bbox_ratio': 0,
                    'center_x': None,
                    'center_y': None
                })
                continue

            # 3. Ler anotações YOLO (class x_cen y_cen w h)
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                if not lines: # Arquivo vazio = background
                    data.append({
                        'filename': img_file.name,
                        'img_width': w_img,
                        'img_height': h_img,
                        'has_object': False,
                        'class_id': -1,
                        'class_name': 'background',
                        'bbox_area_norm': 0,
                        'bbox_ratio': 0,
                         'center_x': None,
                        'center_y': None
                    })
                    continue

                for line in lines:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:])
                    
                    # Nome da classe
                    cls_name = self.class_names[cls_id] if self.class_names else str(cls_id)
                    
                    # Cálculos de Geometria
                    area_norm = w * h
                    aspect_ratio = w / h if h > 0 else 0
                    
                    data.append({
                        'filename': img_file.name,
                        'img_width': w_img,
                        'img_height': h_img,
                        'has_object': True,
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'bbox_area_norm': area_norm,
                        'bbox_ratio': aspect_ratio,
                        'center_x': x_c,
                        'center_y': y_c,
                        'bbox_w': w,
                        'bbox_h': h
                    })
            except Exception as e:
                print(f"Erro ao ler {label_file}: {e}")

        self.df = pd.DataFrame(data)
        print(f"✅ DataFrame construído com {len(self.df)} anotações.")
        return self.df

    def analyze_class_balance(self):
        """Plota a distribuição de instâncias por classe."""
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=self.df[self.df['has_object']], x='class_name', 
                           order=self.df[self.df['has_object']]['class_name'].value_counts().index,
                           palette="viridis")
        ax.set_title('Distribuição de Instâncias por Classe (Imbalance Check)')
        ax.set_ylabel('Contagem')
        ax.set_xlabel('Classe')
        plt.xticks(rotation=45)
        
        # Adicionar contagem no topo das barras
        for container in ax.containers:
            ax.bar_label(container)
            
        plt.tight_layout()
        plt.show()

    def analyze_bbox_geometry(self):
        """Analisa tamanho e proporção das caixas."""
        df_obj = self.df[self.df['has_object']]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico 1: Área Normalizada (Histograma)
        sns.histplot(df_obj['bbox_area_norm'], bins=50, kde=True, ax=axes[0], color='#39d353')
        axes[0].set_title('Distribuição da Área das BBoxes (Normalizada)')
        axes[0].set_xlabel('Área (Width * Height)')
        axes[0].set_ylabel('Frequência')
        axes[0].axvline(0.01, color='red', linestyle='--', label='Small Objects Threshold (<1%)')
        axes[0].legend()

        # Gráfico 2: Aspect Ratio (Boxplot)
        sns.boxplot(x='class_name', y='bbox_ratio', data=df_obj, ax=axes[1], palette="coolwarm")
        axes[1].set_title('Aspect Ratio (Largura / Altura) por Classe')
        axes[1].set_xlabel('Classe')
        axes[1].set_ylabel('Ratio ( >1: Wide, <1: Tall)')
        axes[1].axhline(1.0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

    def analyze_spatial_heatmap(self):
        """Heatmap de onde os objetos aparecem na imagem."""
        df_obj = self.df[self.df['has_object']]
        
        plt.figure(figsize=(8, 8))
        plt.hist2d(df_obj['center_x'], df_obj['center_y'], bins=50, cmap='inferno', range=[[0, 1], [0, 1]])
        plt.gca().invert_yaxis() # Y cresce para baixo em imagens
        plt.colorbar(label='Densidade de Objetos')
        plt.title('Mapa de Calor Espacial (Onde estão os objetos?)')
        plt.xlabel('Posição X Normalizada')
        plt.ylabel('Posição Y Normalizada')
        plt.show()

# --- Exemplo de Uso ---
if __name__ == "__main__":
    # Substitua pelo caminho do seu dataset e suas classes
    DATASET_PATH = "./dataset_para_treino/train"  # Ajuste para onde estão suas pastas
    CLASSES = ["triangulo", "estrela", "hexagono"] # Ajuste para suas classes
    
    explorer = YOLOExplorer(DATASET_PATH, CLASSES)
    explorer.build_dataframe()
    explorer.analyze_class_balance()
    explorer.analyze_bbox_geometry()
    explorer.analyze_spatial_heatmap()
    print("Script carregado. Configure DATASET_PATH e CLASSES no bloco main para rodar.")