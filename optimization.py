import os
import torch
from ultralytics import YOLO

MODELO_BASE = 'yolo11n.pt'
IMG_SIZE = 512
WORKERS = 0
PROJECT_NAME = 'projeto_otimizacao' # Pasta onde os testes serão salvos
TUNE_EPOCHS = 30                   
TUNE_ITERATIONS = 30                

def run_optimization():
    torch.cuda.empty_cache()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, 'dataset_para_treino', 'data.yaml')

    print(f"INICIANDO OTIMIZAÇÃO (TUNE)")
    print(f"Dataset: {DATASET_PATH}")
    
    # Carregar Modelo
    model = YOLO(MODELO_BASE)

    # Rodar o Tune
    try:
        model.tune(
            data=DATASET_PATH,
            epochs=TUNE_EPOCHS,
            iterations=TUNE_ITERATIONS,
            optimizer='AdamW',
            plots=False,
            save=False,
            val=True,
            imgsz=IMG_SIZE,
            workers=WORKERS, 
            project=PROJECT_NAME, 
            name='tune_run'
        )
        print("\nOTIMIZAÇÃO CONCLUÍDA")
        
    except Exception as e:
        print(f"\nERRO CRÍTICO no Tune: {e}")

if __name__ == '__main__':
    run_optimization()