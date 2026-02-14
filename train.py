import os
import yaml
import torch
from ultralytics import YOLO

from optimization import run_optimization

MODELO_BASE = 'yolo11n.pt'
IMG_SIZE = 512
WORKERS = 0
PROJECT_NAME_OPT = 'projeto_otimizacao' 
PROJECT_NAME_FINAL = 'DeltaV_Vision'    
FINAL_EPOCHS = 200
PATIENCE = 50


NOME_DA_PASTA_TUNE = 'tune_run' # alterar conforme o nome do arquivo dos best_param

def start_final_training():
    torch.cuda.empty_cache()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, 'dataset_para_treino', 'data.yaml')
    
    path_tune_escolhido = os.path.join(BASE_DIR, PROJECT_NAME_OPT, NOME_DA_PASTA_TUNE)
    yaml_path = os.path.join(path_tune_escolhido, 'best_hyperparameters.yaml')

    print(f"--- PREPARANDO TREINO FINAL ---")

    best_params = {}

    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            best_params = yaml.safe_load(f) or {}
            
        # Correção de Float -> Int 
        if best_params:
            integers_keys = ['close_mosaic', 'epochs', 'batch', 'warmup_epochs', 'copy_paste']
            for k in integers_keys:
                if k in best_params:
                    best_params[k] = int(best_params[k])
        else:
            print("AVISO: O arquivo yaml estava vazio. Usando padrões.")
    else:
        print(f"\nERRO: O arquivo não foi encontrado em '{NOME_DA_PASTA_TUNE}'.")
        print("Verifique se o nome da pasta está correto nas Configurações Gerais.")
        print("Rodando com parâmetros PADRÃO (sem otimização).")

    # Treino Final
    print(f"\n--- INICIANDO TREINAMENTO: {PROJECT_NAME_FINAL} ---")
    
    model = YOLO(MODELO_BASE)

    model.train(
        data=DATASET_PATH,
        epochs=FINAL_EPOCHS,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        batch=5,           
        device=0,
        workers=WORKERS,
        project=PROJECT_NAME_FINAL,
        name='detector_final',
        cos_lr=True,        
        optimizer='AdamW',
        **best_params       # Injeta os melhores parâmetros 
    )

if __name__ == "__main__":
    start_final_training()