from ultralytics import YOLO
import torch

def start_training():

    torch.cuda.empty_cache()

    model = YOLO('yolo11n.pt')

    model.train(
        data='dataset_para_treino/data.yaml' ,
        epochs=200,
        imgsz=512,
        batch=4,          
        device=0,         
        workers=4,        
        amp=True,         # Mixed precision (ajuda na economia de VRAM)
        exist_ok=True,
        project='DeltaV_Vision',
        name='detector',
        optimizer= 'AdamW'
    )

if __name__ == "__main__":
    start_training()