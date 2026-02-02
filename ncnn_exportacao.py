from ultralytics import YOLO

# 1. Carrega o seu treino original
model = YOLO('best.pt')


# Isso faz internamente: PT -> ONNX -> NCNN (FP16)
model.export(format='ncnn', imgsz=320, half=True, opset = 12, simplify = True)

print("Pronto. Pegue a pasta 'best_ncnn_model' e coloque na Raspberry Pi.")