from ultralytics import YOLO

# Carrega o .pt original
model = YOLO('best.pt')

# Exporta com SIMPLIFY=TRUE (Isso limpa operações inúteis do grafo)
# Isso é OBRIGATÓRIO para ter boa performance no NCNN depois
model.export(format='onnx', imgsz=320, opset=12, simplify=True)