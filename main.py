from ultralytics import YOLO
import cv2

# 1. Carrega o modelo
model = YOLO('plataforma_voo_v1.pt')

# 2. Testa se a câmera abre antes de rodar o YOLO
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Erro: Não consegui acessar a Logitech Brio. Verifique o cabo ou se outra aba (Chrome/Discord) está usando a câmera.")
else:
    print("✅ Câmera detectada! Iniciando inferência...")
    cap.release() # Fecha o teste para o YOLO assumir

# 3. Roda a detecção com loop manual (mais estável para debug)
results = model.predict(source="2", show=True, stream=True, conf=0.9)

for r in results:
    # Este loop mantém o programa vivo enquanto houver frames
    pass
