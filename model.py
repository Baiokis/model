import os
import torch
from ultralytics import YOLO  # type: ignore

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo em uso: {device}")

model = YOLO('yolov8n.pt')

def train_model():
    try:
        model.train(
            data='PPEs-8/data.yaml',
            epochs=125,
            imgsz=800,
            batch=16,
            lr0=0.0005,
            lrf=0.01,
            freeze=10,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.5,
            translate=0.1,
            scale=0.5,
            shear=0.5,
            flipud=0.5,
            fliplr=0.5,
            save_period=50,
            val=True,
            device=device
        )
        print("Treinamento concluído.")
    except Exception as e:
        print(f"Erro no treinamento: {e}")

def validate_model():
    try:
        results = model.val(device=device)
        print(f"Validação concluída. Resultados: {results}")
    except Exception as e:
        print(f"Erro na validação: {e}")

def detect_images(directory_path):
    try:
        image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print("Nenhuma imagem encontrada no diretório.")
            return

        for image_file in image_files:
            image_path = os.path.join(directory_path, image_file)
            print(f"Processando imagem: {image_file}")
            results = model(image_path)
            for result in results:
                print(f"Classes detectadas: {result.boxes.cls.tolist()}")
            results[0].plot(show=True)

        print("Processamento concluído para todas as imagens.")
    except Exception as e:
        print(f"Erro no processamento das imagens: {e}")

if __name__ == "__main__":
    train_model()
    validate_model()
    detect_images('images')
