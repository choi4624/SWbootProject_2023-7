# app.py

from fastapi import FastAPI, HTTPException
import torch
import torchvision.transforms as transforms
from PIL import Image

app = FastAPI()

# 미리 학습된 파이토치 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("path/to/your/ckpt_file.pth", map_location=device)
model.eval()

# 이미지 변환 함수
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

# API 엔드포인트 정의
@app.post("/predict/")
async def predict(image_file: bytes = Upload(...)):
    try:
        # 이미지 파일을 PIL Image로 변환
        image = Image.open(io.BytesIO(image_file)).convert("RGB")
        # 이미지를 모델 입력에 맞게 전처리
        input_tensor = preprocess_image(image)
        # 배치 차원 추가
        input_batch = input_tensor.unsqueeze(0)
        # 추론
        with torch.no_grad():
            input_batch = input_batch.to(device)
            output = model(input_batch)
        # 추론 결과 반환
        _, predicted_idx = torch.max(output, 1)
        return {"predicted_class": predicted_idx.item()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

