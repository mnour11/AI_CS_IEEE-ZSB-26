from fastapi import FastAPI, UploadFile, File
from app.model import predict_image

app = FastAPI(title="Arabic OCR Service - IEEE Zagazig")

# مسار الـ Health Check
@app.get("/")
def health_check():
    return {"status": "System is running perfectly"}

# مسار التنبؤ
@app.post("/predict")
async def predict_text(file: UploadFile = File(...)):
    # قراءة الصورة المرفوعة كـ bytes
    image_bytes = await file.read()
    
    # تمرير الصورة للموديل واسترجاع النص
    text = predict_image(image_bytes)
    
    # إرجاع النتيجة بصيغة JSON زي ما التاسك طالب
    return {
        "filename": file.filename,
        "predicted_text": text
    }