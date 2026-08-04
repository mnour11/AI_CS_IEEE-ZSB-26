import os
import json
import numpy as np
import cv2
import tensorflow as tf

# 1. تحديد مسارات الملفات
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_ocr_model.keras")
VOCAB_PATH = os.path.join(WEIGHTS_DIR, "vocab.json")

# 2. تحميل القاموس وإعداد طبقة فك التشفير
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_chars = json.load(f)

num_to_char = tf.keras.layers.StringLookup(
    vocabulary=vocab_chars, mask_token=None, invert=True
)

# 3. تحميل الموديل (بيحصل مرة واحدة في بداية تشغيل السيرفر)
print("Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded Successfully!")

def decode_predictions(pred):
    # دالة لفك تشفير مخرجات الموديل (CTC Decode)
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text[0]

def preprocess_image(image_bytes):
    # قراءة الصورة من الـ bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # تغيير الحجم وتطبيع الألوان والقلب (نفس خطوات التدريب)
    img = cv2.resize(img, (600, 50))
    img = img.astype(np.float32) / 255.0
    img = img.T
    
    # إضافة الأبعاد الناقصة (Batch و Channels)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_image(image_bytes):
    # الدالة النهائية اللي هيستخدمها الـ API
    processed_img = preprocess_image(image_bytes)
    preds = model.predict(processed_img)
    predicted_text = decode_predictions(preds)
    return predicted_text