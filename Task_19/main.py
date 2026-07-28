import cv2
import time
from ultralytics import YOLO

# 1. تحميل النموذج اللي أنت دربته
model = YOLO('best.pt')

# اسم الكائن اللي عايزين نعده (ميزة البونص) - تقدر تغير 'apple' لأي حاجة تانية
TARGET_CLASS = 'apple'

def process_video(source, is_video_file=False):
    cap = cv2.VideoCapture(source)
    
    # تجهيز إعدادات حفظ الفيديو لو المستخدم اختار ملف فيديو
    out = None
    if is_video_file:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_out = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_annotated.mp4', fourcc, fps_out, (width, height))
    
    prev_time = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # 2. عمل الاستدلال (Inference)
        results = model(frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        
        # 3. حساب الـ FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # 4. حساب عداد الكائن (البونص)
        target_count = 0
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name.lower() == TARGET_CLASS.lower():
                target_count += 1
                
        # 5. طباعة الـ FPS والعداد على الشاشة
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'{TARGET_CLASS} Count: {target_count}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        # عرض الإطار
        cv2.imshow("YOLO Detection", annotated_frame)
        
        # حفظ الإطار لو بنعالج فيديو
        if out:
            out.write(annotated_frame)
        
        # الضغط على 'q' للخروج
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

# ==========================================
# قائمة التشغيل
# ==========================================
print("Choose a mode:")
print("1. Live Camera")
print("2. Video File")
choice = input("Enter 1 or 2: ")

if choice == '1':
    process_video(0)
elif choice == '2':
    video_path = input("Enter video name (e.g., test.mp4): ")
    process_video(video_path, is_video_file=True)
else:
    print("Invalid choice!")