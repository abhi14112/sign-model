import cv2
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('sign_model.1.h5')
label_list = ['0','1','2','3','4','5','6','7','8','9','A','B','C']
def preprocess_frame(frame):
    if frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(frame,(5,5),1.5)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9,2)
    ret, res = cv2.threshold(th3, 100, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    frame = res
    frame = cv2.resize(frame, (98, 98))
    frame = np.array(frame) / 255.0
    return np.expand_dims(frame, axis=-1) 
def classify_frame(frame):
    preprocessed_frame = preprocess_frame(frame)
    if preprocessed_frame is None:
        return None, None
    predictions = model.predict(np.expand_dims(preprocessed_frame, axis=0))
    class_idx = np.argmax(predictions)
    label = label_list[class_idx]
    confidence = predictions[0][class_idx]
    return label, confidence
cap = cv2.VideoCapture(0)
window_width = 1200
window_height = 900
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)
x, y, w, h = 700, 100, 500, 500
while True:
    ret, frame = cap.read()
    flipped_frame = cv2.flip(frame, 1)
    cv2.rectangle(flipped_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    frame = flipped_frame[y+4:y+h-4, x+4:x+w-4]
    if not ret:
        print("Error: Failed to capture frame")
        break
    label, confidence = classify_frame(frame)
    if label is None:
        print("Error: Frame classification failed")
        break
    cv2.putText(flipped_frame, f'{label} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera', flipped_frame )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()