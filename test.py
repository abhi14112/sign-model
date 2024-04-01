import cv2
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('my_cnn_model.h5')
fgbg = cv2.createBackgroundSubtractorMOG2() 
label_list = ['0','1','2','3','4','5','6','7','8','9']
# label_list = ['A','B','C','D','E','F','G','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Z']
def preprocess_frame(frame):
    if frame is None:
        return None
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    frame = cv2.resize(frame, (100, 100))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=-1)  # Adjust shape to (100, 100, 1)
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
x, y, w, h = 800, 200, 350, 350
while True:
    ret, frame = cap.read()
    flipped_frame = cv2.flip(frame, 1)
    cv2.rectangle(flipped_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    frame = flipped_frame[y:y+h, x:x+w]
    frame = fgbg.apply(frame) 
    if not ret:
        print("Error: Failed to capture frame")
        break
    label, confidence = classify_frame(frame)
    if label is None:
        print("Error: Frame classification failed")
        break

    cv2.putText(flipped_frame, f'{label} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera', flipped_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()