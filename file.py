import cv2

cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()
window_width = 1200
window_height = 900

cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('flipped_video_with_rectangle.avi', fourcc, 20.0, (window_width, window_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame")
        break

    flipped_frame = cv2.flip(frame, 1)

    x, y, w, h = 800, 200, 350, 350
    color = (0, 255, 0)  
    thickness = 2  
    cv2.rectangle(flipped_frame, (x, y), (x + w, y + h), color, thickness)
    out.write(flipped_frame)

    cv2.imshow('Flipped Video with Rectangle', flipped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
