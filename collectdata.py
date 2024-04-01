import os
import cv2

fgbg = cv2.createBackgroundSubtractorMOG2() 

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 36
dataset_size = 5000
cap = cv2.VideoCapture(0)
window_width = 1200
window_height = 900
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

x, y, w, h = 800, 200, 350, 350

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))
        
    print('Collecting data for class {}'.format(j))
    
    while True:
        ret, frame = cap.read()
        flipped_frame = cv2.flip(frame, 1)
        cv2.rectangle(flipped_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(flipped_frame, 'Ready? Press "R"', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', flipped_frame)
        if cv2.waitKey(25) == ord('r'):
            break
    counter = 0

    # while counter < dataset_size:
    #     ret, frame = cap.read()
    #     flipped_frame = cv2.flip(frame, 1)
    #     cv2.rectangle(flipped_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     roi = flipped_frame[y:y+h, x:x+w]
    #     cv2.imshow('frame', flipped_frame)
    #     cv2.waitKey(25)
    #     cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), roi)
    #     counter += 1

    flag = True
    while counter < dataset_size:
        ret, frame = cap.read()
        flipped_frame = cv2.flip(frame, 1)
        cv2.rectangle(flipped_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = flipped_frame[y:y + h, x:x + w]
 
        roi = fgbg.apply(roi) 

        cv2.imshow('frame', flipped_frame)
        key = cv2.waitKey(25)
        if key & 0xFF == ord('s'):
            flag = not flag
            print("Capturing :", flag)
        if flag:
            cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), roi)
            counter += 1
        if key & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
