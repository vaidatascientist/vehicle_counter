import cv2
import numpy as np

# Web Camera
capture = cv2.VideoCapture('video.mp4')

min_width_rectangle = 80
min_height_rectangle = 80

count_line_position = 550

offset = 6

counter = 0

# Initialize Subtractor
algorithm = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

while True:
    ret, frame = capture.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    
    # Apply on each frame
    frame_subtractor = algorithm.apply(blur)
    dilate = cv2.dilate(frame_subtractor, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilateada = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilateada = cv2.morphologyEx(dilateada, cv2.MORPH_CLOSE, kernel)
    contour, h = cv2.findContours(dilateada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    
    for i, c in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contour = (w >= min_width_rectangle) and (h >= min_height_rectangle)
        if not validate_contour:
            continue
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'VEHICLE: ' + str(counter), (x + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        center = (int((x + x + w) / 2), int((y + y + h) / 2))
        cv2.circle(frame, center, 4, (0, 0, 255), -1)
        
        if (center[1] >= count_line_position - offset) and (center[1] <= count_line_position + offset):
            cv2.line(frame, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
            counter += 1
            print('Car Detected: ' + str(counter))
    
    # cv2.imshow('Detecter', dilateada)
    cv2.putText(frame, 'VEHICLE COUNT: ' + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5) 
    cv2.imshow('VideoFrame', frame)
    
    if cv2.waitKey(1) == 13: break
    
cv2.destroyAllWindows()
capture.release()