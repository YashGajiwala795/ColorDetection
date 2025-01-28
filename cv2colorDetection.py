import cv2
import numpy as np

red_L1 = np.array([0,120,70])
red_U1 = np.array([10,255,255])
red_L2 = np.array([170,120,70])
red_U2 = np.array([180,255,255])

green_L1 = np.array([36,50,70])
green_U1 = np.array([70,255,255])
green_L2 = np.array([76,50,70])
green_U2 = np.array([90,255,255])

blue_L1 = np.array([100,150,50])
blue_U1 = np.array([130,255,255])
blue_L2 = np.array([90,50,50])
blue_U2 = np.array([110,255,255])


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame Not Capture...")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, red_L1, red_U1)
    mask2 = cv2.inRange(hsv, red_L2, red_U2)
    red_mask = mask1+mask2

    g_mask1 = cv2.inRange(hsv, green_L1, green_U1)
    g_mask2 = cv2.inRange(hsv, green_L2, green_U2)
    g_mask = g_mask1+g_mask2

    b_mask1 = cv2.inRange(hsv, blue_L1, blue_U1)
    b_mask2 = cv2.inRange(hsv, blue_L2, blue_U2)
    b_mask = b_mask1+b_mask2

    kernel = np.ones((5,5), np.uint8)
    
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    g_mask = cv2.morphologyEx(g_mask, cv2.MORPH_OPEN, kernel)
    g_mask = cv2.morphologyEx(g_mask, cv2.MORPH_CLOSE, kernel)

    b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_OPEN, kernel)
    b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_CLOSE, kernel)

    contours,_ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours1,_ = cv2.findContours(g_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2,_ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour)>500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x,y), (x+w , y+h),(0,0,255), 2)
            cv2.putText(frame, "RED OBJECT", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            
    for contour1 in contours1:
        if cv2.contourArea(contour1)>500:
            x, y, w, h = cv2.boundingRect(contour1)
            cv2.rectangle(frame, (x,y), (x+w , y+h),(0,255,0), 2)
            cv2.putText(frame, "Green OBJECT", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    for contour2 in contours2:
        if cv2.contourArea(contour2)>500:
            x, y, w, h = cv2.boundingRect(contour2)
            cv2.rectangle(frame, (x,y), (x+w , y+h),(255,0,0), 2)
            cv2.putText(frame, "Blue OBJECT", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)



            
    cv2.imshow("RED Green Blue DETECTED", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break




            
