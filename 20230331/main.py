import cv2

cap = cv2.VideoCapture('01.mp4')

while True:
    
    ret, img = cap.read()
        
    if ret == False:
        break
    
    cv2.rectangle(img, pt1=(200, 200), pt2=(400, 400), color=(255, 0, 0), thickness=2)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, dsize=(640, 360))
    
    img = img[100:300, 100:800]
    
    cv2.imshow('result', img)
    
    if cv2.waitKey(100) == ord('q'):
        break
    
    
    
    