import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

cap = cv2.VideoCapture(0)
sticker_img = cv2.imread('imgs/nose03.png', cv2.IMREAD_UNCHANGED)
sticker_img2 = cv2.imread('imgs/eye06.png', cv2.IMREAD_UNCHANGED)
sticker_img3 = cv2.imread('imgs/mouth13.png', cv2.IMREAD_UNCHANGED)
sticker_img4 = cv2.imread('imgs/flower.png', cv2.IMREAD_UNCHANGED)

while True:
    ret, img = cap.read()

    if ret == False:
        break

    dets = detector(img)

    for det in dets:
        shape = predictor(img, det)

        try:
            x1 = det.left()
            y1 = det.top()
            x2 = det.right()
            y2 = det.bottom()

            # compute pig nose coordinates
            center_x = shape.parts()[4].x
            center_y = shape.parts()[4].y - 10

            h, w, c = sticker_img.shape

            nose_w = int((x2 - x1) / 6)
            nose_h = int(h / w * nose_w)

            nose_x1 = int(center_x - nose_w / 2)
            nose_x2 = nose_x1 + nose_w

            nose_y1 = int(center_y - nose_h / 2)
            nose_y2 = nose_y1 + nose_h

            # overlay nose
            overlay_img = sticker_img.copy()
            overlay_img = cv2.resize(overlay_img, dsize=(nose_w, nose_h))
            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha
            img[nose_y1:nose_y2, nose_x1:nose_x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[nose_y1:nose_y2, nose_x1:nose_x2]

            # 입 스티커
            center_x = shape.parts()[4].x
            center_y = shape.parts()[4].y + 50

            h, w, c = sticker_img3.shape

            mouth_w = int((x2 - x1) / 2)
            mouth_h = int(h / w * mouth_w)

            mouth_x1 = int(center_x - mouth_w / 2)
            mouth_x2 = mouth_x1 + mouth_w

            mouth_y1 = int(center_y - mouth_h / 2)
            mouth_y2 = mouth_y1 + mouth_h

            # overlay mouth
            overlay_img = sticker_img3.copy()
            overlay_img = cv2.resize(overlay_img, dsize=(mouth_w, mouth_h))
            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha
            img[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

            # 머리장식 스티커
            center_x = shape.parts()[0].x 
            center_y = shape.parts()[0].y - 150

            h, w, c = sticker_img4.shape

            tip_w = int((x2 - x1) / 3)
            tip_h = int(h / w * tip_w)

            tip_x1 = int(center_x - tip_w / 2)
            tip_x2 = tip_x1 + tip_w

            tip_y1 = int(center_y - tip_h / 2)
            tip_y2 = tip_y1 + tip_h

            # overlay mouth
            overlay_img = sticker_img4.copy()
            overlay_img = cv2.resize(overlay_img, dsize=(tip_w, tip_h))
            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha
            img[tip_y1:tip_y2, tip_x1:tip_x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[tip_y1:tip_y2, tip_x1:tip_x2]
            
            # 눈 스티커
            eye_x1 = shape.parts()[2].x - 20
            eye_x2 = shape.parts()[0].x + 20
            
            h, w, c = sticker_img2.shape
            
            eye_w = eye_x2 - eye_x1
            eye_h = int(h / w * eye_w)
            
            center_y = (shape.parts()[0].y + shape.parts()[2].y) / 2
            
            eye_y1 = int(center_y - eye_h / 2)
            eye_y2 = eye_y1 + eye_h
            
            overlay_img = sticker_img2.copy()
            overlay_img = cv2.resize(overlay_img, dsize=(eye_w, eye_h))
            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha
            img[eye_y1:eye_y2, eye_x1:eye_x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[eye_y1:eye_y2, eye_x1:eye_x2]

        except:
            pass

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break