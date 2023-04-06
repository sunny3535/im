import cv2

img = cv2.imread('phone.jpg')
overlay_img = cv2.imread('tree.png', cv2.IMREAD_UNCHANGED)
print(overlay_img.shape)

overlay_img = cv2.resize(overlay_img, dsize=(350,350))

overlay_alpha = overlay_img[:, :, 3:] / 255.0  
background_alpha = 1.0 - overlay_alpha  

x1 = 430
y1 = 90
x2 = x1 + 350
y2 = y1 + 350

img[y1:y2, x1:x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[y1:y2, x1:x2]

cv2.imshow('img', img)
cv2.waitKey(0)

