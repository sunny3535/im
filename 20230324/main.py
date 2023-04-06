import cv2

img = cv2.imread('phone.jpg')

print(img)
print(img.shape)

cv2.rectangle(img, pt1=(300, 110), pt2=(540, 420), color=(255, 0, 0), thickness=2)
cv2.circle(img, center=(420, 260), radius=200, color=(0,0,255), thickness=3)

cropped_img = img[110:420, 300:540]
resized_img = cv2.resize(img, (512, 256))

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('result', img_rgb)
cv2.waitKey(0)

cv2.imshow('resized', resized_img)
cv2.imshow('crop', cropped_img)
cv2.imshow('img', img)
cv2.waitKey(0)















