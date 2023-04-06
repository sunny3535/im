import cv2
import numpy as np

# 모델, 이미지 로드
net = []
net.append(cv2.dnn.readNetFromTorch('models/eccv16/composition_vii.t7'))
net.append(cv2.dnn.readNetFromTorch('models/eccv16/la_muse.t7'))
net.append(cv2.dnn.readNetFromTorch('models/eccv16/starry_night.t7'))
net.append(cv2.dnn.readNetFromTorch('models/eccv16/the_wave.t7'))

img = cv2.imread('imgs/cafe.jpg')
h, w, c = img.shape
wd = int(w/4)
cropped_img = []
output = []

for i in range(0, 4):
    cropped_img.append(img[:, i*wd:(i+1)*wd])

# 이미지 전처리
h, w, c = cropped_img[0].shape
MEAN_VALUE = [103.939, 116.779, 123.680]

for i in range(0, 4):
    cropped_img[i] = cv2.resize(cropped_img[i], dsize=(500, int(h / w * 500)))
    blob = cv2.dnn.blobFromImage(cropped_img[i], mean=MEAN_VALUE)
    
    # 추론
    net[i].setInput(blob)
    output.append(net[i].forward())
    
    # 이미지 후처리
    output[i] = output[i].squeeze().transpose((1,2,0))
    output[i] += MEAN_VALUE
    output[i] = np.clip(output[i], 0, 255)
    output[i] = output[i].astype('uint8')


# 이미지 출력
output = np.concatenate([output[0], output[1], output[2], output[3]], axis=1)
h, w, c = output.shape
output = cv2.resize(output, dsize=(500, int(h / w * 500)))
cv2.imshow('output', output)
h, w, c = img.shape
img = cv2.resize(img, dsize=(500, int(h / w * 500)))
cv2.imshow('img', img)
cv2.waitKey(0)



