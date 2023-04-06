import cv2
import numpy as np

# 모델, 이미지 로드
net1 = cv2.dnn.readNetFromTorch('models/eccv16/composition_vii.t7')
net2 = cv2.dnn.readNetFromTorch('models/eccv16/la_muse.t7')

img = cv2.imread('imgs/cafe.jpg')
h, w, c = img.shape
wd = int(w/2)
cropped_img1 = img[:, 0:wd]
cropped_img2 = img[:, wd:w]

# 이미지 전처리
h, w, c = cropped_img1.shape
cropped_img1 = cv2.resize(cropped_img1, dsize=(500, int(h / w * 500)))
cropped_img2 = cv2.resize(cropped_img2, dsize=(500, int(h / w * 500)))

MEAN_VALUE = [103.939, 116.779, 123.680]
blob1 = cv2.dnn.blobFromImage(cropped_img1, mean=MEAN_VALUE)
blob2 = cv2.dnn.blobFromImage(cropped_img2, mean=MEAN_VALUE)

# 추론
net1.setInput(blob1)
net2.setInput(blob2)
output1 = net1.forward()
output2 = net2.forward()

# 이미지 후처리
output1 = output1.squeeze().transpose((1,2,0))
output2 = output2.squeeze().transpose((1,2,0))
output1 += MEAN_VALUE
output2 += MEAN_VALUE

output1 = np.clip(output1, 0, 255)
output2 = np.clip(output2, 0, 255)
output1 = output1.astype('uint8')
output2 = output2.astype('uint8')

# 이미지 출력
output = np.concatenate([output1, output2], axis=1)
h, w, c = output.shape
output = cv2.resize(output, dsize=(500, int(h / w * 500)))
cv2.imshow('output', output)
h, w, c = img.shape
img = cv2.resize(img, dsize=(500, int(h / w * 500)))
cv2.imshow('img', img)
cv2.waitKey(0)



