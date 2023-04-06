import cv2
import numpy as np

# 모델, 이미지 로드
net = cv2.dnn.readNetFromTorch('models/eccv16/la_muse.t7')
img = cv2.imread('imgs/school2.jpg')
cropped_img = img[480:1020, 600:1010]

# 이미지 전처리
h, w, c = cropped_img.shape
cropped_img = cv2.resize(cropped_img, dsize=(500, int(h / w * 500)))

MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(cropped_img, mean=MEAN_VALUE)

# 추론
net.setInput(blob)
output = net.forward()

# 이미지 후처리
output = output.squeeze().transpose((1,2,0))
output += MEAN_VALUE

output = np.clip(output, 0, 255)
output = output.astype('uint8')

# 이미지 출력
cv2.imshow('output', output)
cv2.waitKey(0)



