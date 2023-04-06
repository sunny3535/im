import cv2
import numpy as np

# 모델, 이미지 로드
net1 = cv2.dnn.readNetFromTorch('models/eccv16/composition_vii.t7')
net2 = cv2.dnn.readNetFromTorch('models/eccv16/la_muse.t7')
net3 = cv2.dnn.readNetFromTorch('models/eccv16/starry_night.t7')
net4 = cv2.dnn.readNetFromTorch('models/eccv16/the_wave.t7')

img = cv2.imread('imgs/cafe.jpg')
h, w, c = img.shape
wd = int(w/4)
cropped_img1 = img[:, 0:wd]
cropped_img2 = img[:, wd:wd*2]
cropped_img3 = img[:, wd*2:wd*3]
cropped_img4 = img[:, wd*3:w]

# 이미지 전처리
h, w, c = cropped_img1.shape
cropped_img1 = cv2.resize(cropped_img1, dsize=(500, int(h / w * 500)))
cropped_img2 = cv2.resize(cropped_img2, dsize=(500, int(h / w * 500)))
cropped_img3 = cv2.resize(cropped_img3, dsize=(500, int(h / w * 500)))
cropped_img4 = cv2.resize(cropped_img4, dsize=(500, int(h / w * 500)))

MEAN_VALUE = [103.939, 116.779, 123.680]
blob1 = cv2.dnn.blobFromImage(cropped_img1, mean=MEAN_VALUE)
blob2 = cv2.dnn.blobFromImage(cropped_img2, mean=MEAN_VALUE)
blob3 = cv2.dnn.blobFromImage(cropped_img3, mean=MEAN_VALUE)
blob4 = cv2.dnn.blobFromImage(cropped_img4, mean=MEAN_VALUE)

# 추론
net1.setInput(blob1)
net2.setInput(blob2)
net3.setInput(blob3)
net4.setInput(blob4)
output1 = net1.forward()
output2 = net2.forward()
output3 = net3.forward()
output4 = net4.forward()

# 이미지 후처리
output1 = output1.squeeze().transpose((1,2,0))
output2 = output2.squeeze().transpose((1,2,0))
output3 = output3.squeeze().transpose((1,2,0))
output4 = output4.squeeze().transpose((1,2,0))
output1 += MEAN_VALUE
output2 += MEAN_VALUE
output3 += MEAN_VALUE
output4 += MEAN_VALUE

output1 = np.clip(output1, 0, 255)
output2 = np.clip(output2, 0, 255)
output3 = np.clip(output3, 0, 255)
output4 = np.clip(output4, 0, 255)
output1 = output1.astype('uint8')
output2 = output2.astype('uint8')
output3 = output3.astype('uint8')
output4 = output4.astype('uint8')

# 이미지 출력
output = np.concatenate([output1, output2, output3, output4], axis=1)
h, w, c = output.shape
output = cv2.resize(output, dsize=(500, int(h / w * 500)))
cv2.imshow('output', output)
h, w, c = img.shape
img = cv2.resize(img, dsize=(500, int(h / w * 500)))
cv2.imshow('img', img)
cv2.waitKey(0)



