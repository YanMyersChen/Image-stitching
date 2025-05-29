# _*_ coding: utf-8 _*_
# ===================================================================
# Copyright (C) 2025 - 2025 Ranjun Mo, Inc. All Rights Reserved 
# ===================================================================
# @Time    : 2025/5/27 09:53
# @Author  : Ranjun Mo
import cv2
import numpy as np
from feature_matcher import extract_features,match_features
from image_stitcher import stitch_images
from inpainting import apply_inpainting
from utils import add_title
# 读取图像
img1 = cv2.imread('Image/im1.png')
img2 = cv2.imread('Image/im0.png')

# 灰度 + SIFT
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
method = 'SIFT'
kp1, des1 = extract_features(gray1, method=method)
kp2, des2 = extract_features(gray2, method=method)
matches = match_features(des1, des2, method=method)
# 调用拼接函数
stitched = stitch_images(img1, img2, kp1, kp2, matches, keep_ratio=0.20, ransac_thresh=3)
inpainted = apply_inpainting(stitched)
target_height = 300
target_width = 400
size = (target_width, target_height)
img1_resized = cv2.resize(img1, size)
img2_resized = cv2.resize(img2, size)
stitched_resized = cv2.resize(stitched, size)
inpainted_resized = cv2.resize(inpainted, size)
concat = np.hstack((img1_resized, img2_resized, stitched_resized, inpainted_resized))


# 加标题
img1_titled = add_title(img1_resized, "Original Left")
img2_titled = add_title(img2_resized, "Original Right")
stitched_titled = add_title(stitched_resized, "Stitched")
inpainted_titled = add_title(inpainted_resized, "Inpainted")

# 拼成 2x2 图像
top_row = np.hstack((img1_titled, img2_titled))
bottom_row = np.hstack((stitched_titled, inpainted_titled))
grid = np.vstack((top_row, bottom_row))

# 显示
cv2.imshow("Result", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()

