# _*_ coding: utf-8 _*_
# ===================================================================
# Copyright (C) 2025 - 2025 Ranjun Mo, Inc. All Rights Reserved 
# ===================================================================
# @Time    : 2025/5/27 09:52
# @Author  : Ranjun Mo
import cv2
import numpy as np

def apply_inpainting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 更宽容地提取掩膜
    mask = (gray < 10).astype(np.uint8) * 255

    # 可选：膨胀一下掩膜，避免锯齿边缘
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 修补
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return result
