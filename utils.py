# _*_ coding: utf-8 _*_
# ===================================================================
# Copyright (C) 2025 - 2025 Ranjun Mo, Inc. All Rights Reserved 
# ===================================================================
# @Time    : 2025/5/27 10:22
# @Author  : Ranjun Mo
import cv2
import numpy as np

def add_title(img, title, height=30):
    """在图像上方添加标题栏"""
    h, w = img.shape[:2]
    title_bar = np.full((height, w, 3), 255, dtype=np.uint8)  # 白色背景
    cv2.putText(title_bar, title, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack((title_bar, img))