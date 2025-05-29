# _*_ coding: utf-8 _*_
# ===================================================================
# Copyright (C) 2025 - 2025 Ranjun Mo, Inc. All Rights Reserved 
# ===================================================================
# @Time    : 2025/5/27 09:52
# @Author  : Ranjun Mo
import cv2
import numpy as np


def stitch_images(img1, img2, kp1, kp2, matches, keep_ratio=0.2, ransac_thresh=3.0):
    #筛选匹配点
    good = matches[:int(len(matches) * keep_ratio)]

    if len(good) < 4:
        raise ValueError("匹配点不足，无法估计单应矩阵。")

    # 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 使用更严格的RANSAC阈值估计单应矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if H is None:
        raise ValueError("单应矩阵估计失败。")

    # 计算输出图像尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    all_corners = np.concatenate((warped_corners,
                                  np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    shift = [-xmin, -ymin]
    H_translation = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])

    # 图像变形
    result = cv2.warpPerspective(img1, H_translation @ H, (xmax - xmin, ymax - ymin))

    # 创建纵向渐变融合掩模
    y_start = shift[1]
    y_end = y_start + h2
    x_start = shift[0]
    x_end = x_start + w2

    # 提取叠加区域
    overlay = result[y_start:y_end, x_start:x_end]

    # 生成纵向渐变权重（从顶部到底部，img2权重从1降到0.3）
    rows = overlay.shape[0]
    alpha = np.linspace(1, 0.3, rows).reshape(-1, 1)

    # 扩展维度以匹配图像通道
    if len(overlay.shape) == 3:
        alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

    # 混合图像
    blended = (overlay * (1 - alpha) + img2 * alpha).astype(np.uint8)

    # 创建区域掩模（只处理非零区域）
    non_zero_mask = (overlay != 0).any(axis=2)
    blended = np.where(non_zero_mask[:, :, np.newaxis], blended, img2)

    result[y_start:y_end, x_start:x_end] = blended

    return result
