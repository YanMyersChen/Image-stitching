# _*_ coding: utf-8 _*_
# ===================================================================
# Copyright (C) 2025 - 2025 Ranjun Mo, Inc. All Rights Reserved 
# ===================================================================
# @Time    : 2025/5/27 09:49
# @Author  : Ranjun Mo
# feature_matcher.py
import cv2

def extract_features(img, method='SIFT'):
    if method == 'SIFT':
        detector = cv2.SIFT_create()
    elif method == 'ORB':
        detector = cv2.ORB_create()
    else:
        raise ValueError("Unsupported feature method: choose 'SIFT' or 'ORB'")

    keypoints, descriptors = detector.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(desc1, desc2, method='SIFT'):
    if method == 'SIFT':
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    elif method == 'ORB':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        raise ValueError("Unsupported match method: choose 'SIFT' or 'ORB'")

    matches = matcher.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return good

