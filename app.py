# _*_ coding: utf-8 _*_
# ===================================================================
# Copyright (C) 2025 - 2025 Ranjun Mo, Inc. All Rights Reserved
# ===================================================================
# @Time    : 2025/5/27 11:52
# @Author  : Ranjun Mo
import streamlit as st
import numpy as np
import io
from image_stitcher import stitch_images
from inpainting import apply_inpainting
from feature_matcher import *
from PIL import Image
import base64

def get_base64_bg(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    return f"data:image/png;base64,{encoded}"

bg_img = get_base64_bg("Background/background.png")

st.set_page_config(
    page_title="图像拼接神器",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{bg_img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("图像拼接Demo")
st.markdown("上传两张图像，选择特征算法，实现图像拼接与补全流程。")

# 图像上传布局
col1, col2 = st.columns(2)

with col1:
    uploaded1 = st.file_uploader("📤 上传图像1", type=["jpg", "png"], key="img1")

with col2:
    uploaded2 = st.file_uploader("📤 上传图像2", type=["jpg", "png"], key="img2")

if uploaded1 and uploaded2:
    file_bytes1 = np.asarray(bytearray(uploaded1.read()), dtype=np.uint8)
    file_bytes2 = np.asarray(bytearray(uploaded2.read()), dtype=np.uint8)

    img1 = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 任务命名
    task_name = st.text_input("📝 输入拼接任务名称（可选）", value="image")
    img_width = st.slider("图像预览宽度", min_value=200, max_value=800, value=400, step=50)
    st.image([img1_rgb, img2_rgb], caption=["图像1", "图像2"], width=img_width)

    gray1 = cv2.cvtColor(img1_rgb, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_rgb, cv2.COLOR_BGR2GRAY)
    match_method = st.selectbox("选择特征检测算法", ["ORB", "SIFT"])


    if st.button("执行拼接与补全"):
        with st.spinner("🔧 正在处理图像拼接与修复，请稍候..."):
            kp1, des1 = extract_features(gray1, method=match_method)
            kp2, des2 = extract_features(gray2, method=match_method)
            matches = match_features(des1, des2, method=match_method)
            stitched = stitch_images(img1_rgb, img2_rgb, kp1, kp2, matches)
            inpainted = apply_inpainting(stitched)

        st.success("🎉 拼接与修复完成！")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧵 拼接图像")
            st.image(stitched, use_container_width=True)

        with col2:
            st.subheader("🩹 补全图像")
            st.image(inpainted, use_container_width=True)

        # 下载按钮
        result_pil = Image.fromarray(inpainted)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            label="📥 下载结果图像",
            data=buf.getvalue(),
            file_name=f"{task_name}_stitched.png",
            mime="image/png"
        )
else:
    st.info("请上传两张图像以开始处理。")

