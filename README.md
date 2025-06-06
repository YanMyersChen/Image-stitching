# Image-stitching
A demo for Image stitching

# 图像拼接与补全过程演示系统说明文档

## 项目简介

本项目为一款基于 Streamlit 构建的交互式图像编辑 Web 应用，用户可上传任意两张图像，通过特征匹配实现图像自动拼接，并对拼接过程中出现的空洞区域进行图像补全（inpainting）处理。

项目核心功能包括：

- 图像上传与预览
- 特征点提取与匹配（支持 ORB / SIFT）
- 图像拼接（基于单应性矩阵）
- 空洞区域自动修复
- 拼接图像与修复图像并列展示与下载
- 自定义背景界面与视觉交互优化

---

## 项目结构

```
├── app.py                # 主界面程序（Streamlit Web 端）
├── main.py               # 命令行批处理入口，用于离线拼接演示
├── feature_matcher.py    # 特征提取与匹配封装
├── image_stitcher.py     # 图像拼接模块（外部提供）
├── inpainting.py         # 图像修复模块（外部提供）
├── Background/           # 背景图像资源文件夹
├── Image/                # 默认测试图像
└── utils.py              # 辅助函数如加标题等
```

---

## 核心功能说明

### 1. 图像上传与预览

用户通过界面上传两张 JPG/PNG 图像，系统将自动显示图像预览，支持调整预览宽度。

### 2. 特征提取与匹配（`feature_matcher.py`）

支持 SIFT 和 ORB 两种常用图像特征方法：

- `extract_features(img, method='SIFT')`: 提取关键点与描述符
- `match_features(desc1, desc2, method='SIFT')`: 基于 KNN + 比率测试进行特征匹配

> 默认使用 `cv2.BFMatcher`，ORB 使用 Hamming 距离，SIFT 使用 L2 距离。

### 3. 图像拼接（`stitch_images`）

- 通过特征点计算图像间单应性矩阵
- 使用 `cv2.warpPerspective` 将一张图映射到另一张图的坐标空间
- 使用非零掩膜合成输出图像

### 4. 空洞修复（`apply_inpainting`）

- 针对拼接结果中透明/黑色区域进行基于 OpenCV 的修复（如 `cv2.inpaint()`）

---

## 用户界面亮点（`app.py`）

- 背景图支持自定义：通过 `get_base64_bg()` 函数读取本地图像作为背景
- ️并列显示拼接结果与修复图
- 支持选择算法与命名任务
- 支持下载最终图像结果
- 实时交互，无需重启或命令行操作

---

## 示例运行效果（Web 模式）

1. 启动界面：

   ```bash
   streamlit run app.py
   ```

2. 上传图像 → 选择算法 → 点击“执行拼接与补全”

3. 页面展示：

   - 图像预览（上传）
   - 拼接图像（Stitched）
   - 修复图像（Inpainted）
   - 下载按钮

---

## 示例运行（命令行离线模式）

可运行 `main.py`，自动读取 `Image/im0.png` 与 `Image/im1.png` 进行拼接并弹出结果窗口：

```bash
python main.py
```

---

## 环境依赖

- Python 3.8+
- OpenCV-Python
- NumPy
- Pillow
- Streamlit >= 1.25

安装依赖建议：

```bash
pip install opencv-python numpy pillow streamlit
```

---
## 👤 作者信息

- 🧑‍💻 作者：Ranjun Mo  
- 📅 时间：2025年5月  
- 🔒 版权所有 © 2025 Ranjun Mo, Inc.
