# yolo_realtime.py - Using YOLO for 2D Keypoint Detection in Real-time
import os
import cv2
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import time
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from video import Keypoint, getKptsFromImage
from utilsCombine import videopose_model_load, interface, videoInfo, resize_img ,draw_3Dimg, draw_2Dimg

# Load YOLO model for real-time pose estimation
pose_model = Keypoint(model_path='weights/yolov8x-pose.onnx')
interface2D = getKptsFromImage
interface3D = interface
model3D = videopose_model_load()


def main(VideoName):
    cap, cap_length = videoInfo(VideoName)
    kpt2Ds = []
    for i in tqdm(range(cap_length)):
        _, frame = cap.read()
        frame, W, H = resize_img(frame)

        try:
            t0 = time.time()
            # Using the YOLO model to extract 2D keypoints
            bboxs, joint2D = interface2D(pose_model, frame)
            print('YOLO consume {:0.3f} s'.format(time.time() - t0))

        except Exception as e:
            print(e)
            continue

        if i == 0:
            for _ in range(30):
                kpt2Ds.append(joint2D)
        elif i < 30:
            kpt2Ds.append(joint2D)
            kpt2Ds.pop(0)
        else:
            kpt2Ds.append(joint2D)

        joint3D = interface3D(model3D, np.array(kpt2Ds), W, H)
        joint3D_item = joint3D[-1] #(17, 3)
        draw_3Dimg(joint3D_item, frame, display=1, kpt2D=joint2D)

if __name__ == '__main__':
    # 初始化 Tkinter，隐藏主窗口
    root = tk.Tk()
    root.withdraw()

    # 打开文件对话框以选择视频文件
    VideoName = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )

    # 检查是否选择了文件
    if not VideoName:
        print("No video file selected.")
    else:
        print('Input Video Name is ', VideoName)
        main(VideoName)

