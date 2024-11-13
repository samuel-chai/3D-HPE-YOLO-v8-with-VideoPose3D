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

from yolo_detectors.yolov8_detection import Keypoint
from tool.utils import videopose_model_load as Model3Dload
model3D = Model3Dload()
from tool.utils import interface as VideoPoseInterface
interface3D = VideoPoseInterface
from tool.utils import draw_3Dimg, videoInfo, resize_img

model_path = 'weights/yolov8x-pose.onnx'
keydet = Keypoint(model_path)


def main(VideoName):
    cap, cap_length = videoInfo(VideoName)
    kpt2Ds = []
    processed_frames = []

    for i in tqdm(range(cap_length)):
        _, frame = cap.read()
        frame, W, H = resize_img(frame)

        try:
            t0 = time.time()
            joint2D = np.array(keydet.getkptsFromImg(frame))
            print('YOLO consume {:0.3f} s'.format(time.time() - t0))
            # 檢查形狀, 若數據不對跳過frame
            if joint2D.shape[-2:] != (17, 3):
                print("Skipping frame due to incorrect shape:", joint2D.shape)
                continue
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
        joint3D_item = joint3D[-1] # (17, 3)
        processed_frame = draw_3Dimg(joint3D_item, frame, display=1, kpt2D=joint2D)
        processed_frames.append(processed_frame)
     # 在所有幀處理完成後，保存處理過的幀為視頻
    if processed_frames:
        height, width, layers = processed_frames[0].shape
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

        for frame in processed_frames:
            out.write(frame)
        out.release()
        print("Processed video saved as 'output.avi'.")
            
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


