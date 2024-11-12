import onnxruntime as ort
import numpy as np
import cv2
import sys
import io
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
model_path = 'weights/yolov8x-pose.onnx'


# 定义一个调色板数组，其中每个元素是一个包含RGB值的列表，用于表示不同的颜色
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])
# 定义人体17个关键点的连接顺序
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

def getModel():
    session = ort.InferenceSession(model_path, providers=[('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    return session, input_name, label_name

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

def pre_process(img):
    img = img / 255.
    img = np.transpose(img, (2, 0, 1))
    data = np.expand_dims(img, axis=0)
    return data

class Keypoint():
    def __init__(self, model_path):
        self.session, self.input_name, self.label_name = getModel()

    def inference(self, image):
        # 前處理
        img = letterbox(image)
        data = pre_process(img)
        # 模型推論
        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        pred = pred[0]
        pred = np.transpose(pred, (1, 0))
        conf = 0.7
        pred = pred[pred[:, 4] > conf]
        if len(pred) == 0:
            print("沒有檢測到任何關鍵點")
            return None
        else:
            # 返回 bboxs 和關鍵點數據
            bboxs = self.xywh2xyxy(pred)
            kpts = pred[:, 5:]  # 假設 kpts 是從第5個元素開始
            return bboxs, kpts

    

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def nms(self, dets, iou_thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        output = []
        for i in keep:
            output.append(dets[i].tolist())
        return np.array(output)

    def xyxy2xywh(self, a):
        b = np.copy(a)
        b[:, 2] = a[:, 2] - a[:, 0]
        b[:, 3] = a[:, 3] - a[:, 1]
        return b

def getKptsFromImage(pose_model, image):
    # 從模型得到邊界框和關鍵點
    result = pose_model.inference(image)
    
    if result is not None:
        bboxs, kpts = result
        # 如果檢測到的關鍵點數量小於 17，則補全到 17 個
        if kpts.shape[0] < 17:
            # 創建一個空陣列，填充檢測到的關鍵點
            keypoints_with_confidence = np.zeros((17, 3))
            for i in range(kpts.shape[0]):
                x, y = kpts[i][0], kpts[i][1]
                confidence = bboxs[i][4] if i < len(bboxs) else 0
                keypoints_with_confidence[i] = [x, y, confidence]
            return keypoints_with_confidence
        else:
            # 若關鍵點數量足夠，直接返回
            confidences = np.expand_dims(bboxs[:, 4], axis=1)
            return np.hstack((kpts, confidences))
    else:
        # 若未檢測到任何關鍵點，返回 None
        return None
