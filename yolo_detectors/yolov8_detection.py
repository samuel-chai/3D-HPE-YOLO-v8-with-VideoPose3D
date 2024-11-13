import onnxruntime as ort
import numpy as np
import cv2
import time
import sys
import io
import tkinter as tk
from tkinter import filedialog

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class common():
    # 定义一个调色板数组，其中每个元素是一个包含RGB值的列表，用于表示不同的颜色
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])
    skeleton_parents =  np.array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])

    # 定义人体17个关键点的连接顺序，每个子列表包含两个数字，代表要连接的关键点的索引, 1鼻子 2左眼 3右眼 4左耳 5右耳 6左肩 7右肩 8左肘 9右肘 10左手腕 11右手腕 12左髋 13右髋 14左膝 15右膝 16左踝 17右踝
    skeleton = [[0, 1], [1, 3], [0, 2], [2, 4],
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                [5, 11], [6, 12], [11, 12],
                [11, 13], [12, 14], [13, 15], [14, 16]]
    # 通过索引从调色板中选择颜色，用于绘制人体骨架的线条，每个索引对应一种颜色
    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    # 通过索引从调色板中选择颜色，用于绘制人体的关键点，每个索引对应一种颜色
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    pad = (243 - 1) // 2 # Padding on each side
    causal_shift = 0
    
    # 对称关节的定义（左、右）
    kps_left = [1, 3, 5, 7, 9, 11, 13, 15]  # 左肩、左肘、左手腕、左髋、左膝、左踝
    kps_right = [2, 4, 6, 8, 10, 12, 14, 16]  # 右肩、右肘、右手腕、右髋、右膝、右踝
    
    # 定义左关节和右关节（这些是更详细的关节连接用于对称处理）
    joints_left = list([4, 5, 6, 11, 12, 13])  # 左肩、左肘、左手腕、左髋、左膝、左踝
    joints_right = list([1, 2, 3, 14, 15, 16])  # 右肩、右肘、右手腕、右髋、右膝、右踝
    rot = np.array([ 0.14070565, -0.15007018, -0.7552408 ,  0.62232804], dtype=np.float32)

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,  # 可以选择GPU设备ID，如果你有多个GPU
    }),
    'CPUExecutionProvider',  # 也可以设置CPU作为备选
]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    '''  调整图像大小和两边灰条填充  '''
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    # 计算pad长宽
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 保证缩放后图像比例不变
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
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

def xywh2xyxy(x):
    ''' 中心坐标、w、h ------>>> 左上点，右下点 '''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def nms(dets, iou_thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bboxx下标
    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
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

def xyxy2xywh(a):

    b = np.copy(a)
    b[:, 2] = a[:, 2] - a[:, 0]  # w
    b[:, 3] = a[:, 3] - a[:, 1]  # h
    return b

def scale_boxes(img1_shape, boxes, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    boxes[:, 0] -= pad[0]
    boxes[:, 1] -= pad[1]
    boxes[:, :4] /= gain  # 检测框坐标点还原到原图上
    num_kpts = boxes.shape[1] // 3  # 56 // 3 = 18
    for kid in range(2, num_kpts + 1):
        boxes[:, kid * 3 - 1] = (boxes[:, kid * 3 - 1] - pad[0]) / gain
        boxes[:, kid * 3] = (boxes[:, kid * 3] - pad[1]) / gain
    # boxes[:, 5:] /= gain  # 关键点坐标还原到原图上
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    top_left_x = boxes[:, 0].clip(0, shape[1])
    top_left_y = boxes[:, 1].clip(0, shape[0])
    bottom_right_x = (boxes[:, 0] + boxes[:, 2]).clip(0, shape[1])
    bottom_right_y = (boxes[:, 1] + boxes[:, 3]).clip(0, shape[0])
    boxes[:, 0] = top_left_x  # 左上
    boxes[:, 1] = top_left_y
    boxes[:, 2] = bottom_right_x  # 右下
    boxes[:, 3] = bottom_right_y

def plot_skeleton_kpts(im, kpts, pose_kpt_color, pose_limb_color, skeleton, steps=3):
    num_kpts = len(kpts) // steps  # 51 / 3 =17
    # 画点
    for kid in range(num_kpts):
        r, g, b = common.pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        if conf > 0.5:  # 关键点的置信度必须大于 0.5
            cv2.circle(im, (int(x_coord), int(y_coord)), 5, (int(r), int(g), int(b)), -1)
    # 画骨架
    for sk_id, sk in enumerate(common.skeleton):
        r, g, b = common.pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
        pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
        conf1 = kpts[(sk[0] - 1) * steps + 2]
        conf2 = kpts[(sk[1] - 1) * steps + 2]
        if conf1 > 0.5 and conf2 > 0.5:  # 对于肢体，相连的两个关键点置信度 必须同时大于 0.5
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

class Keypoint():
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name
    def inference(self, image):
        img = letterbox(image)
        data = pre_process(img)
        # 预测输出float32[1, 56, 8400]
        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        # [56, 8400]
        pred = pred[0]
        # [8400,56]
        pred = np.transpose(pred, (1, 0))
        # 置信度阈值过滤
        conf = 0.7
        pred = pred[pred[:, 4] > conf]
        if len(pred) == 0:
            print("没有检测到任何关键点")
            return None
        else:
            bboxs = xywh2xyxy(pred)
            bboxs = nms(bboxs, iou_thresh=0.6)
            bboxs = np.array(bboxs)
            bboxs = xyxy2xywh(bboxs)
            bboxs = scale_boxes(img.shape, bboxs, image.shape)
            for box in bboxs:
                det_bbox, det_scores, kpts = box[0:4], box[4], box[5:]
                label = "Person {:.2f}".format(det_scores)
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                plot_skeleton_kpts(image, kpts, common.pose_kpt_color, common.pose_limb_color, common.skeleton)

            return image
    def getkptsFromImg(self, image):
        img = letterbox(image)
        data = pre_process(img)
        # 预测输出float32[1, 56, 8400]
        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        # [56, 8400]
        pred = pred[0]
        # [8400,56]
        pred = np.transpose(pred, (1, 0))
        # 置信度阈值过滤
        conf = 0.7
        pred = pred[pred[:, 4] > conf]
        if len(pred) == 0:
            print("没有检测到任何关键点")
            return None
        else:
            bboxs = xywh2xyxy(pred)
            bboxs = nms(bboxs, iou_thresh=0.6)
            bboxs = np.array(bboxs)
            bboxs = xyxy2xywh(bboxs)
            bboxs = scale_boxes(img.shape, bboxs, image.shape)
            keypoints_list = []
            for box in bboxs:
                kpts = box[5:]  # 获取关键点信息
                keypoints = []
                for i in range(0, len(kpts), 3):
                    x, y, c = kpts[i], kpts[i+1], kpts[i+2]
                    keypoints.append([x, y, c])
                keypoints_list.append(keypoints)
            if keypoints_list:
                return np.array(keypoints_list)  # 应该返回形状为 (N, 17, 3) 的数组
            return None
            
if __name__ == '__main__':
    main()



