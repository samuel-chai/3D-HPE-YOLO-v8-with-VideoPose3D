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
import onnxruntime as ort
import io
import torch.nn as nn
import torch
from itertools import zip_longest
from mpl_toolkits.mplot3d import Axes3D # projection 3D 必须要这个
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt



path = os.path.split(os.path.realpath(__file__))[0]
main_path = os.path.join(path, '..')


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utilsCombine import videoInfo, resize_img, draw_2Dimg, draw_3Dimg

class common():
    # 定义一个调色板数组，其中每个元素是一个包含RGB值的列表，用于表示不同的颜色
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])
    # 定义人体17个关键点的连接顺序，每个子列表包含两个数字，代表要连接的关键点的索引, 1鼻子 2左眼 3右眼 4左耳 5右耳 6左肩 7右肩 8左肘 9右肘 10左手腕 11右手腕 12左髋 13右髋 14左膝 15右膝 16左踝 17右踝
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], # 左右脚
                [12, 13], [6, 12], [7, 13], # 身體
                [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], # 手臂
                [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    # 通过索引从调色板中选择颜色，用于绘制人体骨架的线条，每个索引对应一种颜色
    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    # 通过索引从调色板中选择颜色，用于绘制人体的关键点，每个索引对应一种颜色
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    pad = (243 - 1) // 2 # Padding on each side
    causal_shift = 0
    
    # 对称关节的定义（左、右）
    kps_left = [5, 7, 9, 11, 13, 15]  # 左肩、左肘、左手腕、左髋、左膝、左踝
    kps_right = [6, 8, 10, 12, 14, 16]  # 右肩、右肘、右手腕、右髋、右膝、右踝
    
    # 定义左关节和右关节（这些是更详细的关节连接用于对称处理）
    joints_left = [5, 7, 9, 11, 13, 15]
    joints_right = [6, 8, 10, 12, 14, 16]
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

class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [ filter_widths[0] // 2 ]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out*3, 1)


    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2*frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = self._forward_blocks(x)

        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)

        return x

class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)

            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            # clip
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]

            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))

        x = self.shrink(x)
        return x

def videopose_model_load():
    # load trained model
    chk_filename = main_path + '/checkpoint/cpn-pt-243.bin'
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)# 把loc映射到storage
    model_pos = TemporalModel(17, 2, 17,filter_widths=[3,3,3,3,3] , causal=False, dropout=False, channels=1024, dense=False)
    model_pos = model_pos.cuda()
    model_pos.load_state_dict(checkpoint['model_pos'])
    receptive_field = model_pos.receptive_field()
    return model_pos

def wrap(func, *args, unsqueeze=False):
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

def qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X/w*2 - [1, h/w]

def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

class UnchunkedGenerator:
    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d

    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count

    def augment_enabled(self):
        return self.augment

    def set_augment(self, augment):
        self.augment = augment

    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d in zip_longest(self.cameras, self.poses_3d, self.poses_2d):
            print("Shape of seq_2d before padding:", seq_2d.shape)
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)

            if self.augment:
                # Append flipped version
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1

                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam, batch_3d, batch_2d

def evaluate(test_generator, model_pos, action=None, return_predictions=False):
    model_pos = model_pos.cuda() if torch.cuda.is_available() else model_pos
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
            predicted_3d_pos = model_pos(inputs_2d)
            if test_generator.augment_enabled():
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

def interface(model_pos, keypoints, W, H):

    # 检查并调整维度
    if len(keypoints.shape) == 4:  # 如果形状是 (N, 1, 17, 3)
        keypoints = keypoints.squeeze(1)  # 移除第二个维度，变成 (N, 17, 3)
        keypoints = keypoints[..., :2]    # 只取前两个坐标值，变成 (N, 17, 2)
    
    print("Adjusted keypoints shape:", keypoints.shape)
    
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=1000, h=1002)
    input_keypoints = keypoints.copy()
    # test_time_augmentation True
    gen = UnchunkedGenerator(None, None, [input_keypoints], pad=common.pad, causal_shift=common.causal_shift, augment=True, kps_left=common.kps_left, kps_right=common.kps_right, joints_left=common.joints_left, joints_right=common.joints_right)
    prediction = evaluate(gen, model_pos, return_predictions=True)
    prediction = camera_to_world(prediction, R=common.rot, t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    return prediction

# def draw_2Dimg(img, kpt, display=None):
#     # kpts : (17, 3) -> 17个关键点，每个关键点包含 (x, y, score)
#     im = img.copy()  # 复制输入图像，避免修改原图
    
#     # 将关键点重新排列为单个列表，以匹配 plot_skeleton_kpts() 的输入格式
#     kpts = []
#     for item in kpt:
#         if len(item) == 3 and isinstance(item[2], (int, float)):  # 确保 score 是标量值
#             kpts.extend([item[0], item[1], item[2]])  # 将 (x, y, score) 添加到列表中
#         else:
#             kpts.extend([item[0], item[1], 0.0])  # 如果没有 score，默认设为 0.0
    
#     # 调用 plot_skeleton_kpts() 函数来绘制人体关键点和骨架
#     plot_skeleton_kpts(im, kpts, pose_kpt_color=common.pose_kpt_color, pose_limb_color=common.pose_limb_color, skeleton=common.skeleton)
    
#     # 如果设置了 display 参数，则显示图像
#     if display:
#         cv2.imshow('im', im)
#         cv2.waitKey(3)

#     return im  # 返回绘制后的图像

def draw_keypoints_from_yolo(image, model_path):
    keypoint_detector = Keypoint(model_path)
    keypoints_list = keypoint_detector.getkptsFromImg(image)
    
    if keypoints_list is None or len(keypoints_list) == 0:
        print("沒有檢測到任何關鍵點")
        return image  
    
    for keypoints in keypoints_list:
        image = draw_2Dimg(image, keypoints)
    
    return image

def draw_3Dimg(pos, image, display=None, kpt2D=None):
    fig = plt.figure(figsize=(12,6))
    canvas = FigureCanvas(fig)

    model_path = 'weights/yolov8x-pose.onnx'
    # 2D
    fig.add_subplot(121)
    if isinstance(kpt2D, np.ndarray):
        plt.imshow(draw_keypoints_from_yolo(image, model_path))
    else:
        plt.imshow(image)

    # 3D
    ax = fig.add_subplot(122, projection='3d')
    radius = 1.7
    ax.view_init(elev=15., azim=70.)
    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    ax.set_aspect('equal')
    # 坐标轴刻度
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5
    skeleton = common.skeleton
    joints_right = common.joints_right

    # 確保 pos 是一個 NumPy 陣列
    pos = np.array(pos)

    # 檢查 pos 的形狀是否正確
    if len(pos.shape) != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos 必須是形狀 (N, 3)，但是得到的形狀是 {pos.shape}")

    # 繪製3D骨架
    for idx, (j, j_parent) in enumerate(skeleton):
        # 這裡的 `j` 和 `j_parent` 是基於1的索引，你需要減去1以適應 Python 的0基索引
        j -= 1
        j_parent -= 1

        # 檢查索引是否在範圍內
        if j < 0 or j_parent < 0 or j >= len(pos) or j_parent >= len(pos):
            continue

        # 根據 joints_right 設置顏色
        col = 'red' if j in joints_right or j_parent in joints_right else 'black'

        # 繪製3D骨架連接
        ax.plot([pos[j, 0], pos[j_parent, 0]],
                [pos[j, 1], pos[j_parent, 1]],
                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)
    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas.draw()       # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    if display:
        cv2.imshow('im', image)
        cv2.waitKey(3)

    return image


def main(VideoName):
    model_path = 'weights/yolov8x-pose.onnx'
    keydet = Keypoint(model_path)
    model3D = videopose_model_load()

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
                
        joint3D = interface(model3D, np.array(kpt2Ds), W, H)
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

