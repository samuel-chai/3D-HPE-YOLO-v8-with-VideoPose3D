import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
path = os.path.split(os.path.realpath(__file__))[0]
main_path = os.path.join(path, '..')


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



def resize_img(frame, max_length=640):
    H, W = frame.shape[:2]
    if max(W, H) > max_length:
        if W>H:
            W_resize = max_length
            H_resize = int(H * max_length / W)
        else:
            H_resize = max_length
            W_resize = int(W * max_length / H)
        frame = cv2.resize(frame, (W_resize, H_resize), interpolation=cv2.INTER_AREA)
        return frame, W_resize, H_resize

    else:
        return frame, W, H


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


def draw_2Dimg(image, model_path):
    from yolo_detectors.yolov8_detection import Keypoint
    model_path = 'weights/yolov8x-pose.onnx'
    keypoint_detector = Keypoint(model_path)
    keypoints_list = keypoint_detector.getkptsFromImg(image)
    
    if keypoints_list is None or len(keypoints_list) == 0:
        print("沒有檢測到任何關鍵點")
        return image  
    
    for keypoints in keypoints_list:
        image = draw_2Dimg(image, keypoints)
    
    return image

def draw_3Dimg(pos, image, display=None, kpt2D=None):
    from mpl_toolkits.mplot3d import Axes3D # projection 3D 必须要这个
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    fig = plt.figure(figsize=(12,6))
    canvas = FigureCanvas(fig)

    # 2D
    fig.add_subplot(121)
    if isinstance(kpt2D, np.ndarray):
        plt.imshow(draw_2Dimg(image, kpt2D))
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
    parents = common.skeleton_parents
    joints_right = common.joints_right

    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue

        col = 'red' if j in joints_right else 'black'
        # 画图3D
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


def evaluate(test_generator, model_pos, action=None, return_predictions=False):
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()


def videoInfo(VideoName):
    cap = cv2.VideoCapture(VideoName)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, length

def videopose_model_load():
    # 加载已训练的模型
    from common.model import TemporalModel
    # load trained model
    chk_filename = main_path + '/checkpoint/cpn-pt-243.bin'
    checkpoint = torch.load(chk_filename, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model_pos = TemporalModel(17, 2, 17,filter_widths=[3,3,3,3,3] , causal=False, dropout=False, channels=1024, dense=False)
    model_pos = model_pos.cuda() if torch.cuda.is_available() else model_pos
    model_pos.load_state_dict(checkpoint['model_pos'])
    receptive_field = model_pos.receptive_field()
    return model_pos


def interface(model_pos, keypoints, W, H):
    # 检查并调整维度
    if len(keypoints.shape) == 4:  # 如果形状是 (N, 1, 17, 3)
        keypoints = keypoints.squeeze(1)  # 移除第二个维度，变成 (N, 17, 3)
        keypoints = keypoints[..., :2]    # 只取前两个坐标值，变成 (N, 17, 2)

    # 从 common.camera 导入用于坐标转换的函数
    from common.camera import normalize_screen_coordinates_new, camera_to_world, normalize_screen_coordinates
    # 将关键点进行归一化处理，使其适应网络输入要求
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=1000, h=1002)
    input_keypoints = keypoints.copy()
    
    # 使用 UnchunkedGenerator 创建非分块的数据生成器
    from common.generators import UnchunkedGenerator
    gen = UnchunkedGenerator(None, None, [input_keypoints], pad=common.pad, causal_shift=common.causal_shift, augment=True, kps_left=common.kps_left, kps_right=common.kps_right, joints_left=common.joints_left, joints_right=common.joints_right)
    
    # 使用 evaluate 函数进行预测
    prediction = evaluate(gen, model_pos, return_predictions=True)
    # 将预测结果从相机坐标系转换为世界坐标系
    prediction = camera_to_world(prediction, R=common.rot, t=0)
    # 调整 z 坐标，使最小值为零
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    return prediction
