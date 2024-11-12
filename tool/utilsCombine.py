import numpy as np
from itertools import zip_longest
import torch
import os
import cv2
import matplotlib.pyplot as plt


path = os.path.split(os.path.realpath(__file__))[0]
main_path = os.path.join(path, '..')


class common():
    keypoints_symmetry = [[1, 3, 5, 7, 9, 11, 13, 15],[2, 4, 6, 8, 10, 12, 14, 16]]
    rot = np.array([ 0.14070565, -0.15007018, -0.7552408 ,  0.62232804], dtype=np.float32)
    skeleton_parents =  np.array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    pairs = [(1,2), (5,4),(6,5),(8,7),(8,9),(10,1),(11,10),(12,11),(13,1),(14,13),(15,14),(16,2),(16,3),(16,4),(16,7)]

    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
    pad = (243 - 1) // 2 # Padding on each side
    causal_shift = 0
    joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                [5, 11], [6, 12], [11, 12],
                [11, 13], [12, 14], [13, 15], [14, 16]]


class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing. Sequences are returned one at a time (i.e. batch size = 1), 
    without chunking. If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2), 
    the second of which is a mirrored version of the first.
    """
    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0, augment=False, 
                 kps_left=None, kps_right=None, joints_left=None, joints_right=None):
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
        return sum(p.shape[0] for p in self.poses_2d)

    def augment_enabled(self):
        return self.augment

    def set_augment(self, augment):
        self.augment = augment

    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d in zip_longest(self.cameras, self.poses_3d, self.poses_2d):
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                                             ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                                             'edge'), axis=0)

            if self.augment:
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

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
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
    """
    Rotate vector(s) v about the rotation described by 四元数quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))


def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape)-1)


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

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]

def draw_2Dimg(img, kpt, display=None):
    # kpts : (17, 3)  3-->(x, y, score)
    im = img.copy()
    joint_pairs = common.joint_pairs
    for item in kpt:
        score = item[-1]
        if score > 0.1:
            x, y = int(item[0]), int(item[1])
            cv2.circle(im, (x, y), 1, (255, 5, 0), 5)
    for pair in joint_pairs:
        j, j_parent = pair
        pt1 = (int(kpt[j][0]), int(kpt[j][1]))
        pt2 = (int(kpt[j_parent][0]), int(kpt[j_parent][1]))
        cv2.line(im, pt1, pt2, (0,255,0), 2)

    if display:
        cv2.imshow('im', im)
        cv2.waitKey(3)
    return im

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


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

def videopose_model_load():
    # load trained model
    from common.model import TemporalModel
    chk_filename = main_path + '/checkpoint/cpn-pt-243.bin'
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)# 把loc映射到storage
    model_pos = TemporalModel(17, 2, 17,filter_widths=[3,3,3,3,3] , causal=False, dropout=False, channels=1024, dense=False)
    model_pos = model_pos.cuda()
    model_pos.load_state_dict(checkpoint['model_pos'])
    receptive_field = model_pos.receptive_field()
    return model_pos


def interface(model_pos, keypoints, W, H):
    """
    Interface function to preprocess keypoints, create a generator, and predict 3D poses.
    """
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints)

    # Normalize keypoints for the network input
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=W, h=H)
    input_keypoints = keypoints.copy()

    # Create unchunked generator for data
    gen = UnchunkedGenerator(None, None, [input_keypoints], pad=common.pad, causal_shift=common.causal_shift, augment=True, kps_left=common.kps_left, kps_right=common.kps_right, joints_left=common.joints_left, joints_right=common.joints_right)
    # Perform prediction
    prediction = evaluate(gen, model_pos, return_predictions=True)
    # Convert predictions to world coordinates
    prediction = camera_to_world(prediction, R=np.eye(3), t=0)
    # Adjust z coordinate so the minimum value is zero
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    return prediction

