import numpy as np
import sys
import time
import argparse
import cv2
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses
from enum import Enum
np.set_printoptions(threshold=sys.maxsize)

class VideoPoseBody(Enum):
    b_torso = 0
    l_hip = 1
    l_knee = 2
    l_foot = 3
    r_hip = 4
    r_knee = 5
    r_foot = 6
    c_torso = 7
    u_torso = 8
    neck = 9
    head = 10
    r_shoulder = 11
    r_elbow = 12
    r_hand = 13
    l_shoulder = 14
    l_elbow = 15
    l_hand = 16

class LightWeightBody(Enum):
    neck = 0
    nose = 1
    center = 2
    l_shoulder = 3
    l_elbow = 4
    l_wrist = 5
    l_hip = 6
    l_knee = 7
    l_ankle = 8
    r_shoulder = 9
    r_elbow = 10
    r_wrist = 11
    r_hip = 12
    r_knee = 13
    r_ankle = 14
    r_eye = 15
    l_eye = 16
    r_ear = 17
    l_ear = 18

def move_to_root(pose_3d):
    root_point = np.multiply(np.add(pose_3d[LightWeightBody.l_ankle.value], pose_3d[LightWeightBody.r_ankle.value]), np.array([0.5, 0.5, 0.5]))
    for i in range(len(pose_3d)):
        pose_3d[i] = np.subtract(pose_3d[i], root_point)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='keypoint reader')
    parser.add_argument('-r', '--read', default=None, type=str, help="read .npz/.npy keypoint file", metavar="FILE", required=True)
    parser.add_argument('-w', '--write', default=None, type=str, help="write .npz/.npy keypoint file", metavar="FILE", required=True)
    args = parser.parse_args()

    video_pose_keypoints = np.load(args.read, allow_pickle=True)
    light_weight_keypoints = np.zeros((video_pose_keypoints.shape[0], 19, 3))
    light_weight_keypoints[:,:-2] = video_pose_keypoints

    light_weight_sequence = [VideoPoseBody.neck.value,
                             VideoPoseBody.head.value,
                             -1,
                             VideoPoseBody.l_shoulder.value,
                             VideoPoseBody.l_elbow.value,
                             VideoPoseBody.l_hand.value,
                             VideoPoseBody.l_hip.value,
                             VideoPoseBody.l_knee.value,
                             VideoPoseBody.l_foot.value,
                             VideoPoseBody.r_shoulder.value,
                             VideoPoseBody.r_elbow.value,
                             VideoPoseBody.r_hand.value,
                             VideoPoseBody.r_hip.value,
                             VideoPoseBody.r_knee.value,
                             VideoPoseBody.r_foot.value,
                             VideoPoseBody.head.value,
                             VideoPoseBody.head.value,
                             VideoPoseBody.head.value,
                             VideoPoseBody.head.value]
    light_weight_keypoints = light_weight_keypoints[:, light_weight_sequence]

    light_weight_keypoints = np.multiply(light_weight_keypoints, 100)

    print(light_weight_keypoints)

    while True:
        for frame in light_weight_keypoints:
            move_to_root(frame)
            frame = frame.reshape(19*3)
            poses_3d_copy = frame.copy().reshape(19*3)
            x = poses_3d_copy[0::3]
            y = poses_3d_copy[1::3]
            z = poses_3d_copy[2::3]
            frame[0::3], frame[1::3], frame[2::3] = -z, -x, -y

            frame = frame.reshape(19, -1)[:, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(frame.reshape([1, 19, 3]).shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
            canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
            plotter = Plotter3d(canvas_3d.shape[:2])
            plotter.plot(canvas_3d, frame.reshape([1, 19, 3]), edges)
            cv2.imshow('Canvas 3D', canvas_3d)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)
        
        time.sleep(1)