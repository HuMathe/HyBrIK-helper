import yaml
from easydict import EasyDict as edict
import copy

import numpy as np

import argparse
import os

from estimate import estimate
import pickle

from scipy.spatial.transform import Rotation

from third_parties.smpl.smpl_numpy import SMPL

parser = argparse.ArgumentParser(description='HybrIK Helper')
parser.add_argument('--cfg', help='path to the data config file', required=True, type=str)
opt = parser.parse_args()

smpl_model = SMPL(sex='neutral', model_dir='third_parties/smpl/models/')


def load_config_file(file_path:str):
    with open(file_path, 'r') as fp:
        config = edict(yaml.load(fp, yaml.FullLoader))
    return config

def load_dataset_mod(module_name:str):

    from importlib import import_module as impmod

    comp_path = module_name.split('.')
    assert len(comp_path) >=2, \
      'module xxx in file.py should be formated as module="file.xxx" in config' 
    
    if len(comp_path) == 2:
        data_io_module = getattr(impmod(comp_path[0]), comp_path[1])
    else:
        data_io_module = getattr(impmod('.' + comp_path[-2], '.'.join(comp_path[:-2])), comp_path[-1])

    assert hasattr(data_io_module, 'get_cams')
    assert hasattr(data_io_module, 'get_frames')
    assert hasattr(data_io_module, 'get_cam_imgs')
    assert hasattr(data_io_module, 'save_results')

    return data_io_module

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def get_camera(hybrik_data: dict, index):
    bbox = hybrik_data['bbox'][index].copy()
    h, w = hybrik_data['height'][index], hybrik_data['width'][index]
    R = hybrik_data['pred_thetas'][index].reshape((24, 3, 3)).copy()[0]
    T = hybrik_data['transl'][index].copy()
    
    beta = hybrik_data['pred_betas'][index].copy()

    _, jts = smpl_model(np.zeros(72), beta)

    bbox_xywh = xyxy2xywh(bbox)
    focal = 1000 / 256 * bbox_xywh[2]
    K = np.eye(3)

    K[[0, 1], [0, 1]] = np.array([focal, focal])
    K[[0, 1], 2] = np.array([w, h]) / 2

    T -= R @ jts[0]

    return K, R, T

def merge_data(pose_data, main_cam, frame_range):    
    # for cam, pose in pose_data.items():
    cam_info = {
    k: {
        'K': np.zeros((3, 3)),
        'R': np.zeros((3, 3)),
        'T': np.zeros(3)
    }
    for k in set(list(pose_data.keys()) + [main_cam])
    }

    params = []

    cam_list = list(set(pose_data.keys()).difference({main_cam}))
    data_main = pose_data[main_cam]
    for idx, frame in enumerate(frame_range):
        # for cam in cam_list:
        full_pose = data_main['pred_thetas'][idx].copy().reshape((24, 3, 3))
        betas = data_main['pred_betas'][idx].copy()
        trans = data_main['transl'][idx].copy()
        vertices, joints = smpl_model(full_pose, betas)
        vertices = vertices + trans - joints[0]

        param = {
            'full_poses': full_pose,    # [24 ,3, 3]
            'betas': betas,             # [10]
            'trans': trans - np.dot(full_pose[0], joints[0]), 
                                        # [3]
            'vertices': vertices        # [6890, 3]
        }

        params.append(param)
        K0, R0, T0 = get_camera(pose_data[main_cam], idx)
        cam_info[main_cam]['K'] += K0
        cam_info[main_cam]['R'] += np.eye(3)
        cam_info[main_cam]['T'] += np.zeros(3)

        for cam in cam_list:
            K, R, T = get_camera(pose_data[cam], idx)
            R = R.dot(R0.T)
            T = T - (R @ T0)

            cam_info[cam]['K'] += K
            cam_info[cam]['R'] += R
            cam_info[cam]['T'] += T
        
    num_frames = len(frame_range)
    for cam in set(cam_list + [main_cam]):
        cam_info[cam]['K'] /= num_frames
        cam_info[cam]['R'] = Rotation.from_matrix(cam_info[cam]['R'] / num_frames).as_matrix()
        # cam_info[cam]['R'] = cam_info[cam]['R'] / num_frames
        cam_info[cam]['T'] *= 1000/num_frames

    return {
        'cam_info': cam_info,
        'params': params
    }



def main():

    # GetSubList, GetCamList, GetFrameList, GetImagePath = load_retrieve_funcs(opt.fmt_dir, opt.dataset)
    
    conf = load_config_file(opt.cfg)
    DataIO = load_dataset_mod(conf.dataset_io_module)
    dataset = DataIO(conf)

    print('Processing...')

    pose_data = {}
    for cam in dataset.get_cams():
        images = dataset.get_cam_imgs(cam)
        print(f'estimating smpl params under camera: {cam} | {len(images)} frames in total')
        result = estimate(images)
        pose_data[cam] = result

    print('Saving...')
    data = merge_data(pose_data, conf.main_view, dataset.get_frames())
    dataset.save_results(data)

    print('Done.')

if __name__ == '__main__':
    main()