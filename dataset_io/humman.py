import os
import copy
import json
import numpy as np
from scipy.spatial.transform import Rotation

class HuMMan:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_root = os.path.join(cfg.dataset_path, cfg.subject)
        self.frame_range = range(cfg.frames.start, cfg.frames.stop, cfg.frames.step)
        if isinstance(cfg.pred_views, list):
            self.pred_views = cfg.pred_views
        elif isinstance(cfg.pred_views, int):
            self.pred_views = [cfg.pred_views]
        else:
            self.pred_views = eval(cfg.pred_views)

    def get_cams(self):
        return copy.deepcopy(self.pred_views)
    
    def get_frames(self):
        return copy.deepcopy(self.frame_range)

    def get_cam_imgs(self, cam):
        return [
            os.path.join(self.data_root, f'kinect_color/kinect_{cam:03d}/{frame:06d}.png')
            for frame in self.frame_range
        ]

    def save_results(self, data):
        cfg_o = self.cfg.output
        cam_info = data['cam_info']
        params = data['params']

        os.makedirs(os.path.join(self.data_root, cfg_o.smpl_params), exist_ok=True)
        for frame, param in zip(self.frame_range, params):
            full_poses = param['full_poses']
            betas = param['betas']
            trans = param['trans']
            # vertices = param['vertices']

            poses = np.ravel([Rotation.from_matrix(mat).as_rotvec() for mat in full_poses])

            result = {
                'betas': betas,
                'body_pose': poses[3:],
                'global_orient': poses[:3],
                'transl': trans,
            }
            
            np.savez(os.path.join(self.data_root, cfg_o.smpl_params, f'{frame:06}.npz'), **result)

        cameras = {
                f'kinect_color_{cam:03d}': {
                    'K': info['K'].tolist(),
                    'R': info['R'].tolist(),
                    'T': info['T'].tolist(),
                }
                for cam, info in cam_info.items()
        }
        with open(os.path.join(self.data_root, cfg_o.cameras), 'w') as fp:
            json.dump(cameras, fp)