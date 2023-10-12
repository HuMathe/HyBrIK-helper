import os
import copy
import torch
import numpy as np
import pickle as pk
from scipy.spatial.transform import Rotation
import open3d as o3d

class GeneBody:
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
            os.path.join(self.data_root, 'image', f'{cam:02d}', f'{frame:04d}.jpg')
            for  frame in self.frame_range
        ]
    

    def save_results(self, data):
        cfg_o = self.cfg.output
        cam_info = data['cam_info']
        params = data['params']
        
        with open('model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', 'rb') as fp:
            faces = pk.load(fp, encoding='latin1')['f']

        os.makedirs(os.path.join(self.data_root, cfg_o.smpl_param), exist_ok=True)
        for frame, param in zip(self.frame_range, params):
            full_poses = param['full_poses']
            betas = param['betas']
            trans = param['trans']
            vertices = param['vertices']

            result = {
                'transl': torch.from_numpy(trans)[None, ...],
                'global_orient': torch.from_numpy(full_poses[:1])[None, ...],
                'body_pose': torch.from_numpy(full_poses[1:, ...])[None, ...],
                'betas': torch.from_numpy(betas)[None, ...],
                'vertices': torch.from_numpy(vertices)[None, ...],
                'joints': None,
                'full_pose': torch.from_numpy(full_poses)[None, ...],
                'v_shaped': None,
                'faces': faces.copy()
            }
            with open(os.path.join(self.data_root, cfg_o.smpl_param, f'{frame:04d}.pkl'), 'wb') as fp:
                pk.dump(result, fp)
            
            o3d.io.write_triangle_mesh(
                os.path.join(self.data_root, cfg_o.smpl_param, f'{frame:04d}.obj'), 
                o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(vertices), 
                    o3d.utility.Vector3iVector(faces)), 
                write_ascii=True
            )

        annots = {
            'cams': {
                f'{cam:02d}': {
                    'c2w': np.linalg.inv(np.vstack([np.hstack([info['R'], info['T'].reshape((3, 1))]), [0, 0, 0, 1]])),
                    'K': info['K'],
                    'c2w_R': info['R'].T,
                    'c2w_T': info['R'].T.dot(info['T']).reshape((3, 1)),
                    'D': None,
                }
                for cam, info in cam_info.items()
            },
        }
        np.save(os.path.join(self.data_root, cfg_o.annots), annots, allow_pickle=True)
