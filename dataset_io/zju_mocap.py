import os
import copy
import numpy as np
from scipy.spatial.transform import Rotation

IMAGE_FMT = os.path.join('Camera_B{cam}', '{frame:06d}.jpg')

class ZJUMoCap:
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
        annots = np.load(os.path.join(self.data_root, self.cfg.annots), allow_pickle=True).item()
        return [
            os.path.join(self.data_root, annots['ims'][frame]['ims'][cam])
            for  frame in self.frame_range
        ]
    
    def save_results(self, data):
        cam_info = data['cam_info']
        params = data['params']

        cfg_o = self.cfg.output
        os.makedirs(os.path.join(self.data_root, cfg_o.params), exist_ok=True)
        os.makedirs(os.path.join(self.data_root, cfg_o.vertices), exist_ok=True)
        for frame, param in zip(self.frame_range, params):
            full_poses = param['full_poses']
            betas = param['betas']
            trans = param['trans']
            vertices = param['vertices']

            Rh = Rotation.from_matrix(full_poses[0]).as_rotvec()[None, ...]
            Th = trans[None, ...]
            poses = np.array([[0, 0, 0]] + [Rotation.from_matrix(mat).as_rotvec() for mat in full_poses[1:]]).ravel()[None, ...]
            shapes = betas[None, ...]

            np.save(os.path.join(self.data_root, cfg_o.params, f'{frame}.npy'), 
                    {
                        'Rh': Rh,
                        'Th': Th,
                        'poses': poses,
                        'shapes': shapes
                    }, 
                    allow_pickle=True)
            
            np.save(os.path.join(self.data_root, cfg_o.vertices, f'{frame}.npy'), vertices)

        annots = np.load(os.path.join(self.data_root, self.cfg.annots), allow_pickle=True).item()
        # annots = np.load(self.cfg.annots, allow_pickle=True).item()
        num_total_cam = len(annots['cams']['K'])
        annots['cams']['K'] = [None] * num_total_cam
        annots['cams']['T'] = [None] * num_total_cam
        annots['cams']['R'] = [None] * num_total_cam
        annots['cams']['D'] = [None] * num_total_cam
        for cam in self.get_cams():

            if self.cfg.subject in ['CoreView_313', 'CoreView_315']:
                annots['cams']['K'][cam] = cam_info[cam]['K'].tolist()
                annots['cams']['R'][cam] = cam_info[cam]['R'].tolist()
                annots['cams']['T'][cam] = cam_info[cam]['T'].reshape((3, 1)).tolist()
            else:
                annots['cams']['K'][cam] = cam_info[cam]['K']
                annots['cams']['R'][cam] = cam_info[cam]['R']
                annots['cams']['T'][cam] = cam_info[cam]['T'].reshape((3, 1))

        np.save(os.path.join(self.data_root, cfg_o.annots), annots, allow_pickle=True)

        