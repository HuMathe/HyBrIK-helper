"""Image demo script."""
import argparse
import os
import pickle as pk

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def estimate(images: list) -> dict:

    det_transform = T.Compose([T.ToTensor()])

    cfg_file = 'configs/hybrik/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
    CKPT = './pretrained_models/hybrik_hrnet.pth'
    cfg = update_config(cfg_file)

    bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
    bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
    dummpy_set = edict({
        'joint_pairs_17': None,
        'joint_pairs_24': None,
        'joint_pairs_29': None,
        'bbox_3d_shape': bbox_3d_shape
    })

    res_keys = [
        'pred_uvd',
        'pred_xyz_17',
        'pred_xyz_29',
        'pred_xyz_24_struct',
        'pred_scores',
        'pred_camera',
        # 'f',
        'pred_betas',
        'pred_thetas',
        'pred_phi',
        'pred_cam_root',
        # 'features',
        'transl',
        'transl_camsys',
        'bbox',
        'height',
        'width',
        'img_path',

        # 'vertices',
    ]
    res_db = {k: [] for k in res_keys}

    transformation = SimpleTransform3DSMPLCam(
        dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
        color_factor=cfg.DATASET.COLOR_FACTOR,
        occlusion=cfg.DATASET.OCCLUSION,
        input_size=cfg.MODEL.IMAGE_SIZE,
        output_size=cfg.MODEL.HEATMAP_SIZE,
        depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
        bbox_3d_shape=bbox_3d_shape,
        rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
        train=False, add_dpg=False,
        loss_type=cfg.LOSS['TYPE'])

    det_model = fasterrcnn_resnet50_fpn(pretrained=True)

    hybrik_model = builder.build_sppe(cfg.MODEL)

    # print(f'Loading model from {CKPT}...')
    save_dict = torch.load(CKPT, map_location='cpu')
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        hybrik_model.load_state_dict(model_dict)
    else:
        hybrik_model.load_state_dict(save_dict)

    det_model.cuda()
    hybrik_model.cuda()
    det_model.eval()
    hybrik_model.eval()

    img_path_list = images

    prev_box = None
    smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))

    # for img_path in img_path_list:
    for img_path in tqdm(img_path_list):
        basename = os.path.basename(img_path)

        with torch.no_grad():
            # Run Detection
            input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            det_input = det_transform(input_image).cuda()
            det_output = det_model([det_input])[0]

            if prev_box is None:
                tight_bbox = get_one_box(det_output)  # xyxy
                if tight_bbox is None:
                    continue
            else:
                tight_bbox = get_max_iou_box(det_output, prev_box)  # xyxy

            prev_box = tight_bbox

            # Run HybrIK
            # bbox: [x1, y1, x2, y2]
            pose_input, bbox, img_center = transformation.test_transform(
                input_image, tight_bbox)
            pose_input = pose_input.cuda()[None, :, :, :]
            pose_output = hybrik_model(
                pose_input, flip_test=True,
                bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
                img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()
            )
            uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]
            transl = pose_output.transl.detach()

            # Visualization
            image = input_image.copy()
            focal = 1000.0
            bbox_xywh = xyxy2xywh(bbox)
            transl_camsys = transl.clone()
            transl_camsys = transl_camsys * 256 / bbox_xywh[2]

            focal = focal / 256 * bbox_xywh[2]

            vertices = pose_output.pred_vertices.detach()

            verts_batch = vertices
            transl_batch = transl

            color_batch = render_mesh(
                vertices=verts_batch, faces=smpl_faces,
                translation=transl_batch,
                focal_length=focal, height=image.shape[0], width=image.shape[1])

            valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
            image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
            image_vis_batch = (image_vis_batch * 255).cpu().numpy()

            color = image_vis_batch[0]
            valid_mask = valid_mask_batch[0].cpu().numpy()
            input_img = image
            alpha = 0.9
            image_vis = alpha * color[:, :, :3] * valid_mask + (
                1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

            image_vis = image_vis.astype(np.uint8)
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

            # vis 2d
            pts = uv_29 * bbox_xywh[2]
            pts[:, 0] = pts[:, 0] + bbox_xywh[0]
            pts[:, 1] = pts[:, 1] + bbox_xywh[1]
            image = input_image.copy()
            bbox_img = vis_2d(image, tight_bbox, pts)
            bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)

            assert pose_input.shape[0] == 1, 'Only support single batch inference for now'

            pred_xyz_jts_17 = pose_output.pred_xyz_jts_17.reshape(
                17, 3).cpu().data.numpy()
            pred_uvd_jts = pose_output.pred_uvd_jts.reshape(
                -1, 3).cpu().data.numpy()
            pred_xyz_jts_29 = pose_output.pred_xyz_jts_29.reshape(
                -1, 3).cpu().data.numpy()
            pred_xyz_jts_24_struct = pose_output.pred_xyz_jts_24_struct.reshape(
                24, 3).cpu().data.numpy()
            pred_scores = pose_output.maxvals.cpu(
            ).data[:, :29].reshape(29).numpy()
            pred_camera = pose_output.pred_camera.squeeze(
                dim=0).cpu().data.numpy()
            pred_betas = pose_output.pred_shape.squeeze(
                dim=0).cpu().data.numpy()
            pred_theta = pose_output.pred_theta_mats.squeeze(
                dim=0).cpu().data.numpy()
            pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
            pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
            img_size = np.array((input_image.shape[0], input_image.shape[1]))

            res_db['pred_xyz_17'].append(pred_xyz_jts_17)
            res_db['pred_uvd'].append(pred_uvd_jts)
            res_db['pred_xyz_29'].append(pred_xyz_jts_29)
            res_db['pred_xyz_24_struct'].append(pred_xyz_jts_24_struct)
            res_db['pred_scores'].append(pred_scores)
            res_db['pred_camera'].append(pred_camera)
            res_db['pred_betas'].append(pred_betas)
            res_db['pred_thetas'].append(pred_theta)
            res_db['pred_phi'].append(pred_phi)
            res_db['pred_cam_root'].append(pred_cam_root)
            res_db['transl'].append(transl[0].cpu().data.numpy())
            res_db['transl_camsys'].append(transl_camsys[0].cpu().data.numpy())
            res_db['bbox'].append(np.array(bbox))
            res_db['height'].append(img_size[0])
            res_db['width'].append(img_size[1])
            res_db['img_path'].append(img_path)

            # DEBUG
            # res_db['vertices'].append(vertices[0].cpu().data.numpy())

    n_frames = len(res_db['img_path'])
    for k in res_db.keys():
        # print(k)
        res_db[k] = np.stack(res_db[k])
        assert res_db[k].shape[0] == n_frames

    return res_db
