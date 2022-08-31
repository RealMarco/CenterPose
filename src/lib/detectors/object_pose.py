# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import os
import shutil
from os.path import exists
import glob
import simplejson as json

from lib.models.decode import object_pose_decode, _nms, _topk, _transpose_and_gather_feat
from lib.utils.post_process import object_pose_post_process

from .base_detector import BaseDetector

import xlwt   # xlrd, xlutils
import math
import scipy
from scipy.spatial.transform import Rotation as rotation_util
#import objectron.dataset.box as Box
import tools.objectron_eval.objectron.dataset.box as Box

def soft_nms_nvidia(src_boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    N = src_boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

    for i in range(N):
        maxscore = src_boxes[i]['score']
        maxpos = i

        tx1 = src_boxes[i]['bbox'][0]
        ty1 = src_boxes[i]['bbox'][1]
        tx2 = src_boxes[i]['bbox'][2]
        ty2 = src_boxes[i]['bbox'][3]
        ts = src_boxes[i]['score']

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < src_boxes[pos]['score']:
                maxscore = src_boxes[pos]['score']
                maxpos = pos
            pos = pos + 1

        # add max box as a detection

        src_boxes[i]['bbox'] = src_boxes[maxpos]['bbox']
        src_boxes[i]['score'] = src_boxes[maxpos]['score']

        # swap ith box with position of max box
        src_boxes[maxpos]['bbox'] = [tx1, ty1, tx2, ty2]
        src_boxes[maxpos]['score'] = ts

        for key in src_boxes[0]:
            if key is not 'bbox' and key is not 'score':
                tmp = src_boxes[i][key]
                src_boxes[i][key] = src_boxes[maxpos][key]
                src_boxes[maxpos][key] = tmp

        tx1 = src_boxes[i]['bbox'][0]
        ty1 = src_boxes[i]['bbox'][1]
        tx2 = src_boxes[i]['bbox'][2]
        ty2 = src_boxes[i]['bbox'][3]
        ts = src_boxes[i]['score']

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:

            x1 = src_boxes[pos]['bbox'][0]
            y1 = src_boxes[pos]['bbox'][1]
            x2 = src_boxes[pos]['bbox'][2]
            y2 = src_boxes[pos]['bbox'][3]
            s = src_boxes[pos]['score']

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    src_boxes[pos]['score'] = weight * src_boxes[pos]['score']

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if src_boxes[pos]['score'] < threshold:

                        src_boxes[pos]['bbox'] = src_boxes[N - 1]['bbox']
                        src_boxes[pos]['score'] = src_boxes[N - 1]['score']

                        for key in src_boxes[0]:
                            if key is not 'bbox' and key is not 'score':
                                tmp = src_boxes[pos][key]
                                src_boxes[pos][key] = src_boxes[N - 1][key]
                                src_boxes[N - 1][key] = tmp

                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep

class ObjectPoseDetector(BaseDetector):
    def __init__(self, opt):
        super(ObjectPoseDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx
        # To save the yaw_CenterPose_pred in a spreadsheet
        self.row_idx = 0
        self.obj=xlwt.Workbook(encoding='utf-8',style_compression=0)
        self.sheet=self.obj.add_sheet('CenterPose_shoes_yaws',cell_overwrite_ok=True)
        self.sheet.col(0).width=512*20
        self.sheet.write(0,0,'img_name')    # write(row,col,value)
        self.sheet.write(0,1,'yaw_CenterPose_pred')

    def process(self, images, pre_images=None, pre_hms=None, pre_hm_hp=None,
                pre_inds=None, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images, pre_images, pre_hms, pre_hm_hp)[-1]
            output['hm'] = output['hm'].sigmoid_()
            if self.opt.hm_hp and not self.opt.mse_loss:
                output['hm_hp'] = output['hm_hp'].sigmoid_()
            output.update({'pre_inds': pre_inds})

            wh = output['wh'] if self.opt.reg_bbox else None
            reg = output['reg'] if self.opt.reg_offset else None
            hps_uncertainty = output['hps_uncertainty'] if self.opt.hps_uncertainty else None
            hm_hp = output['hm_hp'] if self.opt.hm_hp else None
            hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
            obj_scale = output['scale'] if self.opt.obj_scale else None
            obj_scale_uncertainty = output['scale_uncertainty'] if self.opt.obj_scale_uncertainty else None
            tracking = output['tracking'] if 'tracking' in self.opt.heads else None
            tracking_hp = output['tracking_hp'] if 'tracking_hp' in self.opt.heads else None

            torch.cuda.synchronize()
            forward_time = time.time()

            dets = object_pose_decode(
                output['hm'], output['hps'], wh=wh, kps_displacement_std=hps_uncertainty, obj_scale=obj_scale,
                obj_scale_uncertainty=obj_scale_uncertainty,
                reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, tracking=tracking, tracking_hp=tracking_hp, opt=self.opt,
                Inference=True)

            for k in dets:
                dets[k] = dets[k].detach().cpu().numpy()
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):

        dets = object_pose_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt, Inference=True)

        if scale != 1:
            for i in range(len(dets[0])):
                for k in ['bbox', 'kps', 'kps_displacement_std', 'tracking', 'tracking_hp', 'kps_displacement_mean',
                          'kps_heatmap_mean']:
                    if k in dets[0][i]:
                        dets[0][i][k] = (np.array(
                            dets[0][i][k], np.float32) / scale).tolist()

        # Todo: We use 0 here, since we only work on a single category, need to be updated
        return dets[0]

    def merge_outputs(self, detections):

        # Todo: We use 0 here, since we only work on a single category, need to be updated
        # Group all the detection result from different scales on a single image (We only deal with one iamge input one time)
        results = []
        for det in detections[0]:
            if det['score'] > self.opt.vis_thresh:
                results.append(det)
        results = np.array(results)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            keep = soft_nms_nvidia(results, Nt=0.5, method=2, threshold=self.opt.vis_thresh)
            results = results[keep]

        return results

    def debug(self, debugger, images, dets, output, scale=1, pre_hms=None, pre_hm_hp=None):

        # It will not affect the original dets value as we deepcopy it here
        dets['bboxes'] *= self.opt.down_ratio
        dets['kps'] *= self.opt.down_ratio
        dets['kps_displacement_mean'] *= self.opt.down_ratio
        dets['kps_displacement_std'] *= self.opt.down_ratio
        dets['kps_heatmap_mean'] *= self.opt.down_ratio

        if 'tracking' in self.opt.heads:
            dets['tracking'] *= self.opt.down_ratio
        if 'tracking_hp' in self.opt.heads:
            dets['tracking_hp'] *= self.opt.down_ratio

        # Save heatmap
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
                               img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())

        debugger.add_blend_img(img, pred, 'out_hm_pred')

        # For tracking_pred, temporary use
        pred = debugger.gen_colormap(np.zeros_like(output['hm'][0].detach().cpu().numpy()))
        debugger.add_blend_img(img, pred, 'out_tracking_pred')

        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'out_hmhp_pred')

        debugger.add_img(img, img_id='out_kps_processed_pred')
        heat = output['hm']
        K = 100
        heat = _nms(heat)
        scores, inds, clses, ys, xs = _topk(heat, K=K)  # inds: index in a single heatmap
        for i in range(K):

            if scores[0][i] > self.opt.vis_thresh:

                debugger.add_coco_bbox(dets['bboxes'][0][i], 0, show_txt=False, img_id='out_hm_pred')

                # Save pred_kps for debug
                if self.opt.hps_uncertainty:
                    debugger.add_coco_hp_paper(dets['bboxes'][0, i], dets['kps_displacement_mean'][0, i],
                                               dets['kps_displacement_std'][0, i], img_id='out_hmhp_pred')
                else:
                    debugger.add_coco_hp_paper(dets['bboxes'][0, i], dets['kps_displacement_mean'][0, i],
                                               img_id='out_hmhp_pred')

                if self.opt.tracking == True:
                    debugger.add_arrow(
                        [(dets['bboxes'][0, i, 0] + dets['bboxes'][0, i, 2]) / 2,
                         (dets['bboxes'][0, i, 1] + dets['bboxes'][0, i, 3]) / 2, ],
                        dets['tracking'][0, i],
                        img_id='out_tracking_pred')

                if self.opt.tracking_hp == True:
                    for idx in range(8):
                        if dets['kps'][0, i][idx * 2] == 0 and dets['kps'][0, i][idx * 2 + 1] == 0:
                            continue
                        debugger.add_arrow(
                            dets['kps'][0, i][idx * 2:idx * 2 + 2],
                            dets['tracking_hp'][0, i][idx * 2:idx * 2 + 2],
                            img_id='out_tracking_pred', c=(255, 255, 0))

                # Save peak from displacement and heatmap, two 3D bboxes will be drawn
                debugger.add_coco_hp(dets['kps_displacement_mean'][0, i], img_id='out_kps_processed_pred',
                                     pred_flag='gt')
                debugger.add_coco_hp(dets['kps_heatmap_mean'][0, i], img_id='out_kps_processed_pred', pred_flag='pred')

        if self.pre_images is not None:
            pre_img = self.pre_images[0].detach().cpu().numpy().transpose(1, 2, 0)
            pre_img = np.clip(((
                                   pre_img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        if pre_hms is not None:
            pre_hm = debugger.gen_colormap(pre_hms[0].detach().cpu().numpy())
            debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')
        if pre_hm_hp is not None:
            pre_hmhp = debugger.gen_colormap_hp(pre_hm_hp[0].detach().cpu().numpy())
            debugger.add_blend_img(pre_img, pre_hmhp, 'pre_hmhp')

    def show_results(self, debugger, image, results):
        '''
        when opt.debug >= 1 and opt.debug < 4:
        Parameters
        ----------
        debugger : TYPE
            DESCRIPTION.
        image : TYPE
            DESCRIPTION.
        results : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        debugger.add_img(image, img_id='out_img_pred')
        for bbox in results:
            if bbox['score'] > self.opt.vis_thresh:
                if self.opt.tracking_task == True:
                    if 'kps_pnp_kf' in bbox:
                        kps_disp = bbox['kps_pnp_kf'].copy()
                        kps_disp[:, 0] = kps_disp[:, 0] * image.shape[1]
                        kps_disp[:, 1] = kps_disp[:, 1] * image.shape[0]
                        kps_disp = kps_disp[1:, ]
                        kps_disp = kps_disp.reshape(-1, 1).flatten()
                        debugger.add_coco_hp(kps_disp, img_id='out_img_pred')
                        debugger.add_coco_bbox(bbox['bbox'], 0, bbox['score'], id=bbox['tracking_id'],
                                               img_id='out_img_pred')
                else:
                    if self.opt.reg_bbox:
                        debugger.add_coco_bbox(bbox['bbox'], 0, bbox['score'], img_id='out_img_pred')

                    if 'projected_cuboid' in bbox:
                        debugger.add_coco_hp(bbox['projected_cuboid'], img_id='out_img_pred', pred_flag='pnp')
                    else:
                        debugger.add_coco_hp(bbox['kps'], img_id='out_img_pred')
                if self.opt.obj_scale == True:
                    if self.opt.reg_bbox:
                        debugger.add_obj_scale(bbox['bbox'], bbox['obj_scale'], img_id='out_img_pred')
                        if self.opt.show_axes == True:
                            if self.opt.tracking_task == True:
                                debugger.add_axes(bbox['kps_3d_cam_kf'], self.opt.cam_intrinsic, img_id='out_img_pred')
                            else:
                                # Sometimes, pnp fails then no kps_3d_cam
                                if 'kps_3d_cam' in bbox:
                                    debugger.add_axes(bbox['kps_3d_cam'], self.opt.cam_intrinsic, img_id='out_img_pred')
                    else:
                        # Todo: A temporary location, need updates
                        debugger.add_obj_scale([20, 20, 0, 0], bbox['obj_scale'], img_id='out_img_pred')
        debugger.show_all_imgs(pause=self.pause)

    def save_results_eval(self, debugger, image, results, image_or_path_or_tensor, dict_out=None, video_layout=False):
        '''
        when opt.debug == 6

        Parameters
        ----------
        debugger : TYPE
            DESCRIPTION.
        image : TYPE
            DESCRIPTION.
        results : TYPE
            DESCRIPTION.
        image_or_path_or_tensor : TYPE
            DESCRIPTION.
        dict_out : TYPE, optional
            DESCRIPTION. The default is None.
        video_layout : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        debugger.add_img(image, img_id='out_img_pred')
        for bbox in results:
            if bbox['score'] > self.opt.vis_thresh:
                if self.opt.reg_bbox:
                    debugger.add_coco_bbox(bbox['bbox'], 0, bbox['score'], img_id='out_img_pred')

                    if 'projected_cuboid' in bbox:
                        debugger.add_coco_hp(bbox['projected_cuboid'], img_id='out_img_pred', pred_flag='pnp')

        # Todo: Right now, the path is hard-coded according to tracking task or not from eval_image
        if self.opt.tracking_task or (hasattr(self.opt, 'eval_max_num') and self.opt.eval_max_num == 100):
            video_layout = True

        # Todo: We hard-code the path for eval
        if not os.path.exists('demo/'):
            os.mkdir('demo/')

        root_dir_path = os.path.join('demo/', f'{os.path.splitext(os.path.basename(self.opt.load_model))[0]}')
        if not os.path.exists(root_dir_path):
            os.mkdir(root_dir_path)

        file_id_name = image_or_path_or_tensor[image_or_path_or_tensor.rfind('_') + 1:]
        folder_name = image_or_path_or_tensor[:image_or_path_or_tensor.rfind('_')]
        if video_layout:
            video_dir_path = os.path.join(root_dir_path, folder_name)
            if not os.path.exists(video_dir_path):
                os.mkdir(video_dir_path)
            debugger.save_all_imgs_eval(image_or_path_or_tensor, path=video_dir_path, video_layout=True)
            if dict_out is not None:
                with open(f"{video_dir_path}/{file_id_name}.json", 'w') as fp:
                    json.dump(dict_out, fp)
        else:
            debugger.save_all_imgs_eval(image_or_path_or_tensor, path=root_dir_path, video_layout=False)
            if dict_out is not None:
                with open(f"{root_dir_path}/{folder_name}_{file_id_name}.json", 'w') as fp:
                    json.dump(dict_out, fp)

    def save_results(self, debugger, image, results, image_or_path_or_tensor, dict_out=None):
        '''
        when opt.debug == 4

        Parameters
        ----------
        debugger : TYPE
            DESCRIPTION.
        image : TYPE
            DESCRIPTION.
        results : TYPE
            DESCRIPTION.
        image_or_path_or_tensor : TYPE
            DESCRIPTION.
        dict_out : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        debugger.add_img(image, img_id='out_img_pred')
        
        for bbox in results:
            if bbox['score'] > self.opt.vis_thresh:
                if self.opt.reg_bbox:

                    if not self.opt.paper_display:
                        if self.opt.tracking_task == True:
                            debugger.add_coco_bbox(bbox['bbox'], 0, bbox['score'], id=bbox['tracking_id'], img_id='out_img_pred')
                        else:
                            debugger.add_coco_bbox(bbox['bbox'], 0, bbox['score'], img_id='out_img_pred')

                    if 'projected_cuboid' in bbox:
                        if not self.opt.paper_display:
                            debugger.add_coco_hp(bbox['projected_cuboid'], img_id='out_img_pred', pred_flag='pnp')
                        else:
                            debugger.add_coco_hp(bbox['projected_cuboid'], img_id='out_img_pred', pred_flag='pnp', PAPER_DISPLAY=True)

                        if self.opt.show_axes == True or self.opt.paper_display == True:
                            if self.opt.tracking_task == True:
                                debugger.add_axes(bbox['kps_3d_cam_kf'], self.opt.cam_intrinsic, img_id='out_img_pred')
                                # print(bbox['kps_3d_cam_kf']) # tracking wasn't used
                            else:
                                debugger.add_axes(bbox['kps_3d_cam'], self.opt.cam_intrinsic, img_id='out_img_pred') # bbox['kps_3d_cam'] is a 9x3 array 
                                #print(image_or_path_or_tensor) # '../images/CenterPose/shoe/IMG_20211207_141437.jpg'
                                #print(image.shape)  #  h,w,c 
                                #print(bbox['kps_3d_cam']) # 9x3 array
                                
                                # compute euler angles by as_euler
                                #euler_angles =  self.compute_rotation(bbox['kps_3d_cam']) 
                                #print(euler_angles)
                                
                                # compute euler angles by 2D projection
                                yaw_pred = debugger.cal_rotation(bbox['kps_3d_cam'], self.opt.cam_intrinsic, image.shape[1], image.shape[0]) # yaw_pred is a number
                                # print(yaw_pred) # a number                                
                                
                                # To save the yaw_CenterPose_pred in a spreadsheet
                                # global row_idx
                                self.row_idx = self.row_idx + 1
                                self.sheet.write(self.row_idx,0,image_or_path_or_tensor.split('/')[-1]) 
                                self.sheet.write(self.row_idx,1,yaw_pred)
                                #self.obj.save(os.path.join(self.opt.demo_save,"ShoesYaws_CenterPose.xls"))
                                
        
                if not self.opt.paper_display:
                    if self.opt.obj_scale == True:
                        if self.opt.scale_pool == True:
                            obj_scale = bbox['obj_scale_kf']
                        else:
                            obj_scale = bbox['obj_scale']

                        if self.opt.reg_bbox:
                            debugger.add_obj_scale(bbox['bbox'], obj_scale, img_id='out_img_pred')
                        else:
                            # Todo: A temporary location, need updates
                            debugger.add_obj_scale([20, 20, 0, 0], obj_scale, img_id='out_img_pred')
        
        # To save the yaw_CenterPose_pred in a spreadsheet
        self.obj.save(os.path.join(self.opt.demo_save,"ShoesYaws_CenterPose.xls"))
        
        if os.path.isdir(self.opt.demo):
            # We set the saving folder's name = demo_save folder + source folder
            target_dir_path = os.path.join(self.opt.demo_save,
                                           f'{os.path.basename(self.opt.demo)}')
        else:
            # We set the saving folder's name = demo_save folder + source name
            target_dir_path = os.path.join(self.opt.demo_save,
                                           f'{os.path.splitext(os.path.basename(self.opt.demo))[0]}')
        if not os.path.exists(self.opt.demo_save):
            os.mkdir(self.opt.demo_save)

        if not os.path.exists(target_dir_path):
            os.mkdir(target_dir_path)

        debugger.save_all_imgs_demo(image_or_path_or_tensor, path=target_dir_path)

        if dict_out is not None:
            file_id_name = os.path.splitext(os.path.basename(image_or_path_or_tensor))[0]

            with open(f"{target_dir_path}/{file_id_name}.json", 'w') as fp:
                json.dump(dict_out, fp)
                
    def compute_rotation(self, box):
        
        # box in initial position and rotation
        size_x = np.linalg.norm(box[5] - box[1]) # x
        size_y = np.linalg.norm(box[3] - box[1]) # y
        size_z = np.linalg.norm(box[2] - box[1]) # z
        size = np.asarray([size_x, size_y, size_z])
        instance = Box.UNIT_BOX * size
        
        prediction = Box.Box(box)  # defined in objectron.dataset.box as Box
        annotation = Box.Box(instance)
        gt_rotation_inverse = np.linalg.inv(annotation.rotation) # matrix inversion
        rotation_error = np.matmul(prediction.rotation, gt_rotation_inverse) # matrix product
        #pred_rotation_inverse = np.linalg.inv(prediction.rotation) # matrix inversion
        #rotation_error = np.matmul(annotation.rotation, pred_rotation_inverse) # matrix product

        #error_angles = np.array(rotation_util.from_dcm(rotation_error).as_euler('zxy')) 
        error_angles = np.array(rotation_util.from_dcm(rotation_error).as_euler('zyx'))
        # from_dcm initialize from direction cosine matrix.    
        # as_euler('zxy') lower case 'zxy' stands for extrinsic rotations, i.e., roll, pitch, -yaw
        # as_euler('zyx') stands for roll -yaw pitch
        # the range of the 1st, 2nd, 3rd angles is [-180, 180], [-180, 180] and [-90, 90] ([0,180])
        print(error_angles*180/np.pi)  # 3x1
        
        
        HT = box[2]-box[1]
        HT0 = Box.UNIT_BOX[2] - Box.UNIT_BOX[1] #z0
        HT_cross = np.cross(HT0,HT)
        HT_dot   = np.dot(HT0,HT)
        yaw = np.arctan2(HT_cross, HT_dot)*180/np.pi # arctan2 is identical to the atan2 function of the underlying C library.
        print(yaw)
        
        '''
        abs_error_angles = np.absolute(error_angles)
        abs_error_angles = np.minimum(
            abs_error_angles, np.absolute(math.pi * np.ones(3) - abs_error_angles))
        error = np.linalg.norm(abs_error_angles) # 3 error angle to one 2-ord norm
        print(error) # 1x1
        
        # Compute the error as the angle between the two rotation
        # What are the differences between error and augular distance
        rotation_error_trace = abs(np.matrix.trace(rotation_error))
        angular_distance = math.acos((rotation_error_trace - 1.) / 2.)
        print(angular_distance) # 1x1
        # angle = 2 * acos(|q1.q2|)
        box_quat = np.array(rotation_util.from_dcm(prediction.rotation).as_quat())
        gt_quat = np.array(rotation_util.from_dcm(annotation.rotation).as_quat())
        quat_distance = 2 * math.acos(np.dot(box_quat, gt_quat))
        print(quat_distance) # 1x1
        # The rotation measure from "3D Bounding box estimation using deep learning
        # and geometry"
        rotation_error_log = scipy.linalg.logm(rotation_error)
        rotation_error_frob_norm = np.linalg.norm(rotation_error_log, ord='fro')
        rotation_distance = rotation_error_frob_norm / 1.4142
        print(rotation_distance) # 1x1
        '''

        #return (error, quat_distance, angular_distance, rotation_distance)
        return error_angles # ndarray, shape (3,) or (N, 3)
