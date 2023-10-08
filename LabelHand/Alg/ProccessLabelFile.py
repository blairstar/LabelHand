import traceback

import numpy as np
import pandas as pd
import json
import glob
import os, sys
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from LabelHand.Alg.ManoTx import MANO, recify_finger_pose, render_two_hand, render, local_time
from LabelHand.Alg.MiscFun import cam2pixel, vis_sample, world2cam, load_dataset
from LabelHand.widgets.HandLabelWidget import HandPoseParam, HandBetaParam, HandGlobalParam
import torch
from multiprocessing import queues
import multiprocessing
import glob


def get_mano_param(pose_param, beta_param, is_rhand):
    xy, depth, root_pose, finger_pose = pose_param.to_mano_param(is_rhand)
    beta, recify_rot, ori_root_pos, ori_finger_root_pos = beta_param.to_mano_param()
    finger_pose = recify_finger_pose(finger_pose, recify_rot)
    return xy, depth, root_pose, finger_pose, beta, ori_root_pos


def export_label_file(right_mano, left_mano, label_path):
    param = json.load(open(label_path))
    
    right_pose_param, left_pose_param = HandPoseParam(), HandPoseParam()
    right_beta_param, left_beta_param = HandBetaParam(), HandBetaParam()
    global_hand_param = HandGlobalParam()
    
    right_pose_param.from_dict(param["right_hand_param"])
    right_beta_param.from_dict(param["right_hand_param"])
    left_pose_param.from_dict(param["left_hand_param"])
    left_beta_param.from_dict(param["left_hand_param"])
    global_hand_param.from_dict(param["global_hand_param"])
    
    right_hand_param = get_mano_param(right_pose_param, right_beta_param, True)
    left_hand_param = get_mano_param(left_pose_param, left_beta_param, False)
    global_param = global_hand_param.to_mano_param()
    
    right_draw_config = (False, True, False)
    left_draw_config = (False, True, False)

    img = cv2.imread(label_path.replace(".json", ".jpg"))
    img = img[:, :, ::-1]
    _, canva, joints, _ = render_two_hand(img, right_mano, left_mano, right_hand_param, left_hand_param, global_param, 
                                          right_draw_config, left_draw_config, False)
    
    left, top, right, bottom = global_param[2].tolist()
    width, height = right - left, bottom - top
    bbox = np.array([left, top, width, height])

    focal, princpt = global_param[0], global_param[1]
    princpt = princpt + np.array([left, top])
    
    mask = np.zeros_like(img)
    mask[top:top+height, left:left+width] = canva
    mask = (mask.mean(axis=-1) > 10).astype(np.uint8)
    mask = 100 * mask
    # mask = cv2.dilate(mask, np.ones([21, 21]))
    # mask = cv2.GaussianBlur(mask, ksize=(13, 13), sigmaX=0)
    return canva, joints, bbox, focal, princpt, mask


def export_label_file_tx():
    right_mano = MANO("./Model/MANO", True, False, flat_hand_mean=True)
    left_mano = MANO("./Model/MANO", False, False, flat_hand_mean=True)
    if torch.sum(torch.abs(left_mano.shapedirs[:, 0, :] - right_mano.shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        left_mano.shapedirs[:, 0, :] *= -1
        
    work_dir = "D:\\dev_data\\hand_pose_data\\01\\all_04"
    # work_dir = "D:\dev_data\hand_pose_data\label_return\\all_06"
    # dst_dir = "D:\\dev_data\\hand_pose_data\\01\\misc\\"
    # work_dir = "/cfs/cfs-oom9s50f/users/blairzheng/codes/LabelHand/data/00"
    ls_path = glob.glob(os.path.join(work_dir, "*.json"))
    idx = int(os.path.basename(work_dir).split("_")[-1])
    roi_columns = ["file_name", "bbox", "world_coord", "campos", "camrot", "focal", "princpt", "joint_valid",
                   "hand_type", "hand_type_valid"]
     
    ls_info = []
    for path in ls_path:
        # if os.path.basename(path) != "016_165317_76101.json":
        #     continue
        print(path)
        filename = path.replace(".json", ".jpg")
        canva, joints, bbox, focal, princpt, mask = export_label_file(right_mano, left_mano, path)
        cv2.imwrite(path.replace(".json", "_mask.jpg"), mask)
        # draw skeleton
        # img = cv2.imread(filename)
        # img_coord = cam2pixel(joints, focal, princpt)[:, :2]
        # img_skl = vis_sample(img, img_coord, np.ones(img_coord.shape[0]), offset=0)
        # dst_path = os.path.join(dst_dir, os.path.basename(path).replace(".json", ".jpg"))
        # cv2.imwrite(dst_path, canva[:, :, ::-1])
        # cv2.imwrite(dst_path.replace(".jpg", "_x.jpg"), img_skl)
        # break
        
        joint_valid = np.ones(joints.shape[0]).tolist()
        campos = [0.0, 0.0, 0.0]
        camrot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ls_info.append((os.path.basename(filename), bbox.tolist(), joints.tolist(), campos, camrot, focal.tolist(),
                        princpt.tolist(), joint_valid, "interacting", [1.0]))
    
    df = pd.DataFrame(ls_info, columns=roi_columns)
    print(df)
    df.to_csv(os.path.join(work_dir, f"hand_pose_{idx}.csv"))
    # df.to_csv("/cfs/cfs-oom9s50f/users/blairzheng/codes/hand_3d_couple_pose/datatool/annotations/train/hand_pose_0.csv")
    return


@torch.no_grad()
def calc_2p6_mano_vertices(ann, mano, capture, frame_idx, camrot, campos):
    key = "right" if mano.is_rhand else "left"
    param = ann[str(capture)][str(frame_idx)][key]
    betas = torch.FloatTensor(param['shape']).view(1, -1)
    pose = torch.FloatTensor(param['pose'])
    transl = torch.FloatTensor(param['trans']).view(1, -1)
    root_pose, hand_pose = pose[:3].view(1, -1), pose[3:].view(1, -1)
    
    output = mano.forward(betas, root_pose, hand_pose, transl)

    vertices = torch.matmul(torch.from_numpy(camrot[None, ...]),
                            (output.vertices * 1000 - torch.from_numpy(campos)).permute(0, 2, 1))
    vertices = vertices.permute(0, 2, 1)
    return vertices[0]


def generate_2p6_mask():
    img_dir = "/cfs/cfs-oom9s50f/users/yueya/data_raw/InterHand2.6M/InterHand2.6M_5fps_batch1/images/train"
    model_path = "/cfs/cfs-oom9s50f/users/blairzheng/codes/LabelHand/Model/MANO"
    ann_path = "/cfs/cfs-oom9s50f/users/blairzheng/codes/InterHand/data/InterHand2.6M/annotations/train/"

    data = load_dataset("train")
    ann = json.load(open(os.path.join(ann_path, "InterHand2.6M_train_MANO_NeuralAnnot.json")))
    roi_columns = ["file_name", "world_coord", "camrot", "campos", "focal", "princpt", "capture", "frame_idx"]
    right_mano = MANO(model_path, True, False)
    left_mano = MANO(model_path, False, False)
    if torch.sum(torch.abs(left_mano.shapedirs[:, 0, :] - right_mano.shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        left_mano.shapedirs[:, 0, :] *= -1
    right_face, left_face = right_mano.faces, left_mano.faces
    left_face = left_face + 778

    data = data.sample(frac=1.0)
    
    for ii in data.index:
        # ii = ii + 55644
        roi_vals = data.loc[ii, roi_columns]
        # roi_vals = data.loc[ii+313477, roi_columns]
        file_name, world_coord, camrot, campos, focal, princpt, capture, frame_idx = roi_vals
        
        img_path = os.path.join(img_dir, file_name)
        assert(img_path.endswith(".jpg"))
        mask_path = img_path[:-4] + "_mask.jpg"
        if os.path.exists(mask_path):
            continue
        print(local_time(), ii, img_path)
        
        campos, camrot = np.array(json.loads(campos)), np.array(json.loads(camrot))
        focal, princpt = np.array(json.loads(focal)), np.array(json.loads(princpt))
        
        try:
            right_vertice = calc_2p6_mano_vertices(ann, right_mano, capture, frame_idx, camrot, campos)
            left_vertice = calc_2p6_mano_vertices(ann, left_mano, capture, frame_idx, camrot, campos)
        except:
            traceback.print_exc()
            continue
        
        img = cv2.imread(img_path)
     
        vertice = torch.vstack([right_vertice, left_vertice])
        face = torch.vstack([right_face, left_face])
        focal, princpt = torch.from_numpy(focal), torch.from_numpy(princpt)
        img_out, mask = render(img.shape[:2], vertice, face, focal, princpt, 3, "open3d", False)
        # img_out, mask = img_out, mask * 1.0
        # img_merge = (img * (1 - mask) + img_out * mask)
         
        cv2.imwrite(mask_path, (mask*100).astype(np.uint8))
        # cv2.imwrite(img_path[:-4]+"_mask.jpg", img_merge)
        # break
        
    return


# def generate_2p6_mask_parallel_tx():
#     data = load_dataset("train")
#     print(data)
#     ls_process = []
#     process_count = 3
#     sample_count = len(data)
#     step = int(sample_count / process_count) + process_count
#     for ii, st in enumerate(range(0, sample_count, step)):
#         process = multiprocessing.Process(target=generate_2p6_mask,
#                                           args=(data[st:st+step],))
#         process.start()
#         ls_process.append(process)
# 
#     for process in ls_process:
#         process.join()
#     
#     return


def update_label_file_tx():
    import glob
    # work_dir = "D:\\dev_data\\hand_pose_data\\01\\012\\shipin\\VID_20221026_164422"
    # work_dir = "D:\\codes\\WeSee\\LabelHand\\Template\\"
    work_dir = "D:\\dev_data\\hand_pose_data\\01\\all\\"
    ls_path = glob.glob(os.path.join(work_dir, "*.json"))
    for path in ls_path:
        param = json.load(open(path, encoding="utf-8-sig"))
        # for key, value in param["left_hand_param"].items():
        #     if key in ["sld_index0", "sld_middle0", "sld_ring0", "sld_pinky0"]:
        #         param["left_hand_param"][key] = value * -1
        # param["global_hand_param"]["image_width"] = param["imageWidth"]
        # param["global_hand_param"]["image_height"] = param["imageHeight"]
        
        for hand in ["right_hand_param", "left_hand_param"]:
            for finger in ["index", "middle", "ring", "pinky"]:
                param[hand][f"{finger}0_pitch"] = param[hand][f"sld_{finger}0"]
                param[hand][f"{finger}0_roll"] = 0
                param[hand].pop(f"sld_{finger}0")
        
        json.dump(param, open(path, "w"))
    return


if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")
    export_label_file_tx()
    # generate_2p6_mask()
    # generate_2p6_mask_parallel_tx()
    # update_label_file_tx()