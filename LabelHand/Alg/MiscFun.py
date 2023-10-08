
import os, sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import multiprocessing
import glob

g_joint_parent_inv = [
    1, 2, 3, 20,
    5, 6, 7, 20,
    9, 10, 11, 20,
    13, 14, 15, 20,
    17, 18, 19, 20,
    20,
    22, 23, 24, 41,
    26, 27, 28, 41,
    30, 31, 32, 41,
    34, 35, 36, 41,
    38, 39, 40, 41,
    41
]

g_hori_line_inv = [(7, 11), (11, 15), (15, 19),
                   (28, 32), (32, 36), (36, 40)]

g_joint_parent = [
        0,
        0, 1, 2, 3,
        0, 5, 6, 7,
        0, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19,
        21,
        21, 22, 23, 24,
        21, 26, 27, 28,
        21, 30, 31, 32,
        21, 34, 35, 36,
        21, 38, 39, 40,
    ]
g_hori_line = [(5, 9), (9, 13), (13, 17),
               (26, 30), (30, 34), (34, 38)]


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord


def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord


def vis_sample(img, img_coord, joint_valid=None, inv_joint_order=False, offset=0, msg=""):
    joint_parent = g_joint_parent_inv if inv_joint_order else g_joint_parent
    root_joints = [20, 41] if inv_joint_order else [0, 21]
    
    joint_num = 42
    img = img.copy()
    cm = plt.get_cmap('gist_rainbow')
    for ii in range(img_coord.shape[0]):
        radius = 1 if ii not in root_joints else 2
        scale = 255 if joint_valid is not None and joint_valid[ii] > 0 else 70
        color = (np.array(cm((ii+offset) / joint_num)[:3]) * scale).tolist()[::-1]
        curr_pt, parent_pt = img_coord[ii].astype(int), img_coord[joint_parent[ii]].astype(int)
        cv2.circle(img, tuple(curr_pt), radius, tuple(color), thickness=2)
        cv2.line(img, tuple(curr_pt), tuple(parent_pt), color)

    if msg != "":
        cv2.putText(img, msg, (30, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 200, 0))

    return img


def vis_sample_tx():
    path = "/cfs/cfs-oom9s50f/users/blairzheng/codes/IntagHand/test/t0.png"

    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    assert (img.shape[0] == 256 and img.shape[1] == 256)
    
    csv_path = path.replace(".png", "_intag.csv")
    df = pd.read_csv(csv_path, index_col=0, dtype=float)
    cam_coord = df.to_numpy()
    
    vis_sample(img, cam_coord[:, :2])
    canva = img.copy()
    canva = vis_sample(canva, cam_coord[:, :2], np.ones(cam_coord.shape[0]), False)
    
    img_merge = np.hstack([img, canva])
    cv2.imwrite(path.replace(".png", "_output.png"), img_merge)
    
    coord_to_wrl([csv_path])
        
    return


def vis_label_sample_tx():
    output_dir = "/cfs/cfs-oom9s50f/users/blairzheng/codes/data/hp/01/"
    
    roi_columns = ["file_name", "bbox", "world_coord", "campos", "camrot", "focal", "princpt", "joint_valid",
                   "hand_type", "hand_type_valid", "fuse_coord"]
    
    ls_info = []
    dt = json.load(open("/cfs/cfs-oom9s50f/users/minglangma/work/IntagHand/wild_model_couple_hand_gt.json"))
    for key, val in dt.items():
        # if np.random.random() > 0.01:
        #     continue
            
        img_path = key.replace(".json", ".jpg")
        left_fuse_coord, right_fuse_coord = val["j25d_left"]["__ndarray__"], val["j25d_right"]["__ndarray__"]
        fuse_coord = np.concatenate([np.array(left_fuse_coord), np.array(right_fuse_coord)], axis=0)
        bbox = None
        for shape in json.load(open(key))["shape"]:
            if shape["label"] == "couple_hand":
                bbox = np.array(shape["points"]).flatten().astype(int)
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]
        assert(bbox is not None)
        
        # img = cv2.imread(img_path)
        
        # img = vis_sample(img, fuse_coord[:, :2], np.ones(fuse_coord.shape[0]))
        # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (200, 0, 200), thickness=2)
        # cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), img)
        
        world_coord = fuse_coord.copy()
        world_coord[:, :2] = fuse_coord[:, :2] * fuse_coord[:, 2:]
        joint_valid = np.ones(world_coord.shape[0]).tolist()
        campos = [0.0, 0.0, 0.0]
        camrot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        focal, princpt = [1.0, 1.0], [0.0, 0.0]
        ls_info.append((img_path, bbox.tolist(), world_coord.tolist(), campos, camrot, focal, princpt, joint_valid,
                        "interacting", [1.0], fuse_coord.tolist()))
        
    df = pd.DataFrame(ls_info, columns=roi_columns)
    print(df)
    df.to_csv("/cfs/cfs-oom9s50f/users/blairzheng/codes/hand_3d_couple_pose/datatool/annotations/train/hand_label.csv")
    
    return


def crop_with_bbox(img, bbox, fuse_coord, width_range=(0.10, 0.10), height_range=(0.075, 0.075)):
    left, top, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
    right, bottom = left + width, top + height
    
    fuse_coord = fuse_coord.copy()
    
    width_ratio = width_range[0] + np.random.random()*(width_range[1]-width_range[0])
    height_ratio = height_range[0] + np.random.random()*(height_range[1]-height_range[0])
 
    left_ext, right_ext = left - width*width_ratio, right + width*width_ratio
    top_ext, bottom_ext = top - height*height_ratio, bottom + height*height_ratio

    left_ext, right_ext = max(0, int(left_ext)), min(img.shape[1], int(right_ext))
    top_ext, bottom_ext = max(0, int(top_ext)), min(img.shape[0], int(bottom_ext))
    crop_rect = (left_ext, top_ext, right_ext-left_ext, bottom_ext-top_ext)
    
    img_crop = img[top_ext:bottom_ext, left_ext:right_ext, :]

    fuse_coord[:, 0] = fuse_coord[:, 0] - left_ext
    fuse_coord[:, 1] = fuse_coord[:, 1] - top_ext

    return img_crop, fuse_coord, (left_ext, top_ext), crop_rect


def resize_img_keep_ratio(img, dst_size, fuse_coord):
    src_size = (img.shape[1], img.shape[0])

    ratio_w, ratio_h = dst_size[0] / src_size[0], dst_size[1] / src_size[1]
    ratio = ratio_w if ratio_w < ratio_h else ratio_h
    new_width = int(src_size[0] * ratio)
    new_height = int(src_size[1] * ratio)

    new_size = (new_width, new_height)
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    pad_w = dst_size[0] - new_size[0]
    pad_h = dst_size[1] - new_size[1]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    
    if fuse_coord is not None:
        fuse_coord[:, 0] = left + fuse_coord[:, 0] * ratio
        fuse_coord[:, 1] = top + fuse_coord[:, 1] * ratio
        fuse_coord[:, 2] = fuse_coord[:, 2] * ratio

    return img_new, fuse_coord, ratio, (left, top), (left, top, new_width, new_height)


def load_dataset(phase="train", only_interacting=True):
    dir_path = "/cfs/cfs-oom9s50f/users/blairzheng/codes/InterHand/data/InterHand2.6M/annotations"

    dfi = pd.read_csv(os.path.join(dir_path, phase, "img.csv"), index_col=0)
    dfa = pd.read_csv(os.path.join(dir_path, phase, "ann.csv"), index_col=0)
    dfj = pd.read_csv(os.path.join(dir_path, phase, "joint.csv"), index_col=0)
    dfc = pd.read_csv(os.path.join(dir_path, phase, "cam.csv"), index_col=0)
    
    if only_interacting:
        eff_filter = dfa["hand_type"].isin(["interacting"]) & dfa["hand_type_valid"].isin([1])
        df = dfa[eff_filter].merge(dfi, on="image_id")
    else:
        df = dfa.merge(dfi, on="image_id")
        
    df = df.merge(dfj, on=["capture", "frame_idx"], how="left") \
           .merge(dfc, on=["capture", "camera"], how="left")
    
    return df


def calc_fuse_coord(world_coord, campos, camrot, focal, princpt):
    world_coord = np.array(json.loads(world_coord)).astype(float)
    campos, camrot = np.array(json.loads(campos)), np.array(json.loads(camrot))
    focal, princpt = np.array(json.loads(focal)), np.array(json.loads(princpt))

    cam_coord = world2cam(world_coord.transpose(1, 0), camrot, campos.reshape(3, 1))
    cam_coord = cam_coord.transpose(1, 0)
    img_coord = cam2pixel(cam_coord, focal, princpt)[:, :2]
    fuse_coord = np.concatenate((img_coord, cam_coord[:, 2].reshape(-1, 1)), 1)
    
    return fuse_coord, cam_coord


def eval_tx():
    image_dir = "/cfs/cfs-oom9s50f/users/yueya/data_raw/InterHand2.6M/InterHand2.6M_5fps_batch1/images/val"
    model_path = "/cfs/cfs-oom9s50f/users/blairzheng/codes/hand_3d_couple_pose/output/hand_pose_dataset/blaze_pose/ih05/70_checkpoint.pth.tar"
    data_path = "/cfs/cfs-oom9s50f/users/blairzheng/codes/hand_3d_couple_pose/datatool/annotations/train/hand_label.csv"
    output_dir = "/cfs/cfs-oom9s50f/users/blairzheng/codes/data/hp/09/"

    model = ghostnet(num_points=42 * 3)
    pose_net = model.cuda()
    model.eval()
    print("load model path: ", model_path)
    pose_net.load_state_dict(torch.load(model_path)['state_dict'], strict=True)

    # df = pd.read_csv(data_path, index_col=0)
    # dftrain = df.sample(n=30000, random_state=1)
    # dftest = df.drop(index=dftrain.index)
    
    dftest = load_dataset("val")
    roi_columns = ["file_name", "bbox", "world_coord", "campos", "camrot", "focal", "princpt", "joint_valid",
                   "hand_type", "hand_type_valid", "fuse_coord"]
    
    with torch.no_grad():
        for row in dftest.sample(n=10, random_state=50).itertuples():
            print(row.Index, row.file_name)
            # fuse_coord = np.array(json.loads(row.fuse_coord))
            # output_name = os.path.basename(row.file_name)
            fuse_coord0, cam_coord0 = calc_fuse_coord(row.world_coord, row.campos, row.camrot, row.focal, row.princpt)
            output_name = row.file_name.replace("/", "_")
            
            bbox = np.array(json.loads(row.bbox)).astype(int)
            
            img_path = row.file_name if os.path.exists(row.file_name) else os.path.join(image_dir, row.file_name)
             
            img = cv2.imread(img_path)
            img, fuse_coord, _ = crop_with_bbox(img, bbox, fuse_coord0, False)
            img, fuse_coord, _, _ = resize_img_keep_ratio(img, (192, 160), fuse_coord)
            ts_img = torch.from_numpy(img*1.0/255).permute(2, 0, 1).to(dtype=torch.float, device="cuda:0")
            output = model(ts_img.unsqueeze(0)).cpu().numpy()[0].reshape(-1, 3)
            
            canva1, canva2 = img.copy(), img.copy()
            canva1 = vis_sample(canva1, fuse_coord[:, :2], np.ones(fuse_coord.shape[0]), True)
            canva2 = vis_sample(canva2, output[:, :2], np.ones(output.shape[0]), False)

            skl0, geo0 = plot_3d_hand(cam_coord0, True)
            skl1, geo1 = plot_3d_hand(fuse_coord, True)
            skl2, geo2 = plot_3d_hand(output, False)
            
            img_merge = concat_images([img, canva1, skl1, canva2, skl2, skl0], is_vertical=False)
            cv2.imwrite(os.path.join(output_dir, output_name), img_merge)
            with open(os.path.join(output_dir, output_name.replace(".jpg", "_1.wrl")), "wb") as fid:
                fid.write(geo1)
            with open(os.path.join(output_dir, output_name.replace(".jpg", "_2.wrl")), "wb") as fid:
                fid.write(geo2)
            with open(os.path.join(output_dir, output_name.replace(".jpg", "_0.wrl")), "wb") as fid:
                fid.write(geo0)
            # print(np.array2string(output, formatter={"float_kind": lambda x: "%0.4f" % x}))
            # print(np.array2string(output, formatter={"float_kind": lambda x: "%0.4f" % x}))
            break
            
    return


def eval_imgs_tx():
    model_path = "/cfs/cfs-oom9s50f/users/blairzheng/codes/hand_3d_couple_pose/output/hand_pose_dataset/blaze_pose/ih07/50_checkpoint.pth.tar"
    input_dir = "/cfs/cfs-oom9s50f/users/blairzheng/codes/motion/hand_debug/00"
    output_dir = "/cfs/cfs-oom9s50f/users/blairzheng/codes/data/hp/10/00"

    model = ghostnet(num_points=42 * 3)
    pose_net = model.cuda()
    model.eval()
    print("load model path: ", model_path)
    pose_net.load_state_dict(torch.load(model_path)['state_dict'], strict=True)
    
    ls_path = glob.glob(os.path.join(input_dir, "*it*.jpg"))
    for path in ls_path:
        img_name = os.path.basename(path)
        img = cv2.imread(path)[:, :, ::-1]
        assert(img.shape[0] == 256 and img.shape[1] == 256)
        ts_img = torch.from_numpy(img * 1.0 / 255).permute(2, 0, 1).to(dtype=torch.float, device="cuda:0")
        with torch.no_grad():
            output = model(ts_img.unsqueeze(0)).cpu().numpy()[0].reshape(-1, 3)

        canva0, canva1 = img.copy(), img.copy()
        canva0 = vis_sample(canva0, output[:, :2], np.ones(output.shape[0]), False)
        
        df_intag = pd.read_csv(os.path.join(output_dir, img_name.replace(".jpg", "_intag.csv")), dtype=float)
        intag_coord = df_intag[["x", "y", "z"]].to_numpy()
        canva1 = vis_sample(canva1, intag_coord[:, :2], np.ones(output.shape[0]), False)

        img_merge = concat_images([img, canva0, canva1], is_vertical=False)
        output_name = os.path.basename(path)
        cv2.imwrite(os.path.join(output_dir, output_name), img_merge)
        
        df = pd.DataFrame(output, columns=["x", "y", "z"])
        df.to_csv(os.path.join(output_dir, output_name.replace(".jpg", ".csv")))
            
    coord_to_wrl_tx(output_dir)
    
    return


def dict_to_df_tx(phase="train"):
    # img ann
    db = COCO(f"InterHand2.6M_{phase}_data.json")

    ls_info = []
    ls_img_key2 = ['camera', 'capture', 'file_name', 'frame_idx', 'height', 'id', 'seq_name', 'subject', 'width']
    for key1 in db.imgs.keys():
        ls_val = [db.imgs[key1][key2] for key2 in ls_img_key2]
        ls_info.append(ls_val)
    dfi = pd.DataFrame(ls_info, columns=ls_img_key2)
    dfi = dfi.rename(columns={"id": "image_id"})
    dfi.to_csv("img.csv")

    ls_info = []
    ls_ann_key2 = ['id', 'image_id', 'bbox', 'joint_valid', 'hand_type', 'hand_type_valid']
    for key1 in db.anns.keys():
        ls_val = [db.anns[key1][key2] for key2 in ls_ann_key2]
        ls_info.append(ls_val)
    dfa = pd.DataFrame(ls_info, columns=ls_ann_key2)
    dfa.to_csv("ann.csv")

    # joint
    with open(f"InterHand2.6M_{phase}_joint_3d.json") as f:
        joints = json.load(f)

    ls_info = []
    for k1, v1 in joints.items():
        for k2, v2 in v1.items():
            info = (int(k1), int(k2), v2["world_coord"], v2["joint_valid"], v2["hand_type"], v2["hand_type_valid"])
            ls_info.append(info)
    columns = ["capture", "frame_idx", "world_coord", "d3_joint_valid", "d3_hand_type", "d3_hand_type_valid"]
    dfj = pd.DataFrame(ls_info, columns=columns)
    dfj.to_csv("joint.csv")

    # camera
    with open(f"InterHand2.6M_{phase}_camera.json") as f:
        cam = json.load(f)

    ls_info = []
    for k1, v1 in cam.items():
        campos_keys, camrot_keys = list(v1["campos"].keys()), list(v1["camrot"].keys())
        focal_keys, princpt_keys = list(v1["focal"].keys()), list(v1["princpt"].keys())
        keys = list(set(campos_keys + camrot_keys + focal_keys + princpt_keys))
        for k2 in keys:
            campos, camrot = v1["campos"].get(k2, ""), v1["camrot"].get(k2, "")
            focal, princpt = v1["focal"].get(k2, ""), v1["princpt"].get(k2, "")
            ls_info.append((k1, k2, campos, camrot, focal, princpt))

    dfc = pd.DataFrame(ls_info, columns=["capture", "camera", "campos", "camrot", "focal", "princpt"])
    dfc.to_csv("cam.csv")

    return


def concat_images(ls_image, is_vertical=True, gap=20):
    max_height, max_width = 0, 0
    sum_height, sum_width = 0, 0
    img_count = len(ls_image)

    for ii in range(img_count):
        max_height = max(ls_image[ii].shape[0], max_height)
        max_width = max(ls_image[ii].shape[1], max_width)
        sum_height += ls_image[ii].shape[0]
        sum_width += ls_image[ii].shape[1]
    sum_height += img_count * gap
    sum_width += img_count * gap

    if is_vertical:
        canva = np.ones([sum_height, max_width, 3], dtype=np.float) * 200
        row_st = 0
        for ii in range(img_count):
            img, shape = ls_image[ii], ls_image[ii].shape
            canva[row_st:row_st + shape[0], :shape[1], :] = img
            row_st += (shape[0] + gap)
    else:
        canva = np.ones([max_height, sum_width, 3], dtype=np.float) * 200
        col_st = 0
        for ii in range(img_count):
            img, shape = ls_image[ii], ls_image[ii].shape
            canva[:shape[0], col_st:col_st + shape[1], :] = img
            col_st += (shape[1] + gap)

    return canva


def visualize_samples_tx():
    phase = "train"
    image_dir = '/cfs/cfs-oom9s50f/users/yueya/data_raw/InterHand2.6M/InterHand2.6M_5fps_batch1/images'
    dir_path = "/cfs/cfs-oom9s50f/users/blairzheng/codes/InterHand/data/InterHand2.6M/annotations"

    dfi = pd.read_csv(os.path.join(dir_path, phase, "img.csv"), index_col=0)
    dfa = pd.read_csv(os.path.join(dir_path, phase, "ann.csv"), index_col=0)
    dfj = pd.read_csv(os.path.join(dir_path, phase, "joint.csv"), index_col=0)
    dfc = pd.read_csv(os.path.join(dir_path, phase, "cam.csv"), index_col=0)
    dfj = dfj.rename(columns={"3d_hand_type_valid": "d3_hand_type_valid", "3d_hand_type": "d3_hand_type"})

    capture, frame_idx = 7, 47461

    dfroi = dfi[(dfi["capture"] == capture) & (dfi["frame_idx"] == frame_idx)]
    dfrst = dfroi.merge(dfa, on="image_id", how="left") \
        .merge(dfj, on=["capture", "frame_idx"], how="left") \
        .merge(dfc, on=["capture", "camera"], how="left")

    joint_num = 21  # single hand
    root_joint_idx = {'right': 20, 'left': 41}
    joints_idx = {'right': np.arange(0, joint_num), 'left': np.arange(joint_num, joint_num * 2)}

    cm = plt.get_cmap('gist_rainbow')
    ls_img = []
    for row in dfrst.itertuples():
        img = cv2.imread(os.path.join(image_dir, phase, row.file_name))
        canva = img.copy()

        bbox = np.array(json.loads(row.bbox)).astype(int)
        cv2.rectangle(canva, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (200, 0, 200), thickness=2)

        world_coord = np.array(json.loads(row.world_coord)).astype(float)
        campos, camrot = np.array(json.loads(row.campos)), np.array(json.loads(row.camrot))
        focal, princpt = np.array(json.loads(row.focal)), np.array(json.loads(row.princpt))

        cam_coord = world2cam(world_coord.transpose(1, 0), camrot, campos.reshape(3, 1))
        cam_coord = cam_coord.transpose(1, 0)
        img_coord = cam2pixel(cam_coord, focal, princpt)[:, :2]

        for ii in range(img_coord.shape[0]):
            radius = 1 if ii not in [20, 41] else 2
            color = (np.array(cm(ii / (2 * joint_num))[:3]) * 255).tolist()
            curr_pt, parent_pt = img_coord[ii].astype(int), img_coord[g_joint_parent_inv[ii]].astype(int)
            cv2.circle(canva, curr_pt, radius, color, thickness=2)
            cv2.line(canva, curr_pt, parent_pt, color)
        text = "%s" % (row.camera)
        cv2.putText(img, text, (30, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 200, 0))

        img_merge = np.hstack([img, canva])
        ls_img.append(img_merge)

    ls_col_img = []
    for ii in range(int(len(dfrst) / 12 + 1)):
        ls_row_img = []
        for jj in range(12):
            idx = ii * 12 + jj
            if idx >= len(dfrst):
                continue
            ls_row_img.append(ls_img[idx])
        img_row = concat_images(ls_row_img, is_vertical=False)
        ls_col_img.append(img_row)
    img_all = concat_images(ls_col_img, is_vertical=True)

    d3_hand_type, d3_hand_type_valid = row.d3_hand_type, row.d3_hand_type_valid
    cv2.imwrite(f"{capture}_{frame_idx}_{d3_hand_type}_{d3_hand_type_valid}.png", img_all)

    return


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.set_zlabel("$Z$")

    return


def fig2arr(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = data.reshape((int(h), int(w), -1))
    return img


def plot_3d_hand(hand_coord, inv_joint_order=False, inv_axes=True):
    from mayavi import mlab
    from mayavi.mlab import points3d, plot3d
    
    joint_parent = g_joint_parent_inv if inv_joint_order else g_joint_parent
    f = mlab.figure(bgcolor=(0.1, 0.1, 0.1))
    mlab.clf()

    if inv_axes:
        hand_coord = hand_coord * np.array([[1, -1, -1]])
        
    cm = plt.get_cmap('gist_rainbow')
    for ii in range(hand_coord.shape[0]):
        color = tuple((np.array(cm(ii / 42)[:3])))
        cpt, ppt = hand_coord[ii], hand_coord[joint_parent[ii]]
        mlab.plot3d((ppt[0], cpt[0]), (ppt[1], cpt[1]), (ppt[2], cpt[2]), tube_radius=1, color=color)

    # ls_seg = [(7, 11), (11, 15), (15, 19)]
    # for st, et in ls_seg:
    #     cpt, ppt = hand_coord[st], hand_coord[et]
    #     mlab.plot3d((ppt[0], cpt[0]), (ppt[1], cpt[1]), (ppt[2], cpt[2]), tube_radius=1, color=color)
 
    temp_img_name, temp_wrl_name = "tmp.png", "temp.wrl"
    mlab.gcf().scene._lift()
    
    mlab.view(azimuth=45, elevation=45)
    img0 = mlab.screenshot()
    
    mlab.view(azimuth=80, elevation=20)
    img1 = mlab.screenshot()

    mlab.view(azimuth=20, elevation=80)
    img2 = mlab.screenshot()

    img = np.hstack([img0, img1, img2])[:, :, ::-1]
    
    xmin, ymin, zmin = hand_coord.min(axis=0)
    xmax, ymax, zmax = hand_coord.max(axis=0)
    mlab.outline(extent=[xmin, xmax, ymin, ymax, zmin, zmax])
    print(xmax-xmin, ymax-ymin, zmax-zmin)
    
    mlab.savefig(temp_wrl_name)
    with open(temp_wrl_name, "rb") as fid:
        geo = fid.read()

    return img, geo
    

def plot_3d_hand_tx():
    from mayavi import mlab
    from mayavi.mlab import points3d, plot3d

    # dfj = pd.read_csv("joint.csv", index_col=0)
    # dfi = pd.read_csv("img.csv", index_col=0)
    # dfa = pd.read_csv("ann.csv", index_col=0)
    # dfc = pd.read_csv("cam.csv", index_col=0)
    # df = dfa.merge(dfi, on="image_id").merge(dfj, on=["capture", "frame_idx"], how="left")\
    #                                   .merge(dfc, on=["capture", "camera"], how="left")
    # 
    # wc = np.array(json.loads(df["world_coord"][0]))[:21]
    
    wc = [[062.01,-42.57,965.84], [042.26,-50.33,986.74], [024.70,-62.06,1014.43], [005.97,-40.48,1034.21], [011.03,-75.90,887.16], [006.61,-74.65,911.91], [000.52,-72.51,937.23], [-14.89,-67.79,976.16], [015.26,-50.81,998.12], [025.73,-51.17,985.19], [022.56,-51.03,953.94], [-22.90,-44.73,966.89], [016.95,-35.15,1011.38], [026.24,-34.18,999.84], [034.00,-29.93,972.84], [-12.34,-21.27,970.09], [016.38,-17.83,1010.04], [029.87,-18.19,1001.60], [035.36,-11.09,980.53], [001.12,-00.46,978.65], [-27.27,-20.73,1052.98]]
    wc = np.array(wc)
    
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(*wc.T, s=6, alpha=0.9)

    mlab.figure(1, size=(800, 600))
    mlab.clf()
     
    cm = plt.get_cmap('gist_rainbow')
    for ii in range(wc.shape[0]):
        radius = 1 if ii not in [20, 41] else 2
        color = tuple((np.array(cm(ii / 42)[:3])))
        cpt, ppt = wc[ii], wc[g_joint_parent_inv[ii]]
        # ax.plot((ppt[0], cpt[0]), (ppt[1], cpt[1]), (ppt[2], cpt[2]), color=color, linewidth=3)
        # points3d(cpt[0], cpt[1], cpt[2], s=2, colormap="copper", scale_factor=1)
        # print(ppt, cpt)
        a = mlab.plot3d((ppt[0], cpt[0]), (ppt[1], cpt[1]), (ppt[2], cpt[2]), tube_radius=1, color=color)
        # continue
        
    ls_seg = [(7, 11), (11, 15), (15, 19)]
    for st, et in ls_seg:
        cpt, ppt = wc[st], wc[et]
        # ax.plot((ppt[0], cpt[0]), (ppt[1], cpt[1]), (ppt[2], cpt[2]), color=color, linewidth=3)
        # mlab.plot3d((ppt[0], cpt[0]), (ppt[1], cpt[1]), (ppt[2], cpt[2]), (radius, radius))

    mlab.draw()
    print("haha")
    # mlab.view(azimuth=45, elevation=54)
    mlab.savefig('a.wrl')
    mlab.savefig('a.png')

    # set_axes_equal(ax)
    # fig.tight_layout()
    # 
    # img0 = fig2arr(fig)
    # 
    # ax.view_init(0, 0)
    # img1 = fig2arr(fig)
    # 
    # ax.view_init(90, 0)
    # img2 = fig2arr(fig)
    # 
    # ax.view_init(0, 90)
    # img3 = fig2arr(fig)
    # 
    # img_all = np.hstack([img0, img1, img2, img3])
    # 
    # cv2.imwrite("3d_hand.png", img_all[:, :, ::-1])
    
    return


def calc_tran_error_tx():
    from scipy.spatial import distance
    
    dftest = load_dataset("val")
    print(dftest.columns)
    
    ls_info = []
    with torch.no_grad():
        for row in dftest[:10000].itertuples():
            # print(row.Index, row.file_name)
            if np.array(json.loads(row.joint_valid)).mean() + 0.00001 < 1:
                continue
            
            fuse_coord = calc_fuse_coord(row.world_coord, row.campos, row.camrot, row.focal, row.princpt)

            bbox = np.array(json.loads(row.bbox)).astype(int)
            img = np.zeros((2048, 2048, 3))
            img, fuse_coord, _ = crop_with_bbox(img, bbox, fuse_coord, False)
            img, fuse_coord, _, _ = resize_img_keep_ratio(img, (192, 160), fuse_coord)

            world_coord = np.array(json.loads(row.world_coord)).astype(float)
            world_dist = distance.cdist(world_coord, fuse_coord, 'euclidean')
            fuse_dist = distance.cdist(fuse_coord, fuse_coord, 'euclidean')
            
            wr0 = world_dist[20, 0]/world_dist[20, 3]
            wr1 = world_dist[20, 4] / world_dist[20, 7]
            wr2 = world_dist[20, 8] / world_dist[20, 11]
            wr3 = world_dist[20, 12] / world_dist[20, 15]
            wr4 = world_dist[20, 16] / world_dist[20, 19]
            wrl = world_dist[20, 41] / world_dist[20, 3]

            # wl0 = world_dist[41, 21]/world_dist[41, 24]
            # wl1 = world_dist[41, 25] / world_dist[41, 28]
            # wl2 = world_dist[41, 29] / world_dist[41, 32]
            # wl3 = world_dist[41, 33] / world_dist[41, 36]
            # wl4 = world_dist[41, 37] / world_dist[41, 40]
            
            fr0 = fuse_dist[20, 0] / fuse_dist[20, 3]
            fr1 = fuse_dist[20, 4] / fuse_dist[20, 7]
            fr2 = fuse_dist[20, 8] / fuse_dist[20, 11]
            fr3 = fuse_dist[20, 12] / fuse_dist[20, 15]
            fr4 = fuse_dist[20, 16] / fuse_dist[20, 19]
            frl = fuse_dist[20, 41] / fuse_dist[20, 3]

            ls_info.append((row.file_name, wr0, wr1, wr2, wr3, wr4, wrl, fr0, fr1, fr2, fr3, fr4, frl))
             
    column_names = ["file_name", "wr0", "wr1", "wr2", "wr3", "wr4", "wrl", "fr0", "fr1", "fr2", "fr3", "fr4", "frl"]
    pd.DataFrame(ls_info, columns=column_names).to_csv("dist.csv")
    
    return


def IK_tx():
    # # root joint
    # t0 = np.array([0, 0, 0])
    # 
    # # first joint
    # t1 = t0 + np.array([200, 0, 0])
    # 
    # # second joint
    # t2 = t1 + np.array([0, 100, 0])
    # 
    # # rotation with angle1
    # rt1 = cv2.Rodrigues(np.array([0, np.pi/2, 0]))[0]
    # 
    # # rotation with angle2
    # rt2 = cv2.Rodrigues(np.array([0, 0, np.pi/2]))[0]
    
    # root joint
    t0 = np.array([0.0077,  0.0021,  0.0309])

    # first joint
    t1 = np.array([-0.0288,  0.0060,  0.0338])

    # second joint
    t2 = np.array([-0.0536,  0.0041,  0.0333])

    # rotation with angle0
    rt0 = cv2.Rodrigues(np.array([00.0142, 00.1665, 00.3722]))[0]

    # rotation with angle1
    rt1 = cv2.Rodrigues(np.array([-0.1792, -0.0100, -0.5287]))[0]
    
    # t1 is rotated with angle0
    t11 = rt0.dot(t1 - t0) + t0
    
    # t2 is rotated with angle0
    t21 = rt0.dot(t2 - t0) + t0
    
    # t2 is rotated with angle1
    t22 = rt1.dot(t21 - t11) + t11
    
    # rt0*rt1
    rt0rt1 = np.matmul(rt0, rt1)
    
    # rt1*rt0
    rt2rt1 = np.matmul(rt1, rt0)
    
    t2_rt0rt1 = rt0rt1.dot(t2 - t1) + t11
   
    t2_rt1rt0 = rt2rt1.dot(t2 - t1) + t11

    print("origin", t22)
    print("rt0rt1", t2_rt0rt1)
    print("rt1rt0", t2_rt1rt0)

    from mayavi import mlab
    from mayavi.mlab import points3d, plot3d

    f = mlab.figure(bgcolor=(0.1, 0.1, 0.1))
    mlab.clf()
    
    cm = plt.get_cmap('gist_rainbow')
    
    print(t0.round(), t1.round(), t2.round())
    print(t0.round(), t11.round(), t21.round())
    print(t0.round(), t11.round(), t22.round())
    
    color = tuple((np.array(cm(1 / 42)[:3])))
    mlab.plot3d((t0[0], t1[0]), (t0[1], t1[1]), (t0[2], t1[2]), tube_radius=1, color=color)
    color = tuple((np.array(cm(3 / 42)[:3])))
    mlab.plot3d((t1[0], t2[0]), (t1[1], t2[1]), (t1[2], t2[2]), tube_radius=1, color=color)

    color = tuple((np.array(cm(10 / 42)[:3])))
    mlab.plot3d((t0[0], t11[0]), (t0[1], t11[1]), (t0[2], t11[2]), tube_radius=2, color=color)
    color = tuple((np.array(cm(12 / 42)[:3])))
    mlab.plot3d((t11[0], t21[0]), (t11[1], t21[1]), (t11[2], t21[2]), tube_radius=2, color=color)
 
    color = tuple((np.array(cm(20 / 42)[:3])))
    mlab.plot3d((t11[0], t22[0]), (t11[1], t22[1]), (t11[2], t22[2]), tube_radius=4, color=color)

    color = tuple((np.array(cm(30 / 42)[:3])))
    mlab.plot3d((t0[0], t2_rt0rt1[0]), (t0[1], t2_rt0rt1[1]), (t0[2], t2_rt0rt1[2]), tube_radius=6, color=color)
 
    mlab.savefig("b.wrl")

    return


def onnx_infer_tx():
    import onnxruntime as nxrun
    import numpy as np
    
    ximg = np.random.rand(2, 3, 144, 96).astype(np.float32)
    sess1 = nxrun.InferenceSession("hand_3d_144x96_1018.onnx")
    sess2 = nxrun.InferenceSession("hand_3d_144x96_bs1_1018.onnx")
    result1 = sess1.run(None, {"input": ximg})
    result2 = sess2.run(None, {"input": ximg[:1]})
    
    return


def coord_to_wrl(ls_path, inv_yz_axes=False):
    from mayavi import mlab
    from mayavi.mlab import points3d, plot3d
    
    joint_parent = g_joint_parent
    
    for path in ls_path:
        print(path)
        df = pd.read_csv(path, index_col=0, dtype=float)
        if inv_yz_axes:
            df["y"], df["z"] = -1*df["y"], -1*df["z"]
        
        f = mlab.figure(bgcolor=(0.1, 0.1, 0.1))
        mlab.clf()

        cm = plt.get_cmap('gist_rainbow')
        for ii in range(len(df)):
            cpt = np.array(df.loc[ii, ["x", "y", "z"]])
            ppt = np.array(df.loc[joint_parent[ii], ["x", "y", "z"]])
            color = tuple((np.array(cm(ii / 42)[:3])))
            mlab.plot3d((ppt[0], cpt[0]), (ppt[1], cpt[1]), (ppt[2], cpt[2]), tube_radius=1, color=color)

        mlab.savefig(path.replace(".csv", ".wrl"))
         
    return


def coord_to_wrl_tx(src_dir=None, inv_yz_axes=False):
    if src_dir is None:
        # src_dir = "/cfs/cfs-oom9s50f/users/blairzheng/codes/motion/hand_debug/02"
        src_dir = "/cfs/cfs-oom9s50f/users/blairzheng/codes/hand_3d_couple_pose/IntagHand/test"
        
    ls_path = glob.glob(os.path.join(src_dir, "*.csv"))
    print(len(ls_path))
    
    parallel_count = 10
    step = int(len(ls_path) / parallel_count) + parallel_count
    
    ls_process = []
    for gpu_id, st in enumerate(range(0, len(ls_path), step)):
        ls_sub_path = ls_path[st: st+step]
        process = multiprocessing.Process(target=coord_to_wrl, args=(ls_sub_path, inv_yz_axes))
        process.start()
        ls_process.append(process)

    for process in ls_process:
        process.join()
        
    return


if __name__ == "__main__":
    # vis_label_sample_tx()
    # camera_visualize_tx()
    # visualize_all_camera_tx()
    # dict_to_df_tx()
    # eval_imgs_tx()
    # plot_3d_hand_tx()
    # calc_tran_error_tx()
    IK_tx()
    # coord_to_wrl_tx()
    # vis_sample_tx()