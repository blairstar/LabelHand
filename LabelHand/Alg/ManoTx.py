
from typing import NewType, Optional, Dict, Union
import os
import os.path as osp
import pickle
import numpy as np
from collections import namedtuple
import json
from LabelHand.Alg.MiscFun import world2cam, cam2pixel, pixel2cam, load_dataset, vis_sample
import time
# os.environ['OPEN3D_CPU_RENDERING'] = 'true'
import open3d as o3d
from open3d import utility
import open3d.visualization.rendering as rendering

import torch
import torch.nn as nn
from torch import Tensor
from LabelHand.Alg.lbs import lbs
import cv2

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

g_render = rendering.OffscreenRenderer(512, 512)
g_render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA

MANOOutput = namedtuple("MANOOutput",
                        ["vertices", "joints", "betas", "global_orient", "hand_pose", "full_pose", "faces"])



def o3d_float(array):
    return utility.Vector3dVector(array)

def o3d_int(array):
    return utility.Vector3iVector(array)

def local_time():
    return time.strftime("%H:%M:%S", time.localtime())


def get_cube_from_bound(limits):
    """get the vertices, edges, and faces of a cuboid defined by its limits

    limits = np.array([[x_min, x_max],
                       [y_min, y_max],
                       [z_min, z_max]])
    """
    v = np.array([[0, 0, 0], [0, 0, 1],
                  [0, 1, 0], [0, 1, 1],
                  [1, 0, 0], [1, 0, 1],
                  [1, 1, 0], [1, 1, 1]], dtype=int)

    v = limits[np.arange(3)[np.newaxis, :].repeat(8, axis=0), v]

    e = np.array([[0, 1], [0, 2], [0, 4],
                  [1, 3], [1, 5],
                  [2, 3], [2, 6],
                  [3, 7],
                  [4, 5], [4, 6],
                  [5, 7],
                  [6, 7]], dtype=int)

    f = np.array([[0, 2, 3, 1],
                  [0, 4, 5, 1],
                  [0, 4, 6, 2],
                  [1, 5, 7, 3],
                  [2, 6, 7, 3],
                  [4, 6, 7, 5]], dtype=int)

    return v, e, f


class MANO(nn.Module):
    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = 1
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS

    def __init__(
        self,
        model_path: str,
        is_rhand: bool = True,
        use_pca: bool = True,
        num_pca_comps: int = 6,
        flat_hand_mean: bool = False,
        batch_size: int = 1,
        dtype=torch.float32):
        ''' MANO model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the right
                hand. (default = True)
            hand_pose: torch.tensor, optional, BxP
                The default value for the right hand pose member variable.
                (default = None)
            num_pca_comps: int, optional
                The number of PCA components to use for each hand.
                (default = 6)
            flat_hand_mean: bool, optional
                If False, then the pose of the hand is initialized to False.
            batch_size: int, optional
                The batch size used for creating the member variables
            dtype: torch.dtype, optional
                The data type for the created variables
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''
        super(MANO, self).__init__()
        
        self.num_pca_comps = num_pca_comps
        self.is_rhand = is_rhand
        self.batch_size = batch_size
        self.dtype = dtype
        self.m2t_order = [0,
                          13, 14, 15, 16,
                          1,  2,  3,  17,
                          4,  5,  6,  18,
                          10, 11, 12, 19,
                          7,  8,  9,  20]
        
        # Load the model
        if osp.isdir(model_path):
            model_fn = 'MANO_{}.pkl'.format('RIGHT' if is_rhand else 'LEFT')
            mano_path = os.path.join(model_path, model_fn)
        
        with open(mano_path, 'rb') as mano_file:
            model_data = pickle.load(mano_file, encoding='latin1')
        
        # shape basis
        shapedirs = torch.from_numpy(model_data["shapedirs"]).to(dtype=dtype)
        self.register_buffer("shapedirs", shapedirs)
        self.num_betas = shapedirs.shape[-1]
        
        # The vertices of the template model
        v_template = torch.from_numpy(model_data["v_template"]).to(dtype=dtype)
        self.register_buffer('v_template', v_template)
        
        # J regressor
        J_regressor = torch.from_numpy(model_data["J_regressor"].toarray()).to(dtype=dtype)
        self.register_buffer('J_regressor', J_regressor)
        
        # pose basis
        posedirs = np.reshape(model_data["posedirs"], [-1, model_data["posedirs"].shape[2]])
        posedirs = torch.from_numpy(posedirs.T).to(dtype=dtype)
        self.register_buffer('posedirs', posedirs)

        # indices of parents for each joints
        parents = torch.from_numpy(model_data["kintree_table"][0]).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        
        # lbs weight
        self.register_buffer('lbs_weights', torch.from_numpy(model_data["weights"]).to(dtype=dtype))
        
        # face
        self.register_buffer('faces', torch.from_numpy(model_data["f"].astype(int)).to(dtype=dtype))

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        if self.num_pca_comps == 45:
            self.use_pca = False
        self.flat_hand_mean = flat_hand_mean

        hands_components = model_data["hands_components"][:num_pca_comps]

        self.np_hand_components = hands_components

        if self.use_pca:
            self.register_buffer('hands_components', torch.from_numpy(hands_components).to(dtype=dtype))

        if self.flat_hand_mean:
            hand_mean = np.zeros_like(model_data["hands_mean"])
        else:
            hand_mean = model_data["hands_mean"]

        self.register_buffer('hand_mean', torch.from_numpy(hand_mean).to(dtype=dtype))

        # global orientation
        default_global_orient = torch.zeros([batch_size, 3], dtype=dtype)
        global_orient = nn.Parameter(default_global_orient, requires_grad=True)
        self.register_parameter('global_orient', global_orient)

        # beta parameters
        default_betas = torch.zeros([batch_size, self.num_betas], dtype=dtype)
        self.register_parameter('betas', nn.Parameter(default_betas, requires_grad=True))
        
        # pose parameters
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS
        default_hand_pose = torch.zeros([batch_size, hand_pose_dim], dtype=dtype)
        hand_pose_param = nn.Parameter(default_hand_pose, requires_grad=True)
        self.register_parameter('hand_pose', hand_pose_param)
        
        # Create the buffer for the mean pose.
        pose_mean = self.create_mean_pose()
        pose_mean_tensor = pose_mean.clone().to(dtype)
        # pose_mean_tensor = torch.tensor(pose_mean, dtype=dtype)
        self.register_buffer('pose_mean', pose_mean_tensor)
        
        return

    def create_mean_pose(self):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        pose_mean = torch.cat([global_orient_mean, self.hand_mean], dim=0)
        return pose_mean

    def extra_repr(self):
        msg = [super(MANO, self).extra_repr()]
        if self.use_pca:
            msg.append(f"Number of PCA components: {self.num_pca_comps}")
        msg.append(f"Flat hand mean: {self.flat_hand_mean}")
        return '\n'.join(msg)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False):
        
        ''' Forward pass for the MANO model
        '''
        
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else self.global_orient)
        betas = betas if betas is not None else self.betas
        hand_pose = (hand_pose if hand_pose is not None else self.hand_pose)

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl
        
        is_axis_angle = (hand_pose.shape[-2:] != torch.Size([3, 3]))
        if self.use_pca and is_axis_angle is True:
            hand_pose = torch.einsum('bi,ij->bj', [hand_pose, self.hand_components])
         
        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        if is_axis_angle is True:
            full_pose += self.pose_mean

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=is_axis_angle)
        
        joints = torch.cat([joints, vertices[:, [745, 333, 444, 556, 673]]], dim=1)
        # joints = torch.cat([joints, vertices[:, [745, 317, 444, 556, 673]]], dim=1)
        # joints = torch.cat([joints, vertices[:, [744, 320, 443, 554, 671]]], dim=1)
        joints = joints[:, self.m2t_order]
        
        if apply_trans:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)
        
        output = MANOOutput(vertices=vertices if return_verts else None,
                            joints=joints if return_verts else None,
                            betas=betas,
                            global_orient=global_orient,
                            hand_pose=hand_pose,
                            faces=self.faces,
                            full_pose=full_pose if return_full_pose else None)

        return output
    
    @staticmethod
    def to_obj(vertices, faces, output_path):
        with open(output_path, 'w') as fid:
            for v in vertices:
                fid.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
                fid.write('f %d %d %d\n' % (f[0], f[1], f[2]))
        return
    
    @staticmethod
    def rotation_matrix_from_vectors(vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
     
    def calc_recify_rotation(self, beta):
        # beta = torch.tensor([[-2.5099, 00.1018, -0.6341, 00.0132, 00.2387, -0.1313, 00.1155, 00.2773, -0.1189, -0.2002]])
        beta = torch.from_numpy(beta.astype(np.float32)).unsqueeze(0)
        hand_pose = torch.zeros([1, 15 * 3])
        root_pose = torch.tensor([[0.0, 0.0, 0.0]])
        transl = torch.tensor([[0.0, 0.0, 0.0]])
        output = self.forward(beta, root_pose, hand_pose, transl)
        ori_root_pos = output.joints[0][0].numpy()
        ori_finger_root_pos = output.joints[0][[2, 5, 9, 13, 17]].numpy()
        joints = output.joints * 1000
        vertices = output.vertices * 1000
        
        # ls_pair_idx = [(286, 287), (274, 261), (290, 509), (399, 388), (627, 616)]
        # ls_pair_idx = [(286, 287), (274, 261), (290, 509), (291, 509), (627, 616)]

        recify_rot = []
        for ii in range(5):
            idx = 1 + ii*4 + 1 if ii in [0, 3] else 1 + ii*4
            p0, p1 = joints[0][idx].numpy(), joints[0][idx+1].numpy()
            # st_idx, et_idx = ls_pair_idx[ii]
            # p0, p1 = vertices[0][st_idx].numpy(), vertices[0][et_idx].numpy()
            # p0[1], p1[1] = 0, 0
            v1 = p1 - p0
            dist = np.linalg.norm(v1)
            v2 = np.array([-dist, 0, 0]) if self.is_rhand else np.array([dist, 0, 0])
            axis = np.cross(v1, v2)
            axis = axis/np.linalg.norm(axis)
            angle = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            rot_mat = cv2.Rodrigues(np.arccos(angle)*axis)[0]
            # print(self.is_rhand, ii, v1/dist)
            recify_rot.append(rot_mat)
            
        return recify_rot, ori_root_pos, ori_finger_root_pos


i2t_order = [20,
             3,  2,  1,  0,
             7,  6,  5,  4,
             11, 10, 9,  8,
             15, 14, 13, 12,
             19, 18, 17, 16]


def render(img_shape, vertices, faces, focal, princpt, hand_type=True, render_type="open3d", with_3d=False):
    # print(vertices, "render")
    if render_type.lower() in ["0", "open3d"]:
        vertices, faces = vertices.numpy(), faces.numpy()
        focal, princpt = focal.numpy(), princpt.numpy()
        out = render_by_open3d(img_shape[::-1], vertices, faces, focal, princpt, hand_type, with_3d)
    elif render_type.lower() in ["1", "pytorch3d"]:
        vertices, faces = vertices.unsqueeze(0), faces.unsqueeze(0)
        focal, princpt = focal.unsqueeze(0), princpt.unsqueeze(0)
        out = render_by_pytorch3d(img_shape, vertices, faces, focal, princpt)
    else:
        raise Exception("unsupported render tool")
    
    return out


def render_by_open3d(img_size, vertices, faces, focal, princpt, hand_type, with_3d=False): 
    def get_render(img_size):
        global g_render
        curr_size = np.array(g_render.render_to_image()).shape[:2][::-1]
        if curr_size[0] != img_size[0] or curr_size[1] != img_size[1]:
            g_render = rendering.OffscreenRenderer(img_size[0], img_size[1])
            g_render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA
            print(local_time(), "ManoTx create new render", curr_size, img_size)
        return g_render
    
    def create_mesh(vertices, faces, is_rhand):
        colors = np.zeros([vertices.shape[0], 3])
        colors[:, :] = np.array([[0.8, 1.0, 0.0]]) if is_rhand else np.array([1.0, 0.75, 0.0])
        colors[723, :] = np.array([[0.2, 0.0, 1.0]])
        colors[724, :] = np.array([[0.2, 0.0, 1.0]])
        colors[727, :] = np.array([[0.2, 0.0, 1.0]])
        
        colors[733, :] = np.array([[0.8, 0.0, 0.0]])
        colors[734, :] = np.array([[0.8, 0.0, 0.0]])
        
        colors[750, :] = np.array([[0.0, 1.0, 0.0]])
        colors[765, :] = np.array([[0.0, 1.0, 0.0]])
        
        colors[311, :] = np.array([[0.2, 0.0, 1.0]])
        colors[314, :] = np.array([[0.2, 0.0, 1.0]])
        
        colors[423, :] = np.array([[0.2, 0.0, 1.0]])
        colors[426, :] = np.array([[0.2, 0.0, 1.0]])
        
        colors[534, :] = np.array([[0.2, 0.0, 1.0]])
        colors[537, :] = np.array([[0.2, 0.0, 1.0]])
        
        colors[651, :] = np.array([[0.2, 0.0, 1.0]])
        colors[654, :] = np.array([[0.2, 0.0, 1.0]])
        
        if is_rhand:
            colors[207, :] = np.array([[0.0, 0.0, 0.3]])
            colors[218, :] = np.array([[0.0, 0.0, 0.5]])
            colors[219, :] = np.array([[0.0, 0.0, 0.7]])
            colors[178, :] = np.array([[0.0, 0.0, 0.9]])
        else:
            colors[207, :] = np.array([[0.2, 0.0, 0.3]])
            colors[218, :] = np.array([[0.2, 0.0, 0.5]])
            colors[219, :] = np.array([[0.2, 0.0, 0.7]])
            colors[178, :] = np.array([[0.2, 0.0, 0.9]])

        mesh = o3d.geometry.TriangleMesh(vertices=o3d_float(vertices), triangles=o3d_int(faces))
        mesh.vertex_colors = o3d_float(colors)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        return mesh
    
    # yellow = rendering.MaterialRecord()
    # yellow.base_color = [1.0, 0.75, 0.0, 1.0]
    # yellow.shader = "defaultLit"
    # 
    # green = rendering.MaterialRecord()
    # green.base_color = [0.8, 1.0, 0.0, 1.0]
    # green.shader = "defaultLit"
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    
    g_render = get_render(img_size)
    # g_render = rendering.OffscreenRenderer(img_size[0], img_size[1])
    # g_render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA
    
    geometries = []
    
    if hand_type == 1:
        mesh = create_mesh(vertices, faces, True)
        g_render.scene.add_geometry("right_hand", mesh, material)
        geometries.append({"name": "right", "geometry": mesh, "material": material})
    elif hand_type == 2:
        mesh = create_mesh(vertices, faces, False)
        g_render.scene.add_geometry("left_hand", mesh, material)
        geometries.append({"name": "left", "geometry": mesh, "material": material})
    elif hand_type == 3:
        right_vertices, left_vertices = vertices[:vertices.shape[0]//2], vertices[vertices.shape[0]//2:]
        right_faces, left_faces = faces[:faces.shape[0]//2], faces[faces.shape[0]//2:]
        left_faces = left_faces - right_vertices.shape[0]
        right_mesh = create_mesh(right_vertices, right_faces, True)
        left_mesh = create_mesh(left_vertices, left_faces, False)
        g_render.scene.add_geometry("right_hand", right_mesh, material)
        g_render.scene.add_geometry("left_hand", left_mesh, material)
        geometries.append({"name": "left", "geometry": left_mesh, "material": material})
        geometries.append({"name": "right", "geometry": right_mesh, "material": material})
    else:
        pass
    
    if with_3d:
        ct = vertices.mean(axis=0)
        ret = o3d.visualization.draw(geometries, title="LabelHand", bg_color=[0.0, 0.0, 0.0, 1.0],
                                     lookat=ct, eye=ct-np.array([0, 0, 300]), up=np.array([0, -1, 0]),
                                     raw_mode=False, show_skybox=False, non_blocking_and_return_uid=True)
        print("open3d draw ret", ret)
    
    param = o3d.camera.PinholeCameraIntrinsic(img_size[0], img_size[1], focal[0], focal[1], princpt[0], princpt[1])
    R = np.eye(4, dtype=np.float64)
    # R[2, 2] = -1
    g_render.setup_camera(param, R)
    # intrin = np.array([[focal[0], 0, princpt[0]], [0, focal[0], princpt[1]], [0, 0, 1]], dtype=np.float64)
    # render.scene.camera.set_projection(intrin, 1, 5000, img_size[0], img_size[1])
    # render.scene.camera.look_at([0, 0, 0], [0, 0, 100], [0, 1, 0])
    # render.scene.show_axes(True)
    # render.setup_camera(intrin, np.eye(4, dtype=np.float), img_size[0], img_size[1])
    # render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0], 75000)
    # render.scene.scene.enable_sun_light(True)
    
    img = g_render.render_to_image()
    # o3d.io.write_image("D:\\dev_data\\model_data\\fingerprint\\outputs\\vimages\\hp\\13\\l2.png", img, 9)
    
    g_render.scene.remove_geometry("left_hand")
    g_render.scene.remove_geometry("right_hand")

    img = np.asarray(img)
    mask = (img.mean(axis=-1, keepdims=True) > 30).astype(np.float32)
    return img, mask, geometries


def render_by_pytorch3d(img_size, vertices, faces, focal, princpt):
    import matplotlib.pyplot as plt
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import PerspectiveCameras, OrthographicCameras, PointLights, AmbientLights, TexturesVertex
    from pytorch3d.renderer import RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader
    cm = plt.get_cmap('gist_rainbow')
    
    device = "cpu"
    raster_settings = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1)
    renderer_rgb = MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings),
                                shader=HardPhongShader(device=device))

    amblights = AmbientLights(device="cpu")
    ls_color = []
    for ii in range(778*2):
        val = torch.from_numpy(np.array(cm(ii%50/50)[:3])*200)
        ls_color.append(val.to(device=device))
    color = torch.stack(ls_color).unsqueeze(0)
    
    texture = TexturesVertex(verts_features=color)
    R = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).repeat(1, 1, 1).to(torch.float32)
    camera = PerspectiveCameras(focal_length=focal.to(device, dtype=torch.float32),
                                principal_point=princpt.to(device, dtype=torch.float32),
                                R=R, in_ndc=False, device=device)
    
    mesh = Meshes(verts=vertices.to(device, dtype=torch.float32), faces=faces.to(device), textures=texture)
    output = renderer_rgb(mesh, cameras=camera, lights=amblights, image_size=img_size)
    alpha = output[..., 3].cpu().detach().numpy()
    img = output[..., :3].cpu().detach().numpy().astype(np.uint8)
     
    return img[0], alpha[..., None][0]


@torch.no_grad()
def mano_forward(mano_model, hand_param, focal, princpt):
    device = "cpu"
    xy, depth, root_pose, hand_pose, beta, ori_root_pos = hand_param
    beta = torch.from_numpy(beta[None, ...]).to(dtype=torch.float32, device=device)
    root_pose = torch.from_numpy(root_pose[None, ...]).to(dtype=torch.float32, device=device)
    hand_pose = torch.from_numpy(hand_pose[None, ...]).to(dtype=torch.float32, device=device)
    
    # is_rhand = mano_model.is_rhand
    # to do: ori_root_pos need to changed with different beta
    # ori_root_pos = torch.tensor([0.1077, 0.0074, 0.0068]) if is_rhand else torch.tensor([-0.0836, 0.0074, 0.0068])

    transl = torch.tensor([[0, 0, depth]], dtype=torch.float, device=device)
    transl[0, :2] = (torch.from_numpy(xy) - princpt) * depth / focal
    transl = transl / 1000 - ori_root_pos

    output = mano_model.forward(beta, root_pose, hand_pose, transl)
    vertices = output.vertices * 1000
    joints = output.joints * 1000
    
    # if not is_rhand:
    #     print(local_time(), root_pose)

    return vertices[0], joints[0]


def recify_finger_pose(finger_pose, recify_rot):
    thumb2_rot = Rotation.get_x_roll_mat(-75)

    mano_order = [1, 2, 4, 3, 0]
    for ii in range(5):
        kk = mano_order[ii]
        for jj in range(3):
            idx = ii * 3 + jj
            if ii == 4 and jj in [1, 2]:
                finger_pose[idx] = np.dot(thumb2_rot.transpose(), np.dot(finger_pose[idx], thumb2_rot)) 
            finger_pose[idx] = np.dot(recify_rot[kk].transpose(), np.dot(finger_pose[idx], recify_rot[kk]))
    return finger_pose


@torch.no_grad()
def render_two_hand(image, right_mano_model, left_mano_model, right_hand_param, left_hand_param,
                    global_param, right_draw_config, left_draw_config, with_3d=False):
    device = "cpu"
    
    left, top, right, bottom = global_param[2].tolist()
    img = np.ascontiguousarray(image[top:bottom, left:right].copy())
    
    focal = torch.from_numpy(global_param[0]).to(dtype=torch.float32, device=device)
    princpt = torch.from_numpy(global_param[1]).to(dtype=torch.float32, device=device)
     
    right_vertice, right_joint = mano_forward(right_mano_model, right_hand_param, focal, princpt)
    left_vertice, left_joint = mano_forward(left_mano_model, left_hand_param, focal, princpt)
     
    with_right_img, with_right_mesh, with_right_skl = right_draw_config
    with_left_img, with_left_mesh, with_left_skl = left_draw_config
        
    canva = np.zeros_like(img, dtype=np.float32)
    if with_right_img or with_left_img:
        canva = img.copy()
    
    geo = []
    left_face, right_face = left_mano_model.faces, right_mano_model.faces
    if with_right_mesh and not with_left_mesh:
        img_out, mask, geo = render(canva.shape[:2], right_vertice, right_face, focal, princpt, 1, "open3d", with_3d)
        img_out, mask = img_out, mask * 1.0
        canva = (canva * (1 - mask) + img_out * mask)
    
    if not with_right_mesh and with_left_mesh:
        img_out, mask, geo = render(canva.shape[:2], left_vertice, left_face, focal, princpt, 2, "open3d", with_3d)
        img_out, mask = img_out, mask * 1.0
        canva = (canva * (1 - mask) + img_out * mask)
    
    if with_right_mesh and with_left_mesh:
        vertice = torch.vstack([right_vertice, left_vertice])
        left_face = left_face + right_vertice.shape[0]
        face = torch.vstack([right_face, left_face])
        img_out, mask, geo = render(canva.shape[:2], vertice, face, focal, princpt, 3, "open3d", with_3d)
        img_out, mask = img_out, mask * 1.0
        canva = (canva * (1 - mask) + img_out * mask)
    
    canva = canva.astype(np.uint8)
    
    # 如果只画部分skeleton，可分块画图，各块节点由UI确定
    if with_right_skl > 0:
        fuse_coord = cam2pixel(right_joint.numpy(), focal.numpy(), princpt.numpy())
        img_coord = fuse_coord[:, :2]
        img_coord = img_coord[:1] if with_right_skl == 1 else img_coord
        canva = vis_sample(canva, img_coord, np.ones(img_coord.shape[0]), offset=0)

    if with_left_skl > 0:
        fuse_coord = cam2pixel(left_joint.numpy(), focal.numpy(), princpt.numpy())
        img_coord = fuse_coord[:, :2]
        img_coord = img_coord[:1] if with_left_skl == 1 else img_coord
        canva = vis_sample(canva, img_coord, np.ones(img_coord.shape[0]), offset=21)
    
    joints = np.vstack([right_joint.numpy(), left_joint.numpy()])
    return img, canva, joints, geo


@torch.no_grad()
def render_with_mano(canva, mano_model, beta, xy, depth, root_pose, hand_pose, with_mesh, with_skl):
    device = "cpu"
    focal = torch.tensor([1000, 1000], dtype=torch.float32, device=device)
    princpt = torch.tensor([200, 200], dtype=torch.float32, device=device)
    
    beta = torch.from_numpy(beta[None, ...]).to(dtype=torch.float32, device=device)
    root_pose = torch.from_numpy(root_pose[None, ...]).to(dtype=torch.float32, device=device)
    hand_pose = torch.from_numpy(hand_pose[None, ...]).to(dtype=torch.float32, device=device)
    
    is_rhand = mano_model.is_rhand
    ori_root_pos = torch.tensor([0.1077, 0.0074, 0.0068]) if is_rhand else torch.tensor([-0.0836, 0.0074, 0.0068])
    
    transl = torch.tensor([[0, 0, depth]], dtype=torch.float, device=device)
    transl[0, :2] = (torch.from_numpy(xy) - princpt) * depth / focal
    transl = transl/1000 - ori_root_pos
    
    output = mano_model.forward(beta, root_pose, hand_pose, transl)
    vertices = output.vertices * 1000
    joints = output.joints * 1000
    
    if with_mesh:
        img_out, mask = render(canva.shape[:2], vertices[0], output.faces, focal, princpt, is_rhand)
        img_out, mask = img_out, mask * 0.9
        canva = (canva*(1 - mask) + img_out*mask)
        
    canva = canva.astype(np.uint8)

    fuse_coord = cam2pixel(joints[0].numpy(), focal.numpy(), princpt.numpy())
    img_coord = fuse_coord[:, :2]
    xy_ct = img_coord[0]
    xy_size = np.linalg.norm(img_coord[0] - img_coord[1])
    if with_skl:
        # 如果只画部分skeleton，可分块画图，各块节点由UI确定
        offset = 0 if is_rhand else 21
        canva = vis_sample(canva, img_coord, np.ones(img_coord.shape[0]), offset=offset)
    
    return canva, xy_ct.astype(int), xy_size.astype(int)


def mano_validate_tx():
    img_path = "/cfs/cfs-oom9s50f/users/yueya/data_raw/InterHand2.6M/InterHand2.6M_5fps_batch1/images/val"
    model_path = "/cfs/cfs-oom9s50f/users/blairzheng/codes/smplx/models/mano"
    ann_path = "/cfs/cfs-oom9s50f/users/blairzheng/codes/InterHand/data/InterHand2.6M/annotations/val/"

    data = load_dataset("val")
    roi_vals = data.loc[100000, ["file_name", "world_coord", "camrot", "campos", "focal", "princpt", "capture", "frame_idx"]]
    file_name, world_coord, camrot, campos, focal, princpt, capture, frame_idx = roi_vals
    
    ann = json.load(open(os.path.join(ann_path, "InterHand2.6M_val_MANO_NeuralAnnot.json")))
    param = ann[str(capture)][str(frame_idx)]["right"]
    betas = torch.FloatTensor(param['shape']).view(1, -1)
    pose = torch.FloatTensor(param['pose'])
    transl = torch.FloatTensor(param['trans']).view(1, -1)
    root_pose, hand_pose = pose[:3].view(1, -1), pose[3:].view(1, -1)
    # root_pose = torch.tensor([[00.0000, 00.0000, 00.0000]])
    root_pose = torch.tensor([[-1.3194, 00.2784, -2.7738]])
    hand_pose = torch.zeros([1, 15 * 3])
    transl = torch.tensor([[-0.1896, -0.1467, 01.0980]])
    hand_pose = torch.tensor([[00.0142, 00.1665, 00.3722, -0.1792, -0.0100, -0.5287, 00.1317, -0.2723, -0.2277,
                               -0.1399, 00.1626, 00.4211, -0.0460, 00.0666, 00.3086, -0.0468, -0.0487, 00.2785,
                               -0.3905, 00.3746, 00.2121, -0.1821, 00.2086, 00.3088, -0.4430, 00.1422, 00.4402,
                               -0.3289, 00.2303, 00.5968, -0.2206, 00.1250, 00.2318, -0.2407, -0.0142, 00.5726,
                               -0.0362, -0.3250, -0.2363, -0.0691, 00.0652, 00.3312, 00.1396, -0.2450, -0.2181]])
    
    print(len(data))
    print(file_name)
    print(np.array2string(betas.numpy(), formatter={'float_kind': lambda x: "%07.4f" % x}, separator=",").replace('\n', ''))
    print(np.array2string(hand_pose.numpy(), formatter={'float_kind': lambda x: "%07.4f" % x}, separator=",").replace('\n', ''))
    print(np.array2string(root_pose.numpy(), formatter={'float_kind': lambda x: "%07.4f" % x}, separator=",").replace('\n', ''))
    print(np.array2string(transl.numpy(), formatter={'float_kind': lambda x: "%07.4f" % x}, separator=",").replace('\n', ''))
    
    model = MANO(model_path, True, False)
    
    output = model.forward(betas, root_pose, hand_pose, transl)
    
    world_coord = np.array(json.loads(world_coord)).astype(float)
    campos, camrot = np.array(json.loads(campos)), np.array(json.loads(camrot))
    focal, princpt = np.array(json.loads(focal)), np.array(json.loads(princpt))
    world_coord = world_coord[i2t_order]
    
    cam_coord = world2cam(world_coord.transpose(1, 0), camrot, campos.reshape(3, 1))
    cam_coord = cam_coord.transpose(1, 0)
    img_coord = cam2pixel(cam_coord, focal, princpt)[:, :2]
    
    mano_joints_in = output.joints.detach().cpu().numpy()[0]*1000
    
    joint_regressor = np.load('J_regressor_mano_ih26m.npy')
    mano_joints_wc = np.dot(joint_regressor, output.vertices.detach().cpu().numpy()[0]*1000)[i2t_order]

    mano_joints_cam = world2cam(mano_joints_wc.transpose(1, 0), camrot, campos.reshape(3, 1))
    mano_joints_cam = mano_joints_cam.transpose(1, 0)
    
    print(np.sqrt(((world_coord-mano_joints_in)**2).sum(axis=1)).mean())
    print(np.sqrt(((world_coord-mano_joints_wc)**2).sum(axis=1)).mean())
    
    img = cv2.imread(os.path.join(img_path, file_name))
    
    vertices = torch.matmul(torch.from_numpy(camrot[None, ...]),
                            (output.vertices*1000-torch.from_numpy(campos)).permute(0, 2, 1))
    vertices = vertices.permute(0, 2, 1)

    img_out, mask = render(img.shape[:2], vertices, output.faces.repeat(1, 1, 1),
                           torch.from_numpy(focal).reshape(1, 2), torch.from_numpy(princpt).reshape(1, 2))

    img_out, mask = img_out[0], mask[0] * 0.3
    img_mask = (img_out * mask + img * (1 - mask)).astype(np.uint8)
    
    img_merge = np.hstack([img, img_out, img_mask])
    
    cv2.imwrite("a.bmp", img_merge)
    
    return


def mano_tx():
    model_path = "/cfs/cfs-oom9s50f/users/blairzheng/codes/smplx/models/mano"
    output_dir = "/cfs/cfs-oom9s50f/users/blairzheng/codes/data/hp/12"

    betas = torch.tensor([[-2.5099, 00.1018, -0.6341, 00.0132, 00.2387, -0.1313, 00.1155, 00.2773, -0.1189, -0.2002]])
    # hand_pose = torch.zeros([1, 15*3])
    # root_pose = torch.zeros([1, 1*3])
    transl = torch.tensor([[0.0, 0.0, 1.0]])
    # hand_pose = torch.tensor([[00.0142, 00.1665, 00.3722, -0.1792, -0.0100, -0.5287, 00.1317, -0.2723, -0.2277,
    #                            -0.1399, 00.1626, 00.4211, -0.0460, 00.0666, 00.3086, -0.0468, -0.0487, 00.2785,
    #                            -0.3905, 00.3746, 00.2121, -0.1821, 00.2086, 00.3088, -0.4430, 00.1422, 00.4402,
    #                            -0.3289, 00.2303, 00.5968, -0.2206, 00.1250, 00.2318, -0.2407, -0.0142, 00.5726,
    #                            -0.0362, -0.3250, -0.2363, -0.0691, 00.0652, 00.3312, 00.1396, -0.2450, -0.2181]])
    hand_pose = torch.tensor([[00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, # 食指
                               00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, # 中食
                               -0.3905, 00.3746, 00.2121, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, # 小指
                               00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, # 无名指
                               00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000]]) # 拇指
    root_pose = torch.tensor([[00.0000, 00.0000, 00.0000]])
    # root_pose = torch.tensor([[-1.3194, 00.2784, -2.7738]])
    # transl = torch.tensor([[-0.1896,-0.1467,01.0980]])
    
    model = MANO(model_path, True, False, flat_hand_mean=True)

    output = model.forward(betas, root_pose, hand_pose, transl)
    vertices = output.vertices * 1000
    img = np.zeros([512, 512])
    
    name = "j3"
    img_out, mask = render(img.shape[:2], vertices, output.faces.repeat(1, 1, 1),
                           torch.tensor([1000]).reshape(1, 1), torch.tensor([300, 200]).reshape(1, 2))
    cv2.imwrite(os.path.join(output_dir, f"{name}.png"), img_out[0].astype(np.uint8))
    
    model.to_obj(vertices[0]*torch.tensor([[1, 1, 1]]), output.faces, os.path.join(output_dir, f"{name}.obj"))
    
    return


def calc_depth(root, pinkie2d, focal, princpt, length):
    pinkie_temp = (pinkie2d - princpt)/focal
    a = pinkie_temp[0]**2 + pinkie_temp[1]**2 + 1
    b = -1*(2*pinkie_temp[0]*root[0] + 2*pinkie_temp[1]*root[1] + 2*root[2])
    c = root[0]**2 + root[1]**2 + root[2]**2 - length**2
    
    flag = b**2 - 4*a*c
    x1 = (-b + np.sqrt(flag))/(2*a)
    x2 = (-b - np.sqrt(flag))/(2*a)
    
    return np.array([x1]), np.array([x2])
   

class Rotation:
    def __init__(self):
        return

    @staticmethod
    def add_yaw_mat(mat_curr, theta):
        mat_theta = Rotation.get_z_yaw_mat(theta)
        euler_degree, mat_new = Rotation.add_rotation_mat(mat_curr, mat_theta)
        return euler_degree, mat_new

    @staticmethod
    def add_pitch_mat(mat_curr, theta):
        mat_theta = Rotation.get_y_pitch_mat(theta)
        euler_degree, mat_new = Rotation.add_rotation_mat(mat_curr, mat_theta)
        return euler_degree, mat_new

    @staticmethod
    def add_roll_mat(mat_curr, theta):
        mat_theta = Rotation.get_x_roll_mat(theta)
        euler_degree, mat_new = Rotation.add_rotation_mat(mat_curr, mat_theta)
        return euler_degree, mat_new

    @staticmethod
    def add_rotation_mat(mat_curr, mat_rot, is_inner=True):
        if is_inner:
            mat_new = np.dot(mat_curr, mat_rot)
        else:
            mat_new = np.dot(mat_rot, mat_curr)
        euler_radian = Rotation.rot_mat_to_euler_angle(mat_new)
        euler_degree = 180 * euler_radian / np.pi

        return euler_degree, mat_new
     
    @staticmethod
    def add_yaw(yaw, pitch, roll, theta):
        mat_theta = Rotation.get_z_yaw_mat(theta)
        euler_degree = Rotation.add_rotation(yaw, pitch, roll, mat_theta)
        return euler_degree

    @staticmethod
    def add_pitch(yaw, pitch, roll, theta):
        mat_theta = Rotation.get_y_pitch_mat(theta)
        euler_degree = Rotation.add_rotation(yaw, pitch, roll, mat_theta)
        return euler_degree

    @staticmethod
    def add_roll(yaw, pitch, roll, theta):
        mat_theta = Rotation.get_x_roll_mat(theta)
        euler_degree = Rotation.add_rotation(yaw, pitch, roll, mat_theta)
        return euler_degree
    
    @staticmethod
    def add_rotation(yaw, pitch, roll, mat_rot, is_inner=True):
        mat_yaw = Rotation.get_z_yaw_mat(yaw)
        mat_pitch = Rotation.get_y_pitch_mat(pitch)
        mat_roll = Rotation.get_x_roll_mat(roll)
        mat_curr = np.dot(np.dot(mat_yaw, mat_pitch), mat_roll)
        if is_inner:
            mat_new = np.dot(mat_curr, mat_rot)
        else:
            mat_new = np.dot(mat_rot, mat_curr)
        euler_radian = Rotation.rot_mat_to_euler_angle(mat_new)
        euler_degree = 180*euler_radian/np.pi
        # print("add rotation", yaw, pitch, roll, mat_curr, 180/np.pi*Rotation.rot_mat_to_euler_angle(mat_curr), euler_degree)

        return euler_degree
    
    @staticmethod
    def get_z_yaw_mat(theta, to_radian=True):
        if to_radian:
            theta = np.pi*theta/180
        mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [0,             0,              1]])
        return mat
    
    @staticmethod
    def get_y_pitch_mat(theta, to_radian=True):
        if to_radian:
            theta = np.pi*theta/180
        mat = np.array([[np.cos(theta), 0,              np.sin(theta)],
                        [0,             1,              0],
                        [-np.sin(theta), 0,             np.cos(theta)]])
        return mat
    
    @staticmethod
    def get_x_roll_mat(theta, to_radian=True):
        if to_radian:
            theta = np.pi*theta/180
        mat = np.array([[1,             0,              0],
                        [0,             np.cos(theta),  -np.sin(theta)],
                        [0,             np.sin(theta),  np.cos(theta)]])
        return mat
    
    @staticmethod
    def rot_mat_to_euler_angle(R):
        
        def is_rotation_matrix(R):
            Rt = np.transpose(R)
            should_be_identity = np.dot(Rt, R)
            I = np.identity(3, dtype=R.dtype)
            n = np.linalg.norm(I - should_be_identity)
            return n < 1e-6
        
        assert (is_rotation_matrix(R))

        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([z, y, x])


class TrackBall:
    def __init__(self, radius):
        self.radius = radius
        return

    def get_rotation(self, pt1, pt2):
        z1 = self.calc_z(pt1)
        z2 = self.calc_z(pt2)
        
        v1 = np.array([pt1[0], pt1[1], z1])
        v2 = np.array([pt2[0], pt2[1], z2])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        angle = np.arccos(np.dot(v1, v2) / (norm1 * norm2))
        axis = np.cross(v1 / norm1, v2 / norm2)
        
        return axis * angle

    def calc_z(self, pt):
        plane_r_square = pt[0] ** 2 + pt[1] ** 2
        if plane_r_square < 0.5 * self.radius ** 2:
            z = np.sqrt(self.radius ** 2 - plane_r_square)
        else:
            z = self.radius / 2 / np.sqrt(plane_r_square)
        return -z
    
    
def mano_interactive_tx():
    # model_path = "/cfs/cfs-oom9s50f/users/blairzheng/codes/smplx/models/mano"
    # output_dir = "/cfs/cfs-oom9s50f/users/blairzheng/codes/data/hp/13"
    model_path = "D:\\codes\\WeSee\\labelme\\Model\\MANO"
    output_dir = "D:\\dev_data\\model_data\\fingerprint\\outputs\\vimages\\hp\\13"

    # root_ori = np.array([107.7189,   7.4273,   6.8122])
    # pinkie_ori = np.array([30.9621, -3.8904, -42.3209])
    # 
    # focal = np.array([1000])
    # princpt = np.array([200, 200])
    # root2d = np.array([22, 82])
    # pinkie2d = np.array([88, 85])
    # root_depth = np.array([800])
    # 
    # root_cam2d = (root2d-princpt)*root_depth/focal
    # root = np.hstack([root_cam2d, root_depth])
    # transl = torch.from_numpy((root - root_ori)/1000).unsqueeze(0)
    # length = np.linalg.norm(root_ori-pinkie_ori)
    # 
    # pke_depth1, pke_depth2 = calc_depth(root, pinkie2d, focal, princpt, length)
    # pinkie = np.hstack([pinkie2d, pke_depth1])
    # rotate_axis = root - pinkie
    # rotate_axis = np.pi*rotate_axis/np.linalg.norm(rotate_axis)
    # root_pose = torch.from_numpy(rotate_axis)
    # 
    # print(root_ori)
    # print(root)
    # print(transl)
    
    betas = torch.tensor([[-2.5099, 00.1018, -0.6341, 00.0132, 00.2387, -0.1313, 00.1155, 00.2773, -0.1189, -0.2002]])
    hand_pose = torch.zeros([1, 15*3])
    root_pose = torch.tensor([[0.0, 0.0, 0.0]])
    transl = torch.tensor([[0.0, 0.0, 0.0]])
    # hand_pose = torch.tensor([[00.0142, 00.1665, 00.3722, -0.1792, -0.0100, -0.5287, 00.1317, -0.2723, -0.2277,
    #                            -0.1399, 00.1626, 00.4211, -0.0460, 00.0666, 00.3086, -0.0468, -0.0487, 00.2785,
    #                            -0.3905, 00.3746, 00.2121, -0.1821, 00.2086, 00.3088, -0.4430, 00.1422, 00.4402,
    #                            -0.3289, 00.2303, 00.5968, -0.2206, 00.1250, 00.2318, -0.2407, -0.0142, 00.5726,
    #                            -0.0362, -0.3250, -0.2363, -0.0691, 00.0652, 00.3312, 00.1396, -0.2450, -0.2181]])
    # hand_pose = torch.tensor([[00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000,
    #                            00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000,
    #                            00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000,
    #                            00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000,
    #                            00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000]])
    # root_pose = torch.tensor([[-1.24467953, 0.12537395, -2.68213239]])
    # root_pose = torch.tensor([[-1.3194, 00.2784, -2.7738]])
    # transl = torch.tensor([[-0.1896, -0.1467, 01.0980]])

    model = MANO(model_path, False, False, flat_hand_mean=True)

    output = model.forward(betas, root_pose, hand_pose, transl)
    vertices = output.vertices * 1000
    joints = output.joints * 1000
    
    img = np.zeros([512, 334])

    name = "l0"
    img_out, mask = render(img.shape[:2], vertices[0], output.faces,
                           torch.tensor([1000]), torch.tensor([200, 200]), "0")
    cv2.imwrite(os.path.join(output_dir, f"{name}.png"), img_out.astype(np.uint8))

    model.to_obj(vertices[0] * torch.tensor([[1, 1, 1]]), output.faces, os.path.join(output_dir, f"{name}.obj"))
    
    print(joints[0].numpy()[:4])
    print(joints[0].numpy().min(axis=0))
    print(joints[0].numpy().max(axis=0))

    return


def rot_mat_to_euler_angle_tx():
    rot_mat1 = cv2.Rodrigues(np.array([[-1.3194, 00.2784, -2.7738]]))[0]
    rot = Rotation()
    euler_angle = rot.rot_mat_to_euler_angle(rot_mat1)
    
    yaw_mat = rot.get_z_yaw_mat(euler_angle[0], False)
    pitch_mat = rot.get_y_pitch_mat(euler_angle[1], False)
    roll_mat = rot.get_x_roll_mat(euler_angle[2], False)

    rot_mat2 = np.dot(yaw_mat, np.dot(pitch_mat, roll_mat))
    
    print(180*euler_angle/np.pi)
    print(rot_mat1)
    print(rot_mat2)
    
    print("euler angle")
    print(cv2.Rodrigues(yaw_mat)[0])
    print(cv2.Rodrigues(pitch_mat)[0])
    print(cv2.Rodrigues(roll_mat)[0])
    
    print(cv2.Rodrigues(np.dot(pitch_mat, yaw_mat))[0].reshape(-1))
    return


def open3d_tx():
    model_path = "D:\\codes\\WeSee\\labelme\\Model\\MANO"
    output_dir = "D:\\dev_data\\model_data\\fingerprint\\outputs\\vimages\\hp\\13"
 
    betas = torch.tensor([[-2.5099, 00.1018, -0.6341, 00.0132, 00.2387, -0.1313, 00.1155, 00.2773, -0.1189, -0.2002]]) 
    hand_pose = torch.tensor([[00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000,
                               00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000,
                               00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000,
                               00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000,
                               00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000, 00.0000]])
    root_pose = torch.tensor([[-1.24467953, 0.12537395, -2.68213239]])
    
    device = "cpu"
    focal = torch.tensor([1000], dtype=torch.float32, device=device)
    princpt = torch.tensor([200, 200], dtype=torch.float32, device=device)
    xy, depth = np.array([0, 0]), 0
    transl = torch.tensor([[0, 0, depth]], dtype=torch.float, device=device)
    transl[0, :2] = (torch.from_numpy(xy) - princpt) * depth / focal
    transl = transl / 1000
    print(transl)
    # transl = torch.tensor([[-0.1896, -0.1467, 01.0980]])

    model = MANO(model_path, True, False, flat_hand_mean=True)

    output = model.forward(betas, root_pose, hand_pose, transl)
    vertices = output.vertices * 1000
    joints = output.joints * 1000

    img = np.zeros([512, 334, 3])

    name = "l3"
    img_out, mask = render(img.shape[:2], vertices[0], output.faces,
                           torch.tensor([1000]), torch.tensor([200, 200]), "0")
    
    img_out, mask = img_out, mask * 0.8
    img_mask = (img * (1 - mask) + img_out * mask).astype(np.uint8)
    img_merge = np.hstack([img_out, img_mask])
    cv2.imwrite(os.path.join(output_dir, f"{name}.png"), img_merge.astype(np.uint8))

    model.to_obj(vertices[0] * torch.tensor([[1, 1, 1]]), output.faces, os.path.join(output_dir, f"{name}.obj"))

    print(joints[0].numpy()[:4])
    print(joints[0].numpy().min(axis=0))
    print(joints[0].numpy().max(axis=0))

    return


if __name__ == "__main__":
    # mano_validate_tx()
    # mano_tx()
    mano_interactive_tx()
    # rot_mat_to_euler_angle_tx()
    # open3d_tx()