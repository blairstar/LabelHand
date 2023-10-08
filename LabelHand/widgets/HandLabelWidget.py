from qtpy.QtCore import Qt
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.QtWidgets import (QWidget, QFrame, QLabel, QPushButton, QLCDNumber, QLineEdit, QSlider, QDial,
                            QComboBox, QSpinBox, QCheckBox, QHBoxLayout, QVBoxLayout, QDoubleSpinBox)
from qtpy.QtGui import QIntValidator, QFont
import numpy as np
from LabelHand.Alg.ManoTx import Rotation, local_time
import matplotlib.pyplot as plt
import cv2
import os
import glob


class WidgetSize(object):
    sld_lcd_min_h = 40
    sld_min_w = 60
    sld_btn_fix_w = 20
    sld_btn_fix_h = 20
    dial_min_h = 70
    dial_btn_min_w = 20
    dial_btn_min_h = 20
    dial_reset_btn_min_w = 30
    dial_reset_btn_min_h = 30
    dial_spin_min_h = 40
    kp_coord_min_w = 30
    kp_coord_min_h = 40
    kp_btn_min_w = 15
    kp_btn_min_h = 18
    global_min_h = 30

    def __init__(self):
        self.base_width, self.base_height = 1920, 1080
        return

    def rescale(self, width, height):
        w_ratio = width / self.base_width
        h_ratio = height / self.base_height

        for key in dir(WidgetSize):
            if key.startswith("_"):
                continue
            value = self.__getattribute__(key)
            if key.endswith("_w"):
                self.__setattr__(key, value * w_ratio)
            if key.endswith("_h"):
                self.__setattr__(key, value * h_ratio)
        
        self.base_width, self.base_height = width, height
        return


class HandPoseParam(object):
    def __init__(self):
        self.spin_x, self.spin_y, self.sld_depth = 0, 0, 0
        self.pose_yaw, self.pose_pitch, self.pose_roll = 0, 0, 0
        self.thumb1_yaw, self.thumb1_pitch, self.thumb1_roll, self.sld_thumb2 = 0, 0, 0, 0
        self.index0_pitch, self.index0_roll, self.sld_index1, self.sld_index2, self.sld_index3 = 0, 0, 0, 0, 0
        self.middle0_pitch, self.middle0_roll, self.sld_middle1, self.sld_middle2, self.sld_middle3 = 0, 0, 0, 0, 0
        self.ring0_pitch, self.ring0_roll, self.sld_ring1, self.sld_ring2, self.sld_ring3 = 0, 0, 0, 0, 0
        self.pinky0_pitch, self.pinky0_roll, self.sld_pinky1, self.sld_pinky2, self.sld_pinky3 = 0, 0, 0, 0, 0
        return

    def from_dict(self, dt):
        # only update existing key
        self.__dict__.update((k, dt[k]) for k in set(dt).intersection(self.__dict__))
        return

    def to_dict(self):
        return self.__dict__

    def set_finger_pose(self, pose):
        self.thumb1_yaw, self.thumb1_pitch, self.thumb1_roll, self.sld_thumb2 = pose[0]
        self.index0_pitch, self.index0_roll, self.sld_index1, self.sld_index2, self.sld_index3 = pose[1]
        self.middle0_pitch, self.middle0_roll, self.sld_middle1, self.sld_middle2, self.sld_middle3 = pose[2]
        self.ring0_pitch, self.ring0_roll, self.sld_ring1, self.sld_ring2, self.sld_ring3 = pose[3]
        self.pinky0_pitch, self.pinky0_roll, self.sld_pinky1, self.sld_pinky2, self.sld_pinky3 = pose[4]
        return
    
    def get_coord(self):
        return self.spin_x, self.spin_y, self.sld_depth
    
    def set_coord(self, x, y, depth):
        self.spin_x, self.spin_y, self.sld_depth = x, y, depth
        return

    def recify_thumb(self, thumb_pose1, thumb_pose2, blend_angle):
        angle = 60 * (max(0, abs(blend_angle) - 10)) / (70 - 10)
        rot1 = Rotation.get_x_roll_mat(angle)
        rot2 = Rotation.get_x_roll_mat(-60)
        thumb_pose1 = np.dot(thumb_pose1, rot1)
        thumb_pose2 = np.dot(rot2.transpose(), np.dot(thumb_pose2, rot2))

        return thumb_pose1, thumb_pose2
    
    def get_local_frame(self, is_rhand):
        z_mat = Rotation.get_z_yaw_mat(self.pose_yaw)
        y_mat = Rotation.get_y_pitch_mat(self.pose_pitch)
        x_mat = Rotation.get_x_roll_mat(self.pose_roll)
        root_pose = np.dot(np.dot(z_mat, y_mat), x_mat)
        
        x_unit, y_unit, z_unit = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
        x_unit = x_unit if is_rhand else -1*x_unit
        x_unit, y_unit, z_unit = root_pose.dot(x_unit), root_pose.dot(y_unit), root_pose.dot(z_unit)

        xy = np.array([self.spin_x, self.spin_y])
        depth = np.array([self.sld_depth])
        
        return x_unit, y_unit, z_unit, xy, depth
    
    @staticmethod
    def get_finger_pose(yaw, pitch, roll, angle2, angle3):
        z_mat = Rotation.get_z_yaw_mat(yaw)
        y_mat = Rotation.get_y_pitch_mat(pitch)
        x_mat = Rotation.get_x_roll_mat(roll)
        
        pose0 = np.dot(np.dot(y_mat, x_mat), z_mat)
        # pose0 = np.dot(z_mat, np.dot(y_mat, x_mat))
        pose1 = Rotation.get_z_yaw_mat(angle2)
        pose2 = Rotation.get_z_yaw_mat(angle3)
        return pose0, pose1, pose2

    def to_mano_param(self, is_rhand):
        z_mat = Rotation.get_z_yaw_mat(self.pose_yaw)
        y_mat = Rotation.get_y_pitch_mat(self.pose_pitch)
        x_mat = Rotation.get_x_roll_mat(self.pose_roll)
        root_pose = np.dot(np.dot(z_mat, y_mat), x_mat)
        root_pose = root_pose[None, ...]

        thumb2 = self.sld_thumb2
        index0_pitch, index0_roll, index0_yaw = self.index0_pitch, self.index0_roll, self.sld_index1
        index2, index3 = self.sld_index2, self.sld_index3
        middle0_pitch, middle0_roll, middle0_yaw = self.middle0_pitch, self.middle0_roll, self.sld_middle1
        middle2, middle3 = self.sld_middle2, self.sld_middle3
        ring0_pitch, ring0_roll, ring0_yaw = self.ring0_pitch, self.ring0_roll, self.sld_ring1
        ring2, ring3 = self.sld_ring2, self.sld_ring3
        pinky0_pitch, pinky0_roll, pinky0_yaw = self.pinky0_pitch, self.pinky0_roll, self.sld_pinky1
        pinky2, pinky3 = self.sld_pinky2, self.sld_pinky3

        if not is_rhand:
            thumb2 = -1*thumb2
            index0_yaw, index2, index3 = -1*index0_yaw, -1*index2, -1*index3
            middle0_yaw, middle2, middle3 = -1*middle0_yaw, -1*middle2, -1*middle3
            ring0_yaw, ring2, ring3 = -1*ring0_yaw, -1*ring2, -1*ring3
            pinky0_yaw, pinky2, pinky3 = -1*pinky0_yaw, -1*pinky2, -1*pinky3
        
        if is_rhand:
            index0_pitch, middle0_pitch = -1*index0_pitch, -1*middle0_pitch
            ring0_pitch, pinky0_pitch = -1*ring0_pitch, -1*pinky0_pitch
        
        thumb_pose0 = np.eye(3, dtype=np.float32)
        z_mat = Rotation.get_z_yaw_mat(self.thumb1_yaw)
        y_mat = Rotation.get_y_pitch_mat(self.thumb1_pitch)
        x_mat = Rotation.get_x_roll_mat(self.thumb1_roll)
        thumb_pose1 = np.dot(np.dot(z_mat, y_mat), x_mat)
        thumb_pose2 = Rotation.get_z_yaw_mat(thumb2)

        index_pose = self.get_finger_pose(index0_yaw, index0_pitch, index0_roll, index2, index3)
        index_pose0, index_pose1, index_pose2 = index_pose
        middle_pose = self.get_finger_pose(middle0_yaw, middle0_pitch, middle0_roll, middle2, middle3)
        middle_pose0, middle_pose1, middle_pose2 = middle_pose
        ring_pose = self.get_finger_pose(ring0_yaw, ring0_pitch, ring0_roll, ring2, ring3)
        ring_pose0, ring_pose1, ring_pose2 = ring_pose
        pinky_pose = self.get_finger_pose(pinky0_yaw, pinky0_pitch, pinky0_roll, pinky2, pinky3)
        pinky_pose0, pinky_pose1, pinky_pose2 = pinky_pose

        finger_pose = np.stack([index_pose0, index_pose1, index_pose2,
                                middle_pose0, middle_pose1, middle_pose2,
                                pinky_pose0, pinky_pose1, pinky_pose2,
                                ring_pose0, ring_pose1, ring_pose2,
                                thumb_pose0, thumb_pose1, thumb_pose2])

        xy = np.array([self.spin_x, self.spin_y])
        depth = np.array([self.sld_depth])

        return xy, depth, root_pose, finger_pose


class HandPoseWidget(QWidget):
    sig_config_changed = QtCore.Signal(bool)

    def __init__(self, name, parent, mano, beta_widget, global_widget):
        super(HandPoseWidget, self).__init__(parent)
        self.setObjectName(name)
        self.param = HandPoseParam()
        self.mano = mano
        self.beta_widget = beta_widget
        self.global_widget = global_widget
        
        hbox_pos = self.first_row_widget()
        
        hbox_pose = self.second_row_widget()
        
        hbox_thumb = self.create_thumb_widget(-10, 90)

        dt_sld_index = {"index0": [(-30, 30), (-45, 45)], "sld_index1": (-15, 110),
                        "sld_index2": (-10, 120), "sld_index3": (-10, 90)}
        hbox_index = self.create_finger_widget("index", dt_sld_index)
        
        dt_sld_middle = {"middle0": [(-20, 20), (-45, 45)], "sld_middle1": (-15, 110),
                         "sld_middle2": (-10, 120), "sld_middle3": (-10, 90)}
        hbox_middle = self.create_finger_widget("middle", dt_sld_middle)
        
        dt_sld_ring = {"ring0": [(-25, 25), (-45, 45)], "sld_ring1": (-15, 110),
                       "sld_ring2": (-10, 120), "sld_ring3": (-10, 90)}
        hbox_ring = self.create_finger_widget("ring", dt_sld_ring)
        
        dt_sld_pinky = {"pinky0": [(-40, 40), (-45, 45)], "sld_pinky1": (-15, 110),
                        "sld_pinky2": (-10, 120), "sld_pinky3": (-10, 90)}
        hbox_pinky = self.create_finger_widget("pinky", dt_sld_pinky)

        vbox = QVBoxLayout()
        vbox.setSpacing(5)
        vbox.addLayout(hbox_pos)
        vbox.addLayout(hbox_pose)
        vbox.addSpacing(5)
        vbox.addWidget(self.new_seperator())
        vbox.addSpacing(5)
        vbox.addLayout(hbox_thumb)
        vbox.addLayout(hbox_index)
        vbox.addLayout(hbox_middle)
        vbox.addLayout(hbox_ring)
        vbox.addLayout(hbox_pinky)
        vbox.addStretch(1)
        self.setLayout(vbox)
        
        self.spin_step.setValue(2)
        self.save_param()
        return
    
    def create_thumb_widget(self, thumb2_min, thumb2_max):
        dt_pose = {"0": [40, -16, 0], "1": [0, 0, 0], "2": [-42, -1, 38], "3": [-1, 55, 24], "4": [-32, -50, 20]}
        self.thumb1 = DialThree("thumb1", self, dt_pose, self.tr("thumb pose"), False, False, self.mano.is_rhand)
        self.sld_thumb2 = SliderPlus("sld_thumb2", self, thumb2_min, thumb2_max)
        
        hbox = QHBoxLayout()
        hbox.setSpacing(10)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.thumb1, stretch=8)
        hbox.addWidget(self.sld_thumb2, stretch=2)
        
        return hbox
    
    def create_finger_widget(self, label_name, dt_slide_info):
        hbox = QHBoxLayout()
        hbox.setSpacing(10)
        hbox.setContentsMargins(0, 0, 0, 0)

        for name, info in dt_slide_info.items():
            if name in ["index0", "middle0", "ring0", "pinky0"]:
                tip = self.tr(f"euler angle of {name}")
                obj = CrossPose(name, self, tip, 2, info[0], info[1])
                setattr(self, name, obj)
                hbox.addWidget(obj, stretch=4)
            elif name.find("sld") >= 0:
                min_val, max_val = info
                obj = SliderPlus(name, self, min_val, max_val)
                setattr(self, name, obj)
                hbox.addWidget(obj, stretch=6)
            else:
                raise Exception("unsupported widget")
        return hbox
    
    def append_reset_btn(self, box):
        self.btn_reset = QPushButton("X", self)
        self.btn_reset.setMinimumSize(25, 25)
        self.btn_reset.clicked.connect(self.reset_finger_pose)
        box.addStretch(5)
        box.addWidget(self.btn_reset, stretch=5)
        box.addStretch(5)
        return box
            
    def get_draw_config(self):
        with_img = self.cb_img.isChecked()
        with_mesh = self.cb_mesh.isChecked()
        with_skl = self.cb_skl.checkState()
        return with_img, with_mesh, with_skl
    
    def config_changed(self, is_dirty=True):
        self.save_param()
        self.sig_config_changed.emit(is_dirty)
        return
     
    def display_param(self):
        for key, value in self.param.__dict__.items():
            if key.find("sld") >= 0:
                obj = self.findChild(QSlider, key)
                if obj is not None:
                    obj.setValue(int(value))
            elif key.find("qle") >= 0:
                obj = self.findChild(QLineEdit, key)
                if obj is not None:
                    obj.setText(str(value))
            elif key.find("spin") >= 0:
                obj = self.findChild(QDoubleSpinBox, key)
                if obj is not None:
                    state = obj.blockSignals(True)
                    obj.setValue(value)
                    obj.blockSignals(state)
            elif key.find("pose") >= 0:
                obj = self.findChild(QWidget, "pose")
                if obj is not None:
                    setattr(obj, key.replace("pose_", ""), value)
                    obj.show_state()
            elif key.find("thumb1") >= 0:
                obj = self.findChild(QWidget, "thumb1")
                if obj is not None:
                    setattr(obj, key.replace("thumb1_", ""), value)
                    obj.show_state()
            elif key.split("_")[0] in ["index0", "middle0", "ring0", "pinky0"]:
                obj = self.findChild(CrossPose, key.split("_")[0])
                if obj is not None:
                    setattr(obj, key.split("_")[1], value)
                    obj.show_state()
            else:
                pass
            print("hand label display", key, value, obj is not None)
        return
    
    def save_param(self):
        for key, value in self.param.__dict__.items():
            if key.find("sld") >= 0:
                obj = self.findChild(QSlider, key)
                if obj is not None:
                    setattr(self.param, key, obj.value())
            elif key.find("qle") >= 0:
                obj = self.findChild(QLineEdit, key)
                if obj is not None:
                    setattr(self.param, key, int(obj.text()))
            elif key.find("spin") >= 0:
                obj = self.findChild(QDoubleSpinBox, key)
                if obj is not None:
                    setattr(self.param, key, float(obj.value()))
            elif key.find("pose") >= 0:
                obj = self.findChild(QWidget, "pose")
                if obj is not None:
                    setattr(self.param, key, getattr(obj, key.replace("pose_", "")))
            elif key.find("thumb1") >= 0:
                obj = self.findChild(QWidget, "thumb1")
                if obj is not None:
                    setattr(self.param, key, getattr(obj, key.replace("thumb1_", "")))
            elif key.split("_")[0] in ["index0", "middle0", "ring0", "pinky0"]:
                obj = self.findChild(CrossPose, key.split("_")[0])
                if obj is not None:
                    setattr(self.param, key, getattr(obj, key.split("_")[1], ""))
            else:
                pass
            # print("hand label save", key, value, obj is None)
        return
     
    def new_seperator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setMidLineWidth(1)
        return line

    @staticmethod
    def create_spin(name, parent, init_val, scope, step, tip, decimal=0):
        obj = QDoubleSpinBox(parent)
        obj.setDecimals(decimal)
         
        if scope is not None:
            obj.setRange(scope[0], scope[1])
        else:
            obj.setRange(-9999, 9999)
        obj.setObjectName(name)
        obj.setValue(init_val)
        obj.setSingleStep(step)
        obj.setToolTip(tip)
        obj.setWrapping(True)
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setMinimumSize(40, 40)
        return obj
    
    def create_beta_combo(self):
        beta_list = QComboBox(self)
        beta_list.setEditable(False)
        for ii in range(self.beta_widget.beta_list.count()):
            beta_list.addItem(self.beta_widget.beta_list.itemText(ii))
        beta_list.setCurrentIndex(self.beta_widget.beta_list.currentIndex())
        beta_list.activated[int].connect(self.beta_widget.beta_list.setCurrentIndex)
        beta_list.setContentsMargins(0, 0, 0, 0)
        beta_list.setMinimumHeight(30)
        return beta_list
    
    def create_finger_combo(self):
        pose0 = [[40, -16, 0, 0],
                 [10, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [-10, 0, 0, 0, 0],
                 [-25, 0, 0, 0, 0]]
        pose1 = [[46, 8, -12, 0],
                 [20, 0, 90, 90, 45],
                 [0, 0, 100, 90, 40],
                 [0, 0, 90, 90, 45],
                 [-20, -10, 90, 90, 45]]
        pose2 = [[0, 0, 0, 0],
                 [-5, 0, 0, 0, 0],
                 [-5, 0, 0, 0, 0],
                 [3, 0, 0, 0, 0],
                 [5, 0, 0, 0, 0]]
        pose3 = [[0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]
        
        self.dt_finger_pose = {"0": pose0, "1": pose1, "2": pose2, "3": pose3}
        for key, value in self.dt_finger_pose.items():
            value[0][0] = value[0][0] if self.mano.is_rhand else -1*value[0][0]
            value[0][2] = value[0][2] if self.mano.is_rhand else -1*value[0][2]
            self.dt_finger_pose[key] = value
        obj = QComboBox(self)
        obj.setEditable(False)
        
        for key in self.dt_finger_pose.keys():
            obj.addItem(key)
        obj.setCurrentIndex(0)
        obj.activated[int].connect(self.select_finger_pose)
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setMinimumHeight(30)
        obj.setToolTip(self.tr("finger pose template"))
        return obj
    
    def select_finger_pose(self, index):
        self.param.set_finger_pose(self.dt_finger_pose[str(index)])
        self.display_param()
        self.config_changed()
        return
    
    def create_checkbox(self, name, parent, tip, check_state, is_tristate=False):
        obj = QCheckBox(parent)
        obj.setObjectName(name)
        obj.setToolTip(tip)
        state = obj.blockSignals(True)
        if is_tristate:
            obj.setTristate(True)
            obj.setCheckState(check_state)
        else:
            obj.setChecked(check_state)
        obj.blockSignals(state)
        obj.setContentsMargins(0, 0, 0, 0)
        return obj
     
    def first_row_widget(self):
        self.cb_img = self.create_checkbox("cb_img", self, "with image", True)
        self.cb_mesh = self.create_checkbox("cb_mesh", self, "with mesh", True)
        self.cb_skl = self.create_checkbox("cb_skl", self, "with skeleton", 0, True)
        
        self.list_beta = self.create_beta_combo()
        self.spin_x = self.create_spin("spin_x", self, 100, (0, 5000), 1, self.tr("x coordination of hand position"), 1)
        self.spin_y = self.create_spin("spin_y", self, 100, (0, 5000), 1, self.tr("y coordination of hand position"), 1)
        self.sld_depth = SliderPlus("sld_depth", self, 500, 2000, step=10, default_value=1000)
        self.list_finger_pose = self.create_finger_combo()
        
        self.cb_img.toggled.connect(lambda: self.config_changed(False))
        self.cb_mesh.toggled.connect(lambda: self.config_changed(False))
        self.cb_skl.stateChanged.connect(lambda: self.config_changed(False))
        self.spin_x.valueChanged.connect(lambda: self.config_changed(True))
        self.spin_y.valueChanged.connect(lambda: self.config_changed(True))

        hbox = QHBoxLayout()
        hbox.setSpacing(5)
        hbox.addWidget(self.cb_img, stretch=1)
        hbox.addWidget(self.cb_mesh, stretch=1)
        hbox.addWidget(self.cb_skl, stretch=1)
        hbox.addWidget(self.list_beta, stretch=2)
        hbox.addSpacing(5)
        hbox.addWidget(self.list_finger_pose, stretch=1)
        hbox.addSpacing(5)
        hbox.addWidget(self.spin_x, stretch=3)
        hbox.addWidget(self.spin_y, stretch=3)
        hbox.addWidget(self.sld_depth, stretch=12)
        
        return hbox
    
    def set_hand_pose(self, x, y):
        state = self.spin_x.blockSignals(True)
        self.spin_x.setValue(x)
        self.spin_x.blockSignals(state)
        
        state = self.spin_y.blockSignals(True)
        self.spin_y.setValue(y)
        self.spin_y.blockSignals(state)
        return
    
    def second_row_widget(self):
        hbox = QHBoxLayout()
        hbox.setSpacing(10)
        hbox.setContentsMargins(0, 0, 0, 0)

        dt_pose = {"0": [90, 0, 0], "1": [180, 0, 0], "2": [-180, -55, -95], "3": [-150, -50, -90], "4": [0, 0, 0]}
        self.pose = DialThree("pose", self, dt_pose, self.tr("palm pose"), False, True, self.mano.is_rhand)

        self.spin_step = self.create_spin("step", self, 1, (1, 10), 1, self.tr("step of slider and spin box")) 
        
        self.btn_add_x = self.create_button("+", self, self.tr("add local x, toward wrist"), (25, 30))
        self.btn_sub_x = self.create_button("-", self, self.tr("sub local x, toward finger tip"), (25, 30))
        self.btn_add_z = self.create_button("+", self, self.tr("add local z, toward thumb"), (25, 30), "200, 200, 250")
        self.btn_sub_z = self.create_button("-", self, self.tr("sub local z, toward pinky"), (25, 30), "210, 210, 250")
        self.btn_add_y = self.create_button("+", self, self.tr("add local y, toward palm back"), (25, 30))
        self.btn_sub_y = self.create_button("-", self, self.tr("sub local y, toward palm"), (25, 30))

        self.btn_add_x.clicked.connect(lambda: self.change_local_root("x", 1))
        self.btn_sub_x.clicked.connect(lambda: self.change_local_root("x", -1))
        self.btn_add_z.clicked.connect(lambda: self.change_local_root("z", 1))
        self.btn_sub_z.clicked.connect(lambda: self.change_local_root("z", -1))
        self.btn_add_y.clicked.connect(lambda: self.change_local_root("y", -1))
        self.btn_sub_y.clicked.connect(lambda: self.change_local_root("y", 1))

        vbox_x_sub = QVBoxLayout()
        vbox_x_sub.setSpacing(0)
        vbox_x_sub.setContentsMargins(0, 0, 0, 0)
        vbox_x_sub.addStretch(stretch=1)
        vbox_x_sub.addWidget(self.btn_sub_x, stretch=2)
        vbox_x_sub.addStretch(stretch=1)

        vbox_z = QVBoxLayout()
        vbox_z.setSpacing(0)
        vbox_z.setContentsMargins(0, 0, 0, 0)
        vbox_z.addWidget(self.btn_add_z, stretch=1)
        vbox_z.addWidget(self.btn_sub_z, stretch=1)

        vbox_x_add = QVBoxLayout()
        vbox_x_add.setSpacing(0)
        vbox_x_add.setContentsMargins(0, 0, 0, 0)
        vbox_x_add.addStretch(stretch=1)
        vbox_x_add.addWidget(self.btn_add_x, stretch=2)
        vbox_x_add.addStretch(stretch=1)
        
        vbox_y = QVBoxLayout()
        vbox_y.setSpacing(0)
        vbox_y.setContentsMargins(0, 0, 0, 0)
        vbox_y.addWidget(self.btn_add_y, stretch=1)
        vbox_y.addWidget(self.btn_sub_y, stretch=1)

        hbox_xyz = QHBoxLayout()
        hbox_xyz.setSpacing(0)
        hbox_xyz.setContentsMargins(0, 0, 0, 0)
        hbox_xyz.addLayout(vbox_x_sub, stretch=2)
        hbox_xyz.addLayout(vbox_z, stretch=2)
        hbox_xyz.addLayout(vbox_x_add, stretch=2)
        hbox_xyz.addSpacing(10)
        hbox_xyz.addLayout(vbox_y, stretch=2)

        hbox.addWidget(self.pose, stretch=20)
        hbox.addStretch(1)
        hbox.addWidget(self.spin_step, stretch=2)
        hbox.addStretch(1)
        hbox.addLayout(hbox_xyz, stretch=5)

        self.spin_step.valueChanged.connect(self.set_step)
        return hbox
    
    def change_local_root(self, name, value):
        value = value*self.spin_step.value()
        x_unit, y_unit, z_unit, xy, depth = self.param.get_local_frame(self.mano.is_rhand)
        focal, princpt = self.global_widget.param.get_focal_princpt()
        (cam_x, cam_y), cam_z = (xy-princpt)*depth/focal, depth[0]
        if name.lower() == "x":
            vec = x_unit*value
        elif name.lower() == "y":
            vec = y_unit*value
        else:
            vec = z_unit*value
        new_cam_x, new_cam_y, new_cam_z = cam_x + vec[0], cam_y + vec[1], cam_z + vec[2]
        new_xy = np.array([new_cam_x, new_cam_y])*focal/depth + princpt
        new_depth = new_cam_z
        # print(local_time(), name, xy, cam_x, cam_y, cam_z, vec, cam_x, cam_y, cam_z, new_xy)

        state = self.spin_x.blockSignals(True)
        value = min(max(new_xy[0], self.spin_x.minimum()), self.spin_x.maximum())
        self.spin_x.setValue(value)
        self.spin_x.blockSignals(state)
        
        state = self.spin_y.blockSignals(True)
        value = min(max(new_xy[1], self.spin_y.minimum()), self.spin_y.maximum())
        self.spin_y.setValue(value)
        self.spin_y.blockSignals(state)
        
        self.sld_depth.set_value(int(new_depth+0.5))
        self.config_changed()
        return
    
    def create_slides(self, label_name, dt_slide_info):
        hbox = QHBoxLayout()
        hbox.setSpacing(10)
        hbox.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(self)
        self.label.setMinimumSize(40, 25)
        self.label.setText(label_name)
        hbox.addWidget(self.label)

        total_range = 0
        for name, scope in dt_slide_info.items():
            total_range += abs(scope[1]-scope[0])

        for name, scope in dt_slide_info.items():
            sld = SliderPlus(name, self, scope[0], scope[1], step=2, default_value=0)
            # stretch = int(100*abs(scope[1]-scope[0])/total_range)
            setattr(self, name, sld)
            hbox.addWidget(sld, stretch=1)
    
        return hbox
    
    # def set_pose_step(self, value):
    #     self.sld_yaw.set_step(value)
    #     self.sld_pitch.set_step(value)
    #     self.sld_roll.set_step(value)
    #     return
    
    def set_step(self, value):
        self.spin_x.setSingleStep(value)
        self.spin_y.setSingleStep(value)
        
        self.sld_depth.set_step(max(1, min((value-1)*5, 50)))
        
        self.pose.set_step(value)
        self.thumb1.set_step(value)
        self.sld_thumb2.set_step(value)
        
        self.index0.set_step(value)
        self.sld_index1.set_step(value)
        self.sld_index2.set_step(value)
        self.sld_index3.set_step(value)

        self.middle0.set_step(value)
        self.sld_middle1.set_step(value)
        self.sld_middle2.set_step(value)
        self.sld_middle3.set_step(value)

        self.ring0.set_step(value)
        self.sld_ring1.set_step(value)
        self.sld_ring2.set_step(value)
        self.sld_ring3.set_step(value)

        self.pinky0.set_step(value)
        self.sld_pinky1.set_step(value)
        self.sld_pinky2.set_step(value)
        self.sld_pinky3.set_step(value)

        return

    @staticmethod
    def create_button(name, parent, tip, size, color=None):
        obj = QPushButton(name, parent)
        obj.setToolTip(tip)
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        obj.setMinimumSize(size[0], size[1])
        obj.setAutoRepeat(True)
        obj.setAutoRepeatDelay(300)
        obj.setAutoRepeatInterval(100)
        obj.setStyleSheet("border-style:outset|solid")
        if color:
            obj.setStyleSheet(f"background-color: rgb({color})")
        return obj


class EditPlus(QWidget):
    def __init__(self, name, parent, value=0, step=1, tip=""):
        super(EditPlus, self).__init__(parent)
        self.setObjectName(name)
        self.step = step
        self.setToolTip(tip)
        
        self.qle = QLineEdit("0", parent)
        self.qle.setObjectName(name)
        self.qle.setMinimumSize(40, 40)
        self.qle.setText(str(value))

        validator = QIntValidator(self)
        self.qle.setValidator(validator)

        self.add = QPushButton("+", self)
        self.subtract = QPushButton("-", self)
        self.add.setFixedSize(20, 20)
        self.subtract.setFixedSize(20, 20)
        self.add.clicked.connect(self.add_value)
        self.subtract.clicked.connect(self.subtract_value)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addStretch(1)
        vbox.addWidget(self.add)
        vbox.addWidget(self.subtract)
        vbox.addStretch(1)

        hbox = QHBoxLayout()
        hbox.setSpacing(0)
        hbox.addWidget(self.qle)
        hbox.addLayout(vbox)
        hbox.addStretch(0)
        hbox.setContentsMargins(0, 0, 0, 0)

        self.setLayout(hbox)
        # self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        return

    def add_value(self):
        new_val = int(self.qle.text()) + self.step
        self.setText(str(new_val))
        self.parent().config_changed()
        return

    def subtract_value(self):
        new_val = int(self.qle.text()) - self.step
        self.setText(str(new_val))
        self.parent().config_changed()
        return

    def setText(self, text):
        self.qle.setText(text)
        return
    
    def set_step(self, value):
        self.step = value
        return


class SliderPlus(QtWidgets.QWidget):
    def __init__(self, name, parent, min_val, max_val, digit_count=4, step=1, default_value=0):
        super(SliderPlus, self).__init__(parent)
        self.setObjectName(name)
        self.step = step

        self.lcd = QLCDNumber(self)
        self.lcd.setDigitCount(digit_count)
        self.lcd.setFixedSize(40 if max_val < 1000 else 50, 30)
        self.lcd.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setObjectName(name)
        self.slider.setRange(min_val, max_val)
        self.slider.setSingleStep(step)
        self.slider.setMinimumWidth(60)
        self.slider.valueChanged.connect(self.lcd.display)
        self.slider.sliderMoved.connect(lambda: self.parent().config_changed(False))
        self.slider.sliderReleased.connect(lambda: self.parent().config_changed(True))
        self.slider.setValue(default_value)

        self.add = self.create_button("+", self, "add value", (20, 20))
        self.sub = self.create_button("-", self, "sub value", (20, 20))
        self.add.clicked.connect(self.add_value)
        self.sub.clicked.connect(self.sub_value)

        vbox = QVBoxLayout()
        vbox.setSpacing(0)
        vbox.addStretch(0)
        vbox.addWidget(self.add)
        vbox.addWidget(self.sub)
        vbox.addStretch(0)

        hbox = QHBoxLayout()
        hbox.setSpacing(0)
        hbox.addWidget(self.lcd)
        hbox.addSpacing(2)
        hbox.addWidget(self.slider)
        hbox.addLayout(vbox)
        hbox.setContentsMargins(0, 0, 0, 0)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.setLayout(hbox)
        return

    @staticmethod
    def create_button(name, parent, tip, size):
        obj = QPushButton(name, parent)
        obj.setToolTip(tip)
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setFixedSize(size[0], size[1])
        obj.setAutoRepeat(True)
        obj.setAutoRepeatDelay(300)
        obj.setAutoRepeatInterval(100)
        return obj
    
    def add_value(self):
        new_val = self.slider.value() + self.step
        min_val, max_val = self.slider.minimum(), self.slider.maximum()
        new_val = min(max(min_val, new_val), max_val)
        self.slider.setValue(new_val)
        self.parent().config_changed()
        return

    def sub_value(self):
        new_val = self.slider.value() - self.step
        min_val, max_val = self.slider.minimum(), self.slider.maximum()
        new_val = min(max(min_val, new_val), max_val)
        self.slider.setValue(new_val)
        self.parent().config_changed()
        return
    
    def set_value(self, value):
        self.slider.setValue(value)
        return
    
    def set_step(self, step):
        self.step = step
        self.slider.setSingleStep(step)
        return


class SliderPose(QWidget):
    def __init__(self, name, parent, min_val, max_val, digit_count=5):
        super(SliderPose, self).__init__(parent)
        self.setObjectName(name)

        self.lcd = QLCDNumber(self)
        self.lcd.setDigitCount(digit_count)
        self.lcd.setFixedSize(60, 30)
        self.lcd.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setObjectName(name)
        self.slider.setRange(min_val, max_val)
        self.slider.setSingleStep(1)
        self.slider.setMinimumWidth(60)
        self.slider.valueChanged.connect(self.lcd.display)
        self.slider.sliderMoved.connect(self.config_changed)
        self.slider.sliderPressed.connect(self.config_changed)
        self.slider.sliderReleased.connect(self.config_changed)

        self.add = QPushButton("+", self)
        self.subtract = QPushButton("-", self)
        self.add.setFixedSize(20, 20)
        self.subtract.setFixedSize(20, 20)
        self.add.clicked.connect(self.add_value)
        self.subtract.clicked.connect(self.subtract_value)

        vbox = QVBoxLayout()
        vbox.setSpacing(0)
        vbox.addStretch(0)
        vbox.addWidget(self.add)
        vbox.addWidget(self.subtract)
        vbox.addStretch(0)

        hbox = QHBoxLayout()
        hbox.setSpacing(0)
        hbox.addWidget(self.lcd)
        hbox.addSpacing(2)
        hbox.addWidget(self.slider)
        hbox.addLayout(vbox)
        hbox.setContentsMargins(0, 0, 0, 0)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.setLayout(hbox)
        
        self.step = 1
        return
    
    def set_step(self, step):
        self.step = step
        return
    
    def add_value(self):
        self.add_euler_angle(self.step*1)
        return
    
    def subtract_value(self):
        self.add_euler_angle(self.step*-1)
        return

    def add_euler_angle(self, theta):
        yaw = self.parent().sld_yaw.get_value()
        pitch = self.parent().sld_pitch.get_value()
        roll = self.parent().sld_roll.get_value()
        
        if self.objectName() == "sld_yaw":
            euler_degree = Rotation.add_yaw(yaw, pitch, roll, theta)
        elif self.objectName() == "sld_pitch":
            euler_degree = Rotation.add_pitch(yaw, pitch, roll, theta)
        elif self.objectName() == "sld_roll":
            euler_degree = Rotation.add_roll(yaw, pitch, roll, theta)
        else:
            raise Exception("error")
        new_yaw, new_pitch, new_roll = euler_degree

        # print("yaw", theta, yaw, new_yaw)
        
        self.parent().sld_yaw.set_value(new_yaw)
        self.parent().sld_pitch.set_value(new_pitch)
        self.parent().sld_roll.set_value(new_roll)
        self.parent().config_changed()
        return

    def get_value(self):
        return self.slider.value()
    
    def set_value(self, value):
        value = round(value, 1)
        self.slider.setValue(value)
        return
    
    def config_changed(self):
        self.parent().config_changed()
        return
    

class DialThree(QWidget):
    def __init__(self, name, parent, dt_pose, tip, with_step=True, is_root=True, is_rhand=True):
        super(DialThree, self).__init__(parent)
        self.setObjectName(name)
        self.with_step = with_step
        self.dt_pose = dt_pose
        self.is_rhand = is_rhand
        self.step = 2
        
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.yaw_sign, self.pitch_sign, self.roll_sign = (1, 1, 1) if self.is_rhand else (-1, -1, 1)

        self.label_euler = QLabel(self)
        self.label_euler.setFrameStyle(QFrame.Sunken | QFrame.Panel)
        self.label_euler.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label_euler.setFont(QFont("Times", 8))
        self.label_euler.setToolTip(tip)

        self.dial_yaw = QDial(self)
        self.dial_pitch = QDial(self)
        self.dial_roll = QDial(self)
        
        if is_root and is_rhand:
            style = "background-color: rgb(160, 250, 160)"
        elif is_root and not is_rhand:
            style = "background-color: rgb(250, 250, 160)"
        elif not is_root and is_rhand:
            style = "background-color: rgb(220, 250, 220)"
        elif not is_root and not is_rhand:
            style = "background-color: rgb(250, 250, 220)"
        else:
            style = "background-color: rgb(250, 250, 250)"
        self.dial_yaw.setStyleSheet(style)
        self.dial_pitch.setStyleSheet(style)
        self.dial_roll.setStyleSheet(style)
        
        self.prev_dial_yaw = 0
        self.prev_dial_pitch = 0
        self.prev_dial_roll = 0
        
        self.yaw_add = self.create_button("+", self, "turn down(no color)", (25, 20), (1, 1))
        self.yaw_sub = self.create_button("-", self, "turn up(blue)", (25, 20), (1, 1))
        
        self.pitch_add = self.create_button("+", self, "turn right(red)", (25, 20), (1, 1))
        self.pitch_sub = self.create_button("-", self, "turn left(green)", (25, 20), (1, 1))
        
        self.roll_add = self.create_button("+", self, "clockwise", (25, 20), (1, 1))
        self.roll_sub = self.create_button("-", self, "counter-clockwise", (25, 20), (1, 1))
        
        if self.with_step:
            self.spin_step = QSpinBox(self)
            self.spin_step.setRange(1, 10)
            self.spin_step.setValue(2)
            self.spin_step.setWrapping(True)
    
        self.list_pose = self.create_combox("list_pose", self, self.dt_pose, self.tr("pose template"))
        
        self.signal_to_slot()
        self.set_layout()
        self.show_state()
        return
    
    @staticmethod
    def create_combox(name, parent, dt_pose, tip):
        obj = QComboBox(parent)
        obj.setObjectName(name)
        obj.setEditable(False)
        
        for key in dt_pose.keys():
            obj.addItem(key)
        obj.setCurrentIndex(0)
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setMinimumHeight(30)
        obj.setToolTip(tip)
        return obj
    
    @staticmethod
    def create_button(name, parent, tip, size, size_type):
        ls_policy = [QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding]
        obj = QPushButton(name, parent)
        obj.setToolTip(tip)
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setSizePolicy(ls_policy[size_type[0]], ls_policy[size_type[1]])
        obj.setMinimumSize(size[0], size[1])
        obj.setAutoRepeat(True)
        obj.setAutoRepeatDelay(300)
        obj.setAutoRepeatInterval(100)
        return obj
     
    def show_state(self):
        self.label_euler.setText("%+7.2f\n%+7.2f\n%+7.2f" %
                                 (self.yaw*self.yaw_sign, self.pitch*self.pitch_sign, self.roll*self.roll_sign))
        return
    
    def synchronized(self, is_dirty=True):
        self.show_state()
        self.parent().config_changed(is_dirty)
        return
    
    def select_pose(self, index):
        self.yaw, self.pitch, self.roll = self.dt_pose[str(index)]
        self.yaw, self.pitch, self.roll = self.yaw_sign*self.yaw, self.pitch_sign*self.pitch, self.roll_sign*self.roll
        self.synchronized()
        return
    
    def signal_to_slot(self):
        self.dial_yaw.sliderMoved.connect(lambda: self.change_yaw(False))
        self.dial_pitch.sliderMoved.connect(lambda: self.change_pitch(False))
        self.dial_roll.sliderMoved.connect(lambda: self.change_roll(False))
        self.dial_yaw.sliderReleased.connect(lambda: self.change_yaw(True))
        self.dial_pitch.sliderReleased.connect(lambda: self.change_pitch(True))
        self.dial_roll.sliderReleased.connect(lambda: self.change_roll(True))

        self.yaw_add.clicked.connect(lambda: self.add_euler_angle("yaw", 1*self.yaw_sign*self.step))
        self.yaw_sub.clicked.connect(lambda: self.add_euler_angle("yaw", -1*self.yaw_sign*self.step))

        self.pitch_add.clicked.connect(lambda: self.add_euler_angle("pitch", 1*self.pitch_sign*self.step))
        self.pitch_sub.clicked.connect(lambda: self.add_euler_angle("pitch", -1*self.pitch_sign*self.step))
        
        self.roll_add.clicked.connect(lambda: self.add_euler_angle("roll", 1*self.roll_sign*self.step))
        self.roll_sub.clicked.connect(lambda: self.add_euler_angle("roll", -1*self.roll_sign*self.step))
        
        if self.with_step:
            self.spin_step.valueChanged.connect(self.set_step)
        self.list_pose.activated[int].connect(self.select_pose)

        return
     
    def set_property(self):
        for obj in self.children():
            if isinstance(obj, QDial):
                obj.setContentsMargins(0, 0, 0, 0)
                obj.setWrapping(True)
                obj.setSingleStep(5)
                obj.setRange(0, 180)
                obj.setMinimumSize(70, 70)
                obj.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

            if isinstance(obj, QSpinBox):
                obj.setMinimumHeight(40)
                obj.setContentsMargins(0, 0, 0, 0)
        return
     
    def set_layout(self):
        self.set_property()
        
        vbox_yaw = QVBoxLayout()
        vbox_yaw.addWidget(self.yaw_add)
        vbox_yaw.addWidget(self.yaw_sub)
        
        vbox_pitch = QVBoxLayout()
        vbox_pitch.addWidget(self.pitch_add)
        vbox_pitch.addWidget(self.pitch_sub)
        
        vbox_roll = QVBoxLayout()
        vbox_roll.addWidget(self.roll_add)
        vbox_roll.addWidget(self.roll_sub)
        
        hbox_yaw = QHBoxLayout()
        hbox_yaw.setContentsMargins(0, 0, 0, 0)
        hbox_yaw.addWidget(self.dial_yaw, stretch=3)
        hbox_yaw.addLayout(vbox_yaw, stretch=1)
        
        hbox_pitch = QHBoxLayout()
        hbox_pitch.setContentsMargins(0, 0, 0, 0)
        hbox_pitch.addWidget(self.dial_pitch, stretch=3)
        hbox_pitch.addLayout(vbox_pitch, stretch=1)
        
        hbox_roll = QHBoxLayout()
        hbox_roll.setContentsMargins(0, 0, 0, 0)
        hbox_roll.addWidget(self.dial_roll, stretch=3)
        hbox_roll.addLayout(vbox_roll, stretch=1)
        
        hbox_all = QHBoxLayout()
        hbox_all.setContentsMargins(0, 0, 0, 0)
        hbox_all.setSpacing(0)
        hbox_all.addWidget(self.label_euler, stretch=6)
        hbox_all.addStretch(1)
        hbox_all.addLayout(hbox_yaw, stretch=10)
        hbox_all.addStretch(1)
        hbox_all.addLayout(hbox_pitch, stretch=10)
        hbox_all.addStretch(1)
        hbox_all.addLayout(hbox_roll, stretch=10)
        if self.with_step:
            hbox_all.addStretch(1)
            hbox_all.addWidget(self.spin_step, stretch=2)
        hbox_all.addSpacing(10)
        hbox_all.addWidget(self.list_pose, stretch=2)

        self.setLayout(hbox_all)
        
        return
     
    def set_step(self, value):
        self.step = value
        return
    
    def get_diff_angle(self, name):
        obj = getattr(self, f"dial_{name}")
        period = obj.maximum() - obj.minimum()
        
        prev_value = getattr(self, f"prev_dial_{name}")
        curr_value = obj.value()

        diff1 = (curr_value+period) - prev_value
        diff2 = curr_value - prev_value
        diff3 = curr_value - (prev_value+period)
        diff12 = diff1 if abs(diff1) < abs(diff2) else diff2
        diff = diff12 if abs(diff12) < abs(diff3) else diff3
        theta = diff*180/period
        
        setattr(self, f"prev_dial_{name}", curr_value)
        # print("diff_angle", prev_value, curr_value, diff1, diff2, diff3, theta)
        return theta
    
    def add_euler_angle(self, angle_name, theta, is_dirty=True):
        if angle_name.lower() in ["yaw"]:
            euler_degree = Rotation.add_yaw(self.yaw, self.pitch, self.roll, theta)
        elif angle_name.lower() in ["pitch"]:
            euler_degree = Rotation.add_pitch(self.yaw, self.pitch, self.roll, theta)
        elif angle_name.lower() in ["roll"]:
            euler_degree = Rotation.add_roll(self.yaw, self.pitch, self.roll, theta)
        else:
            raise Exception("unsupported angle")
        self.yaw, self.pitch, self.roll = euler_degree
        
        self.synchronized(is_dirty)
        return
    
    def change_yaw(self, is_dirty=True):
        name = "yaw"
        theta = self.get_diff_angle(name)
        self.add_euler_angle(name, self.yaw_sign*theta, is_dirty)
        return
    
    def change_pitch(self, is_dirty=True):
        name = "pitch"
        theta = self.get_diff_angle(name)
        self.add_euler_angle(name, self.pitch_sign*theta, is_dirty)
        return
    
    def change_roll(self, is_dirty=True):
        name = "roll"
        theta = self.get_diff_angle(name)
        self.add_euler_angle(name, self.roll_sign*theta, is_dirty)
        return
    
    
class CrossPose(QWidget):
    def __init__(self, name, parent, tip, step, pitch_scope, roll_scope):
        super(CrossPose, self).__init__(parent)
        self.setObjectName(name)
        self.pitch = 0
        self.roll = 0
        self.pitch_scope = pitch_scope
        self.roll_scope = roll_scope
        self.step = step
        self.setToolTip(tip)
        
        self.btn_add_pitch = self.create_button("+", parent, self.tr(f"add {name[:-1]}'s pitch"), (20, 20))
        self.btn_sub_pitch = self.create_button("-", parent, self.tr(f"sub {name[:-1]}'s pitch"), (20, 20))
        self.btn_add_roll = self.create_button("+", parent, self.tr(f"add {name[:-1]}'s roll"), (20, 20))
        self.btn_sub_roll = self.create_button("-", parent, self.tr(f"sub {name[:-1]}'s roll"), (20, 20))
         
        self.label_pitch = self.create_label("label_pitch", self.tr(f"{name[:-1]}'s horizontal vibration"))
        self.label_roll = self.create_label("label_roll", self.tr(f"{name[:-1]}'s rolling"))
        self.frame_box = QFrame(self)
        self.frame_box.setContentsMargins(0, 0, 0, 0)
        self.frame_box.setFrameStyle(QFrame.Sunken | QFrame.Panel)

        self.color_map = plt.get_cmap("plasma")

        self.signal_to_slot()
        self.set_layout()
        self.show_state()
        return
    
    def create_label(self, name, tip):
        obj = QLabel(self)
        obj.setObjectName(name)
        obj.setToolTip(tip)
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setMinimumSize(30, 18)
        obj.setTextInteractionFlags(Qt.TextSelectableByMouse)
        obj.setFont(QFont("Times", 9, 600))
        obj.setAlignment(QtCore.Qt.AlignCenter)
        return obj

    def show_state(self):
        pitch_val = (self.pitch-self.pitch_scope[0])/(self.pitch_scope[1]-self.pitch_scope[0])
        color = (np.array(self.color_map(pitch_val*0.9)[:3]) * 255).astype(np.uint8).tolist()
        self.label_pitch.setStyleSheet(f"color: rgb({color[0]}, {color[1]}, {color[2]}); padding:0; border: 0px")
        self.label_pitch.setText("%+3d" % (self.pitch))
        
        roll_val = (self.roll-self.roll_scope[0])/(self.roll_scope[1]-self.roll_scope[0])
        color = (np.array(self.color_map(roll_val*0.9)[:3]) * 255).astype(np.uint8).tolist()
        self.label_roll.setStyleSheet(f"color: rgb({color[0]}, {color[1]}, {color[2]}); padding:0; border: 0px")
        self.label_roll.setText("%+3d" % (self.roll))

        return

    def synchronized(self):
        self.show_state()
        self.parent().config_changed()
        return
    
    def set_step(self, value):
        self.step = value
        return
    
    def reset(self):
        self.pitch, self.roll = 0, 0
        self.synchronized()
        return
    
    @staticmethod
    def create_button(name, parent, tip, size):
        obj = QPushButton(name, parent)
        obj.setToolTip(tip)
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        obj.setMinimumSize(size[0], size[1])
        obj.setAutoRepeat(True)
        obj.setAutoRepeatDelay(500)
        obj.setAutoRepeatInterval(100)
        obj.setStyleSheet("border-style:outset|solid")
        return obj

    def signal_to_slot(self):
        self.btn_add_pitch.clicked.connect(lambda: self.add_angle("pitch", 1*self.step))
        self.btn_sub_pitch.clicked.connect(lambda: self.add_angle("pitch", -1*self.step))
        
        self.btn_add_roll.clicked.connect(lambda: self.add_angle("roll", 1*self.step))
        self.btn_sub_roll.clicked.connect(lambda: self.add_angle("roll", -1*self.step))
        
        return
    
    def set_layout(self):
        vbox_pitch = QVBoxLayout()
        vbox_pitch.setSpacing(0)
        vbox_pitch.setContentsMargins(0, 0, 0, 0)
        vbox_pitch.addWidget(self.btn_add_pitch)
        vbox_pitch.addWidget(self.btn_sub_pitch)
        vbox_pitch.addStretch()

        vbox_roll = QVBoxLayout()
        vbox_roll.setSpacing(0)
        vbox_roll.setContentsMargins(0, 0, 0, 0)
        vbox_roll.addWidget(self.btn_add_roll)
        vbox_roll.addWidget(self.btn_sub_roll)
        vbox_roll.addStretch()
        
        vbox_label = QVBoxLayout()
        vbox_label.setSpacing(0)
        vbox_label.setContentsMargins(0, 0, 0, 0)
        vbox_label.addWidget(self.label_pitch)
        vbox_label.addWidget(self.label_roll)
        self.frame_box.setLayout(vbox_label)

        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.btn_sub_pitch, stretch=4)
        hbox.addWidget(self.btn_add_pitch, stretch=4)
        hbox.addWidget(self.frame_box, stretch=6)
        hbox.addLayout(vbox_roll, stretch=3)
        hbox.setSpacing(0)
        hbox.addStretch()
        
        self.setLayout(hbox)

        return

    def add_angle(self, angle_name, theta):
        if angle_name.lower() in ["pitch"]:
            self.pitch += theta
        elif angle_name.lower() in ["roll"]:
            self.roll += theta
        else:
            raise Exception("unsupported angle")
         
        self.pitch = max(self.pitch_scope[0], min(self.pitch_scope[1], self.pitch))
        self.roll = max(self.roll_scope[0], min(self.roll_scope[1], self.roll))
        
        self.synchronized()
        return

                
class HandBetaParam(object):
    def __init__(self):
        self.sld_beta0, self.sld_beta1, self.sld_beta2, self.sld_beta3, self.sld_beta4 = 0, 0, 0, 0, 0
        self.sld_beta5, self.sld_beta6, self.sld_beta7, self.sld_beta8, self.sld_beta9 = 0, 0, 0, 0, 0
        self.rot_thumb = self.rot_index = self.rot_middle = self.rot_ring = self.rot_pinky = np.eye(3).tolist()
        self.ori_root_x, self.ori_root_y, self.ori_root_z = 0, 0, 0
        
        self.ori_thumb_root_x, self.ori_thumb_root_y, self.ori_thumb_root_z = 0, 0, 0
        self.ori_index_root_x, self.ori_index_root_y, self.ori_index_root_z = 0, 0, 0
        self.ori_middle_root_x, self.ori_middle_root_y, self.ori_middle_root_z = 0, 0, 0
        self.ori_ring_root_x, self.ori_ring_root_y, self.ori_ring_root_z = 0, 0, 0
        self.ori_pinky_root_x, self.ori_pinky_root_y, self.ori_pinky_root_z = 0, 0, 0
        return

    def from_dict(self, dt):
        # only update existing key
        self.__dict__.update((k, dt[k]) for k in set(dt).intersection(self.__dict__))
        return

    def to_dict(self):
        return self.__dict__
    
    def to_mano_param(self):
        beta = self.get_beta()
        recify_rot = np.array([self.rot_thumb, self.rot_index, self.rot_middle, self.rot_ring, self.rot_pinky])
        ori_root_pos = np.array([self.ori_root_x, self.ori_root_y, self.ori_root_z])
        
        ori_finger_root_pos = np.array([[self.ori_thumb_root_x, self.ori_thumb_root_y, self.ori_thumb_root_z],
                                        [self.ori_index_root_x, self.ori_index_root_y, self.ori_index_root_z],
                                        [self.ori_middle_root_x, self.ori_middle_root_y, self.ori_middle_root_z],
                                        [self.ori_ring_root_x, self.ori_ring_root_y, self.ori_ring_root_z],
                                        [self.ori_pinky_root_x, self.ori_pinky_root_y, self.ori_pinky_root_z]])
        
        return beta, recify_rot, ori_root_pos, ori_finger_root_pos

    def get_beta(self):
        beta = np.array([self.sld_beta0, self.sld_beta1, self.sld_beta2, self.sld_beta3, self.sld_beta4,
                         self.sld_beta5, self.sld_beta6, self.sld_beta7, self.sld_beta8, self.sld_beta9])
        beta = beta / 100
        return beta

    def save_recify_rot(self, recify_rot):
        self.rot_thumb = recify_rot[0].tolist()
        self.rot_index = recify_rot[1].tolist()
        self.rot_middle = recify_rot[2].tolist()
        self.rot_ring = recify_rot[3].tolist()
        self.rot_pinky = recify_rot[4].tolist()
        return
    
    def save_root_pos(self, pos):
        self.ori_root_x, self.ori_root_y, self.ori_root_z = pos[0], pos[1], pos[2]
        return
    
    def save_finger_root_pos(self, arr_pos):
        self.ori_thumb_root_x, self.ori_thumb_root_y, self.ori_thumb_root_z = arr_pos[0]
        self.ori_index_root_x, self.ori_index_root_y, self.ori_index_root_z = arr_pos[1]
        self.ori_middle_root_x, self.ori_middle_root_y, self.ori_middle_root_z = arr_pos[2]
        self.ori_ring_root_x, self.ori_ring_root_y, self.ori_ring_root_z = arr_pos[3]
        self.ori_pinky_root_x, self.ori_pinky_root_y, self.ori_pinky_root_z = arr_pos[4]
        return
    

class HandBetaWidget(QWidget):
    sig_config_changed = QtCore.Signal(bool)
    
    def __init__(self, name, parent, mano):
        super(HandBetaWidget, self).__init__(parent)
        self.setObjectName(name)
        self.param = HandBetaParam()
        self.mano = mano
        
        dt_sld_beta03 = {"sld_beta0": (-500, 500, 4, 5), "sld_beta1": (-500, 500, 4, 5), "sld_beta2": (-500, 500, 4, 5)}
        hbox_sld_beta03 = self.create_slides("beta03", dt_sld_beta03)
        dt_sld_beta36 = {"sld_beta3": (-500, 500, 4, 5), "sld_beta4": (-500, 500, 4, 5), "sld_beta5": (-500, 500, 4, 5)}
        hbox_sld_beta36 = self.create_slides("beta36", dt_sld_beta36)
        dt_sld_beta69 = {"sld_beta6": (-500, 500, 4, 5), "sld_beta7": (-500, 500, 4, 5), "sld_beta8": (-500, 500, 4, 5)}
        hbox_sld_beta69 = self.create_slides("beta69", dt_sld_beta69)
        dt_sld_beta9 = {"sld_beta9": (-500, 500, 4, 5)}
        hbox_sld_beta9 = self.create_slides("beta9", dt_sld_beta9)

        self.beta_text = QLabel(self)
        self.beta_text.setFrameStyle(QFrame.Sunken | QFrame.Panel)
        self.beta_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        self.beta_list = self.create_beta_combo()
        hbox_sld_beta9.addWidget(self.beta_list)
        hbox_sld_beta9.addStretch()

        vbox = QVBoxLayout()
        vbox.setSpacing(5)
        vbox.addLayout(hbox_sld_beta03)
        vbox.addSpacing(5)
        vbox.addLayout(hbox_sld_beta36)
        vbox.addSpacing(5)
        vbox.addLayout(hbox_sld_beta69)
        vbox.addSpacing(5)
        vbox.addLayout(hbox_sld_beta9)
        vbox.addSpacing(5)
        vbox.addLayout(hbox_sld_beta9)
        vbox.addSpacing(5)
        vbox.addWidget(self.beta_text)
        vbox.addStretch(1)
        
        self.setLayout(vbox)
        self.beta_list.setCurrentIndex(0)
        
        self.save_param()
        return
     
    def create_beta_combo(self):
        if self.mano.is_rhand:
            self.dt_config = \
                {"cfg0": [-2.5099, 00.1018, -0.6341, 00.0132, 00.2387, -0.1313, 00.1155, 00.2773, -0.1189, -0.2002],
                 "cfg1": [00.3191, 00.0123, -0.4653, -0.0662, -0.0788, 00.0861, 00.0812, 00.0859, 00.0574, 00.2454],
                 "cfg2": [00.6551, 00.2005, -0.3413, 00.2005, -0.0111, -0.1472, 00.0255, 00.0108, 00.0156, 00.1717],
                 "cfg3": [00.6374, 00.0120, -0.2895, -0.0887, -0.1990, -0.3174, -0.2346, 00.0229, 00.0762, 00.0827],
                 "cfg4": [00.0972, 00.0192, -0.2883, -0.0807, -0.1831, -0.1927, -0.0469, 00.1148, -0.0800, 00.0140],
                 "cfg5": [00.1351, -0.0271, -0.0814, -0.1495, -0.0436, -0.1034, -0.0676, 00.0772, 00.0181, 00.1754],
                 "cfg6": [-0.4337, -0.1868, -0.3990, -0.0689, 00.0272, 00.0397, 00.0991, 00.0493, -0.0492, 00.0821]}
        else:
            self.dt_config = \
                {"cfg0": [-2.5099, 00.1018, -0.6341, 00.0132, 00.2387, -0.1313, 00.1155, 00.2773, -0.1189, -0.2002],
                 "cfg1": [00.4168, -0.1520, -0.2788, -0.0768, -0.1167, 00.0338, 00.0651, 00.0879, 00.0825, 00.1934],
                 "cfg2": [01.5070, 00.3057, -0.2899, 00.1919, -0.1978, -0.1859, -0.0483, 00.0473, 00.0898, 00.3221],
                 "cfg3": [00.5335, 00.1420, -0.0376, -0.0960, -0.1564, -0.1390, -0.1370, 00.2291, 00.1402, 00.1139],
                 "cfg4": [-0.8518, 00.0002, -0.4116, 00.1125, 00.2512, -0.0699, 00.0654, 00.0989, -0.0623, -0.0265],
                 "cfg5": [-0.1870, -0.1241, -0.0565, 00.0404, 00.1142, -0.1634, -0.0607, -0.0514, 00.0420, 00.0692],
                 "cfg6": [00.0238, -0.0024, -0.3567, -0.0006, -0.0614, 00.0070, 00.0493, 00.0772, -0.0247, 00.1267]}

        beta_list = QComboBox(self)
        for key, value in self.dt_config.items():
            beta_list.addItem(key)
        
        beta_list.setEditable(False)
        beta_list.currentTextChanged[str].connect(self.combo_changed)

        return beta_list
    
    def combo_changed(self, text):
        betas = self.dt_config[text]
        for ii, val in enumerate(betas):
            self.findChild(QtWidgets.QSlider, f"sld_beta{ii}").setValue(int(val*100))
        self.config_changed()
        return

    def create_slides(self, label_name, dt_slide_info):
        hbox = QHBoxLayout()
        hbox.setSpacing(10)
        hbox.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(self)
        self.label.setFixedSize(40, 25)
        self.label.setText(label_name)
        hbox.addWidget(self.label)

        for name, scope in dt_slide_info.items():
            sld = SliderPlus(name, self, scope[0], scope[1], scope[2], scope[3])
            hbox.addWidget(sld, stretch=abs(scope[1] - scope[0]))
        return hbox
    
    def set_beta_text(self):
        ls_text = []
        for ii in range(10):
            value = getattr(self.param, f"sld_beta{ii}")
            ls_text.append("%+5.2f" % (value/100))
        self.beta_text.setText(",".join(ls_text))
        return
    
    def display_param(self):
        for key, value in self.param.__dict__.items():
            if key.find("sld") >= 0:
                obj = self.findChild(QtWidgets.QSlider, key)
                if obj is not None:
                    obj.setValue(int(value))
            else:
                obj = self.findChild(QtWidgets.QLineEdit, key)
                if obj is not None:
                    obj.setText(str(value))
            # print("hand beta display", key, value, obj is not None)

        recify_rot, ori_root_pos, ori_finger_root_pos = self.mano.calc_recify_rotation(self.param.get_beta())
        # print(local_time(), "display beta param", ori_root_pos)
        self.param.save_recify_rot(recify_rot)
        self.param.save_root_pos(ori_root_pos.tolist())
        self.param.save_finger_root_pos(ori_finger_root_pos.tolist())
        self.set_beta_text()
        return
    
    def save_param(self):
        for key, value in self.param.__dict__.items():
            if key.find("sld") >= 0:
                obj = self.findChild(QtWidgets.QSlider, key)
                if obj is not None:
                    setattr(self.param, key, obj.value())
            else:
                obj = self.findChild(QtWidgets.QLineEdit, key)
                if obj is not None:
                    setattr(self.param, key, int(obj.text()))
            # print("hand beta save", key, value, obj is None)
        
        recify_rot, ori_root_pos, ori_finger_root_pos = self.mano.calc_recify_rotation(self.param.get_beta())
        print(local_time(), "save beta param", ori_root_pos)
        self.param.save_recify_rot(recify_rot)
        self.param.save_root_pos(ori_root_pos.tolist())
        self.param.save_finger_root_pos(ori_finger_root_pos.tolist())

        self.set_beta_text()
        return
    
    def config_changed(self, is_dirty=True):
        self.save_param()
        self.sig_config_changed.emit(is_dirty)
        return


class HandGlobalParam(object):
    def __init__(self):
        self.spin_focal_x, self.spin_focal_y = 0, 0
        self.spin_princpt_x, self.spin_princpt_y = 0, 0
        self.spin_left, self.spin_right = 0, 0
        self.spin_top, self.spin_bottom = 0, 0
        self.image_width, self.image_height = 0, 0
        return

    def from_dict(self, dt):
        # only update existing key
        self.__dict__.update((k, dt[k]) for k in set(dt).intersection(self.__dict__))
        return

    def to_dict(self):
        return self.__dict__
    
    def get_focal_princpt(self):
        focal = np.array([self.spin_focal_x, self.spin_focal_y])
        princpt = np.array([self.spin_princpt_x, self.spin_princpt_y])
        return focal, princpt

    def to_mano_param(self):
        focal = np.array([self.spin_focal_x, self.spin_focal_y])
        princpt = np.array([self.spin_princpt_x, self.spin_princpt_y])
        rect = np.array([self.spin_left, self.spin_top, self.spin_right, self.spin_bottom])
        return focal, princpt, rect
    
    def set_roi(self, left, top, right, bottom):
        self.spin_left = left
        self.spin_top = top
        self.spin_right = right
        self.spin_bottom = bottom
        return
     
    def get_roi_size(self):
        width, height = self.spin_right-self.spin_left, self.spin_bottom-self.spin_top
        return width, height
    
    def get_roi_rect(self):
        rect = self.spin_left, self.spin_top, self.spin_right, self.spin_bottom
        return rect
    
    def get_roi_info(self):
        rect = self.spin_left, self.spin_top, self.spin_right, self.spin_bottom
        size = self.image_width, self.image_height
        return rect, size
    
    def get_roi_rect_ratio(self):
        rect = self.spin_left/self.image_width, self.spin_top/self.image_height,\
               self.spin_right/self.image_width, self.spin_bottom/self.image_height
        return rect


class HandGlobalWidget(QWidget):
    sig_config_changed = QtCore.Signal(bool)
    sig_restore_template = QtCore.Signal(str)
    sig_3d_view = QtCore.Signal()

    def __init__(self, name, parent):
        super(HandGlobalWidget, self).__init__(parent)
        self.setObjectName(name)
        self.param = HandGlobalParam()

        self.spin_focal_x = self.create_spin("spin_focal_x", self, 1000, None, 1, "focal length x")
        self.spin_focal_y = self.create_spin("spin_focal_y", self, 1000, None, 1, "focal length y")
        self.spin_princpt_x = self.create_spin("spin_princpt_x", self, 200, None, 1, "princpt x")
        self.spin_princpt_y = self.create_spin("spin_princpt_y", self, 200, None, 1, "princpt y")
        self.list_template = self.create_template_combo()

        self.spin_left = self.create_spin("spin_left", self, 0, None, 1, self.tr("left of roi box"))
        self.spin_right = self.create_spin("spin_right", self, 512, None, 1, self.tr("right of roi box"))
        self.spin_top = self.create_spin("spin_top", self, 0, None, 1, self.tr("top of roi box"))
        self.spin_bottom = self.create_spin("spin_bottom", self, 512, None, 1, self.tr("bottom of roi box"))
        self.btn_reset_roi = self.create_button("X", self, self.tr("reset roi"), (30, 30), (1, 0))
        
        self.spin_step = self.create_spin("spin_step", self, 2, (2, 10), 2, self.tr("step of adding or subtracting"))
        self.btn_3d = self.create_button("3D", self, self.tr("open 3D view"), (30, 30), (1, 0))
        
        self.set_layout()
        self.signal_to_slot()
        self.set_step(self.spin_step.value())
        
        self.save_param()
        return

    @staticmethod
    def create_button(name, parent, tip, size, size_type):
        ls_policy = [QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding]
        obj = QPushButton(name, parent)
        obj.setToolTip(tip)
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setSizePolicy(ls_policy[size_type[0]], ls_policy[size_type[1]])
        obj.setMinimumSize(size[0], size[1])
        return obj
    
    @staticmethod
    def create_spin(name, parent, init_val, scope, step, tip):
        obj = QSpinBox(parent)
        if scope is not None:
            obj.setRange(scope[0], scope[1])
        else:
            obj.setRange(-9999, 9999)
        obj.setObjectName(name)
        obj.setValue(init_val)
        obj.setSingleStep(step)
        obj.setToolTip(tip)
        obj.setWrapping(True)
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setMinimumSize(50, 30)
        return obj
     
    def set_step(self, value):
        for obj in self.children():
            name = obj.objectName()
            if name.find("spin") >= 0 and name != "spin_step":
                obj.setSingleStep(value)
        return
    
    def config_changed(self, is_dirty=True):
        self.save_param()
        self.sig_config_changed.emit(is_dirty)
        return
    
    def set_layout(self):
        hbox_focal = QHBoxLayout()
        hbox_focal.setContentsMargins(0, 0, 0, 0)
        hbox_focal.setSpacing(0)
        hbox_focal.addWidget(self.spin_focal_x, stretch=2)
        hbox_focal.addStretch(1)
        hbox_focal.addWidget(self.spin_focal_y, stretch=2)
        hbox_focal.addStretch(1)
        hbox_focal.addWidget(self.spin_step, stretch=1)
        hbox_focal.addStretch(1)
        hbox_focal.addWidget(self.btn_3d, stretch=1)
        hbox_focal.addStretch(1)

        hbox_princpt = QHBoxLayout()
        hbox_princpt.setContentsMargins(0, 0, 0, 0)
        hbox_princpt.setSpacing(0)
        hbox_princpt.addWidget(self.spin_princpt_x, stretch=2)
        hbox_princpt.addStretch(1)
        hbox_princpt.addWidget(self.spin_princpt_y, stretch=2)
        hbox_princpt.addStretch(1)
        hbox_princpt.addWidget(self.list_template, stretch=1)
        hbox_princpt.addStretch(1)

        hbox_roi = QHBoxLayout()
        hbox_roi.setContentsMargins(0, 0, 0, 0)
        hbox_roi.setSpacing(0)
        hbox_roi.addWidget(self.spin_left, stretch=4)
        hbox_roi.addStretch(1)
        hbox_roi.addWidget(self.spin_top, stretch=4)
        hbox_roi.addStretch(1)
        hbox_roi.addWidget(self.spin_right, stretch=4)
        hbox_roi.addStretch(1)
        hbox_roi.addWidget(self.spin_bottom, stretch=4)
        hbox_roi.addStretch(1)
        hbox_roi.addWidget(self.btn_reset_roi, stretch=2)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addLayout(hbox_focal, stretch=1)
        vbox.addLayout(hbox_princpt, stretch=1)
        vbox.addLayout(hbox_roi, stretch=1)

        self.setLayout(vbox)
        return

    def create_template_combo(self):
        template_list = QComboBox(self)
        template_list.setEditable(False)

        # dir_path = os.path.dirname(os.path.abspath(__file__))
        # self.template_dir_path = os.path.join(dir_path, "..", "template")
        self.template_dir_path = os.path.join(os.path.abspath(os.path.curdir), "Template")

        ls_name = []
        for path in glob.glob(os.path.join(self.template_dir_path, "*.json")):
            name = os.path.basename(path).replace(".json", "")
            ls_name.append(name)
            
        ls_name = sorted(ls_name, key=lambda x: int(x.split("_")[0]))
        for name in ls_name:
            template_list.addItem(str(name))

        template_list.setCurrentIndex(0)
        template_list.activated[int].connect(self.restore_template)
        template_list.setContentsMargins(0, 0, 0, 0)
        template_list.setMinimumHeight(30)

        return template_list

    def restore_template(self, index):
        path = os.path.join(self.template_dir_path, self.list_template.itemText(index)+".json")
        self.sig_restore_template.emit(path)
        return
    
    def signal_to_slot(self):
        self.btn_reset_roi.clicked.connect(self.reset_roi)
        self.spin_step.valueChanged.connect(self.set_step)
        self.spin_focal_x.valueChanged.connect(lambda: self.config_changed(True))
        self.spin_focal_y.valueChanged.connect(lambda: self.config_changed(True))
        self.spin_princpt_x.valueChanged.connect(lambda: self.config_changed(True))
        self.spin_princpt_y.valueChanged.connect(lambda: self.config_changed(True))
        self.spin_left.valueChanged.connect(lambda: self.config_changed(True))
        self.spin_top.valueChanged.connect(lambda: self.config_changed(True))
        self.spin_right.valueChanged.connect(lambda: self.config_changed(True))
        self.spin_bottom.valueChanged.connect(lambda: self.config_changed(True))
        self.btn_3d.clicked.connect(lambda: self.sig_3d_view.emit())
        return
    
    def set_image_size(self, width, height):
        self.param.image_width, self.param.image_height = width, height
        return
    
    def reset_roi(self):
        self.set_roi(0, 0, self.param.image_width, self.param.image_height)
        self.config_changed()
        return
    
    def set_roi(self, left, top, right, bottom):
        left = max(0, min(left, self.param.image_width))
        top = max(0, min(top, self.param.image_height))
        right = max(0, min(right, self.param.image_width))
        bottom = max(0, min(bottom, self.param.image_height))
        
        dt = {"spin_left": left, "spin_top": top, "spin_right": right, "spin_bottom": bottom}
        for key, value in dt.items():
            obj = self.findChild(QtWidgets.QSpinBox, key)
            state = obj.blockSignals(True)
            obj.setValue(value)
            obj.blockSignals(state)
        return
    
    def display_param(self):
        for key, value in self.param.__dict__.items():
            if key.find("spin") >= 0:
                obj = self.findChild(QtWidgets.QSpinBox, key)
                if obj is not None:
                    state = obj.blockSignals(True)
                    obj.setValue(int(value))
                    obj.blockSignals(state)
            else:
                pass
            # print("hand global display", key, value, obj is not None)
        return

    def save_param(self):
        for key, value in self.param.__dict__.items():
            if key.find("spin") >= 0:
                obj = self.findChild(QtWidgets.QSpinBox, key)
                if obj is not None:
                    setattr(self.param, key, obj.value())
            else:
                pass
            # print("hand label save", key, value, obj is None)
        return
    
    
class Keypoint(QWidget):
    def __init__(self, name, parent, step=1, tip=""):
        super(Keypoint, self).__init__(parent)
        self.setObjectName(name)
        self.step = step
        self.setToolTip(tip)
        
        self.coord = QPushButton("coord", parent)
        self.coord.setContentsMargins(0, 0, 0, 0)
        self.coord.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.coord.setMinimumSize(30, 40)

        self.active = False
        self.x = 0
        self.y = 0
        self.step = step
        self.timer = QtCore.QElapsedTimer()
        
        self.btn_add_x = self.create_button("+", self, "add x")
        self.btn_sub_x = self.create_button("-", self, "sub x")
        self.btn_add_y = self.create_button("+", self, "add y")
        self.btn_sub_y = self.create_button("-", self, "sub y")
        
        self.set_layout()
        
        self.signal_to_slot()
        
        self.show_state()
        return

    @staticmethod
    def create_button(name, parent, tip):
        obj = QPushButton(name, parent)
        obj.setToolTip(tip)
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        obj.setMinimumSize(15, 18)
        return obj

    def signal_to_slot(self):
        self.coord.clicked.connect(self.toggle_state)
        self.coord.pressed.connect(self.coord_press)
        self.coord.released.connect(self.coord_release)
        self.btn_add_x.clicked.connect(lambda: self.add_value(True, 1 * self.step))
        self.btn_sub_x.clicked.connect(lambda: self.add_value(True, -1 * self.step))
        self.btn_add_y.clicked.connect(lambda: self.add_value(False, -1 * self.step))  # reverse y-axis
        self.btn_sub_y.clicked.connect(lambda: self.add_value(False, 1 * self.step))  # reverse y-axis
        return
    
    def toggle_state(self):
        self.active = not self.active
        if self.active:
            self.parent().set_exclusive_state(self)
        self.synchronized()
        return
    
    def set_active(self):
        self.active = True
        self.show_state()
        return
     
    def set_inactive(self):
        self.active = False
        self.show_state()
        return
    
    def is_active(self):
        return self.active
    
    def synchronized(self):
        self.show_state()
        self.parent().config_changed()
        return
    
    def set_xy(self, x, y):
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        self.show_state()
        return
    
    def get_distance(self, x, y):
        return abs(self.x - x) + abs(self.y - y)
    
    def add_value(self, is_x, value):
        if is_x and self.x > 0:
            self.x += value
        if not is_x and self.y > 0:
            self.y += value
            
        self.x = max(0, self.x)
        self.y = max(0, self.y)
        
        self.synchronized()
        return

    def coord_press(self):
        self.timer.start()
        return

    def coord_release(self):
        if self.timer.elapsed() > 1000:
            self.x, self.y = 0, 0
        return

    def show_state(self):
        is_rhand = self.parent().objectName().lower().find("right") >= 0
        active_style = "border: 2px solid; border-color:green; padding:2; border-radius:2px; font-size:7pt;" \
                       "background-color: rgb(200, 250, 200); color:rgb(0, 0, 0)"
        inactive_style = "border: 2px solid; border-color:grey; padding:2; border-radius:2px;font-size:7pt;" \
                         "background-color:white; color:rgb(0, 0, 0)"
        if not is_rhand:
            active_style = active_style.replace("green", "yellow").replace("rgb(200, 250, 200)", "rgb(250, 250, 200)")
        if self.x > 0 and self.y > 0:
            active_style = active_style.replace("rgb(0, 0, 0)", "#ca6702")
            inactive_style = inactive_style.replace("rgb(0, 0, 0)", "#ca6702")

        if self.active:
            self.coord.setStyleSheet(active_style)
        else:
            self.coord.setStyleSheet(inactive_style)
            
        self.coord.setText("%03d\n%03d" % (self.x, self.y))
        return
    
    def set_layout(self):
        vbox_x = QVBoxLayout()
        vbox_x.setSpacing(0)
        vbox_x.setContentsMargins(0, 0, 0, 0)
        vbox_x.addWidget(self.btn_add_x)
        vbox_x.addWidget(self.btn_sub_x)
        vbox_x.addStretch()

        vbox_y = QVBoxLayout()
        vbox_y.setSpacing(0)
        vbox_y.setContentsMargins(0, 0, 0, 0)
        vbox_y.addWidget(self.btn_add_y)
        vbox_y.addWidget(self.btn_sub_y)
        vbox_y.addStretch()
        
        hbox = QHBoxLayout()
        hbox.setSpacing(0)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addLayout(vbox_x, stretch=1)
        hbox.addWidget(self.coord, stretch=2)
        hbox.addLayout(vbox_y, stretch=1)
        hbox.addStretch(stretch=20)
        
        self.setLayout(hbox)
        return
    

class HandKeypointParam(object):
    def __init__(self):
        self.kp_x00 = 0
        self.kp_x01, self.kp_x02, self.kp_x03, self.kp_x04 = 0, 0, 0, 0
        self.kp_x05, self.kp_x06, self.kp_x07, self.kp_x08 = 0, 0, 0, 0
        self.kp_x09, self.kp_x10, self.kp_x11, self.kp_x12 = 0, 0, 0, 0
        self.kp_x13, self.kp_x14, self.kp_x15, self.kp_x16 = 0, 0, 0, 0
        self.kp_x17, self.kp_x18, self.kp_x19, self.kp_x20 = 0, 0, 0, 0

        self.kp_y00 = 0
        self.kp_y01, self.kp_y02, self.kp_y03, self.kp_y04 = 0, 0, 0, 0
        self.kp_y05, self.kp_y06, self.kp_y07, self.kp_y08 = 0, 0, 0, 0
        self.kp_y09, self.kp_y10, self.kp_y11, self.kp_y12 = 0, 0, 0, 0
        self.kp_y13, self.kp_y14, self.kp_y15, self.kp_y16 = 0, 0, 0, 0
        self.kp_y17, self.kp_y18, self.kp_y19, self.kp_y20 = 0, 0, 0, 0
        return

    def from_dict(self, dt):
        # only update existing key
        self.__dict__.update((k, dt[k]) for k in set(dt).intersection(self.__dict__))
        return

    def to_dict(self):
        return self.__dict__


class HandKeypointWidget(QWidget):
    sig_config_changed = QtCore.Signal(bool)
    
    def __init__(self, name, parent):
        super(HandKeypointWidget, self).__init__(parent)
        self.setObjectName(name)
        self.param = HandKeypointParam()

        self.spin_step = HandGlobalWidget.create_spin("spin_step", self, 1, (1, 5), 1, "step of adding or subtracting")
        self.cb_show = QCheckBox("显示", self)
        self.cb_show.toggled.connect(lambda: self.config_changed(True))
        self.cb_show.setChecked(True)

        ls_info = [("kp_00", self, "腕关节")]
        hbox_root = self.create_keypoints(ls_info)
        
        hbox_root.addStretch(stretch=1)
        hbox_root.addWidget(self.cb_show)
        hbox_root.addStretch(stretch=1)
        hbox_root.addWidget(self.spin_step)
        hbox_root.addStretch(stretch=10)
        
        ls_info = [("kp_01", self, "拇指1"), ("kp_02", self, "拇指2"), ("kp_03", self, "拇指3"), ("kp_04", self, "拇指4")]
        hbox_thumb = self.create_keypoints(ls_info)
        
        ls_info = [("kp_05", self, "食指1"), ("kp_06", self, "食指2"), ("kp_07", self, "食指3"), ("kp_08", self, "食指4")]
        hbox_index = self.create_keypoints(ls_info)
        
        ls_info = [("kp_09", self, "中指1"), ("kp_10", self, "中指2"), ("kp_11", self, "中指3"), ("kp_12", self, "中指4")]
        hbox_middle = self.create_keypoints(ls_info)
        
        ls_info = [("kp_13", self, "无名指1"), ("kp_14", self, "无名指2"), ("kp_15", self, "无名指3"), ("kp_16", self, "无名指4")]
        hbox_ring = self.create_keypoints(ls_info)
        
        ls_info = [("kp_17", self, "小指1"), ("kp_18", self, "小指2"), ("kp_19", self, "小指3"), ("kp_20", self, "小指4")]
        hbox_pinky = self.create_keypoints(ls_info)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_root, stretch=2)
        vbox.addStretch(stretch=1)
        vbox.addLayout(hbox_thumb, stretch=2)
        vbox.addStretch(stretch=1)
        vbox.addLayout(hbox_index, stretch=2)
        vbox.addStretch(stretch=1)
        vbox.addLayout(hbox_middle, stretch=2)
        vbox.addStretch(stretch=1)
        vbox.addLayout(hbox_ring, stretch=2)
        vbox.addStretch(stretch=1)
        vbox.addLayout(hbox_pinky, stretch=2)
        
        self.setLayout(vbox)
        self.kp_00.toggle_state()
        
        self.save_param()
        return
    
    def create_keypoints(self, infos):
        hbox = QHBoxLayout()
        hbox.setSpacing(2)
        hbox.setContentsMargins(0, 0, 0, 0)
        for name, parent, tip in infos:
            obj = Keypoint(name, parent, 1, tip)
            setattr(self, name, obj)
            hbox.addWidget(obj, stretch=2)
        return hbox
    
    def display_param(self):
        for key, value in self.param.__dict__.items():
            if key.find("kp_x") >= 0:
                obj = self.findChild(Keypoint, key.replace("kp_x", "kp_"))
                if obj is not None:
                    obj.set_xy(value, None)
            elif key.find("kp_y") >= 0:
                obj = self.findChild(Keypoint, key.replace("kp_y", "kp_"))
                if obj is not None:
                    obj.set_xy(None, value)
            pass
            # print("hand kp display", key, value, obj is not None)
        return

    def save_param(self):
        for key, value in self.param.__dict__.items():
            if key.find("kp_x") >= 0:
                obj = self.findChild(Keypoint, key.replace("kp_x", "kp_"))
                if obj is not None:
                    setattr(self.param, key, obj.x)
            elif key.find("kp_y") >= 0:
                obj = self.findChild(Keypoint, key.replace("kp_y", "kp_"))
                if obj is not None:
                    setattr(self.param, key, obj.y)
            pass
        return

    def config_changed(self, is_dirty=True):
        self.save_param()
        self.sig_config_changed.emit(is_dirty)
        return
    
    def set_exclusive_state(self, obj):
        for child in self.findChildren(Keypoint):
            if child != obj:
                child.set_inactive()
        return
    
    def set_keypoint(self, x, y):
        for child in self.findChildren(Keypoint):
            if child.is_active():
                child.set_xy(x, y)
                child.set_inactive()
                self.set_next_active(child.objectName())
                break
        return
    
    def set_next_active(self, name):
        idx = int(name.replace("kp_", ""))
        idx = (idx + 1) % 21
        obj = self.findChild(Keypoint, "kp_%02d" % idx)
        obj.set_active()
        return
    
    def is_active(self, idx):
        state = getattr(self, "kp_%02d"%idx).is_active()
        return state

    def select_keypoint(self, x, y):
        min_obj, min_dist = None, 100000
        for child in self.findChildren(Keypoint):
            curr_dist = child.get_distance(x, y)
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_obj = child
            
        if min_dist <= 3:
            min_obj.set_active()
            self.set_exclusive_state(min_obj)
        return
    
    def draw_hand_kp(self, img, offset):
        if not self.cb_show.isChecked():
            return img
        
        cm = plt.get_cmap('gist_rainbow')
        for idx in range(21):
            x, y = getattr(self.param, "kp_x%02d"%idx), getattr(self.param, "kp_y%02d"%idx)
            if x == 0 and y == 0:
                continue
            radius, thickness = (2, 3) if self.is_active(idx) else (1, 2)
            color = (np.array(cm((idx+offset)/42)[:3]) * 255).tolist()[::-1]
            cv2.circle(img, (x, y), radius, tuple(color), thickness=thickness)
        return img
