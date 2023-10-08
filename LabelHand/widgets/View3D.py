from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

import time
import numpy as np
from LabelHand.Alg.transformations import Arcball
import open3d as o3d
from open3d import utility
import open3d.visualization.rendering as rendering
from qtpy.QtGui import QImage, qRgb
import copy

g_render = rendering.OffscreenRenderer(512, 512)
g_render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA


# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

MOVE_SPEED = 5.0


def local_time():
    return time.strftime("%H:%M:%S", time.localtime())


class View3D(QtWidgets.QWidget):
 
    def __init__(self, name, parent, tip, init_fov=45):
        super(View3D, self).__init__(parent)
        self.setObjectName(name)
        self.setToolTip(tip)
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
 
        self.mesh = None
        self.init_fov = init_fov
        self.init_arc_ball()
        
        return
    
    def init_arc_ball(self):
        self.arc_ball = Arcball(initial=np.identity(4))
        win_width, win_height = self.size().width(), self.size().height()
        radius = min(win_width, win_height) // 2
        ct_x, ct_y = win_width // 2, win_height // 2
        self.arc_ball.place((ct_x, ct_y), radius)
        self.fov = self.init_fov
        self.update()
        return
    
    def exist_mesh(self):
        return self.mesh is not None
    
    def set_mesh(self, mesh, with_frame=False):
        if not with_frame:
            self.mesh = mesh
            self.update()
            return

        vertices = np.asarray(mesh.vertices)
        ct = vertices.mean(axis=0)
        max_size = (vertices.max(axis=0) - vertices.min(axis=0)).max()
        mc = o3d.geometry.TriangleMesh().create_coordinate_frame(size=max_size//2, origin=ct)
        self.mesh = copy.deepcopy(mesh+mc)
        self.update()
        return

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def mouseMoveEvent(self, ev):
        curr_pt = ev.localPos()
        if QtCore.Qt.LeftButton == ev.buttons() and self.exist_mesh():
            self.arc_ball.drag([self.size().width()-curr_pt.x(), curr_pt.y()])
            self.repaint()
        return

    def mousePressEvent(self, ev):
        curr_pt = ev.localPos()
        
        if QtCore.Qt.LeftButton == ev.buttons() and self.exist_mesh():
            self.arc_ball.down([self.size().width()-curr_pt.x(), curr_pt.y()])
             
        return

    def mouseReleaseEvent(self, ev):
        curr_pt = ev.localPos()
        
        if QtCore.Qt.LeftButton == ev.buttons() and self.exist_mesh():
            self.arc_ball.drag([self.size().width()-curr_pt.x(), curr_pt.y()])
            self.repaint()
 
        return

    def mouseDoubleClickEvent(self, ev):
        if QtCore.Qt.LeftButton == ev.buttons():
            self.init_arc_ball()
        return
     
    def paintEvent(self, event):
        def get_render(img_size):
            global g_render
            curr_size = np.array(g_render.render_to_image()).shape[:2][::-1]
            if curr_size[0] != img_size[0] or curr_size[1] != img_size[1]:
                g_render = rendering.OffscreenRenderer(img_size[0], img_size[1])
                g_render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA
                print(local_time(), "view3d create new render", curr_size, img_size)
            return g_render
        
        if not self.exist_mesh():
            return super(View3D, self).paintEvent(event)

        material = rendering.MaterialRecord()
        material.shader = "defaultLit"

        img_size = np.array([self.size().width(), self.size().height()])
        g_render = get_render(img_size)
        # g_render = rendering.OffscreenRenderer(img_size[0], img_size[1])
        # g_render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA
        g_render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0], 20000)
        g_render.scene.scene.enable_sun_light(True)
        
        R = self.arc_ball.matrix()
        merge_mesh = copy.deepcopy(self.mesh)
         
        merge_mesh = merge_mesh.rotate(R[:3, :3])

        render_by_intrinsic = True
        if render_by_intrinsic:
            ct = merge_mesh.get_center()
            ct[2] = ct[2] - 500
            merge_mesh = merge_mesh.translate(-ct)
            g_render.scene.add_geometry(f"mesh", merge_mesh, material)
            
            focal = 0.5*min(img_size[0], img_size[1])/np.tan(self.fov*np.pi/2/180)
            param = o3d.camera.PinholeCameraIntrinsic(img_size[0], img_size[1], focal, focal, img_size[0]//2, img_size[1]//2)
            extrinsic_matrix = np.eye(4, dtype=np.float64)
            g_render.setup_camera(param, extrinsic_matrix)
        else:
            g_render.scene.add_geometry(f"mesh", merge_mesh, material)
            ct = merge_mesh.get_center()
            lookat, eye, up = ct, ct - np.array([0, 0, 500]), np.array([0, -1, 0])
            g_render.setup_camera(self.fov, lookat, eye, up)
        
        img = g_render.render_to_image()
        image = np.asarray(img)
        g_render.scene.remove_geometry("mesh")
        # print(local_time(), "fov", self.fov, image.shape[:2][::-1], img_size)

        qimg = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        self.pixmap = QtGui.QPixmap.fromImage(qimg)
        
        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.drawPixmap(0, 0, self.pixmap)

        p.end()
        return
    
    def resizeEvent(self, event):
        win_width, win_height = self.size().width(), self.size().height()
        radius = min(win_width, win_height) // 2
        ct_x, ct_y = win_width // 2, win_height // 2
        self.arc_ball.place((ct_x, ct_y), radius)
         
        return super(View3D, self).resizeEvent(event)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.exist_mesh():
            return self.parent().size()
        return super(View3D, self).minimumSizeHint()

    def wheelEvent(self, ev):
        delta = ev.angleDelta()
        units = 1.1 if delta.y() > 0 else 0.9
        self.fov *= units
        self.update()
        ev.accept()
        return

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()

        if modifiers == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_Z: 
            pass
        return

    def keyReleaseEvent(self, ev):
        return

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()
 


def app_tx():
    from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
    
    app = QApplication([])
    app.setStyle('Fusion')
    window = QWidget()
    view = View3D()
    layout = QVBoxLayout()
    layout.addWidget(view)
    window.setLayout(layout)
    
    knot_mesh = o3d.data.KnotMesh()
    mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1, 0.706, 0])
    
    view.set_mesh(mesh, True)
    
    window.show()
    app.exec_()
    
    return


if __name__ == "__main__":
    app_tx()

