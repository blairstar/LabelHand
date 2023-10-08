from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


from LabelHand import QT5
import LabelHand.utils
import time
import cv2
import numpy as np



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


class ROIRect(object):
    def __init__(self):
        self.left, self.top, self.right, self.bottom = None, None, None, None
        self.sub_left, self.sub_top, self.sub_right, self.sub_bottom = None, None, None, None
        self.select_type = None
        self.last_x, self.last_y = None, None
        return
    
    def reset(self):
        self.left, self.top, self.right, self.bottom = None, None, None, None
        self.select_type = None
        self.last_x, self.last_y = None, None
        return
    
    def set_rect(self, left, top, right, bottom, scale=1.0):
        self.left, self.top = int(left*scale), int(top*scale)
        self.right, self.bottom = int(right*scale), int(bottom*scale)
        return
     
    def is_efficient(self):
        state = self.left and self.top and self.right and self.bottom
        return state
    
    def calc_radius(self):
        x_radius = max(3, int(abs(self.right - self.left)*0.1))
        y_radius = max(3, int(abs(self.bottom - self.top)*0.1))
        return x_radius, y_radius
    
    def too_small(self):
        state = abs(self.right-self.left) < 6 or abs(self.bottom-self.left) < 6
        return state
    
    def have_selected(self):
        return self.select_type is not None
    
    def get_rect(self):
        left, right = min(self.left, self.right), max(self.left, self.right)
        top, bottom = min(self.top, self.bottom), max(self.top, self.bottom)
        width, height = right - left, bottom - top
        return left, top, width, height
    
    def set_start_point(self, x, y):
        self.select_type = 0
        self.left, self.top = x, y
        self.right, self.bottom = None, None
        return
     
    def select(self, pos):  # call when mouse press
        x, y = int(pos.x()+0.5), int(pos.y()+0.5)
        self.update_last_pos(x, y)
        
        self.reformat()
        
        if not self.is_efficient():
            self.set_start_point(x, y)
            return
            
        in_roi = self.left <= x <= self.right and self.top <= y <= self.bottom
        if in_roi is False:
            self.set_start_point(x, y)
            return
        
        self.select_type = 5
        x_radius, y_radius = self.calc_radius()
        if abs(x-self.left) <= x_radius and abs(y-self.top) <= y_radius:
            self.select_type = 1
        if abs(x-self.right) <= x_radius and abs(y-self.top) <= y_radius:
            self.select_type = 2
        if abs(x-self.right) <= x_radius and abs(y-self.bottom) <= y_radius:
            self.select_type = 3
        if abs(x-self.left) <= x_radius and abs(y-self.bottom) <= y_radius:
            self.select_type = 4
        
        # print(local_time(), f"select type {self.select_type}")
        return

    def reformat(self):
        if self.is_efficient():
            self.left, self.right = min(self.left, self.right), max(self.left, self.right)
            self.top, self.bottom = min(self.top, self.bottom), max(self.top, self.bottom)
        return
    
    def update_last_pos(self, x, y):
        self.last_x, self.last_y = x, y
        return
    
    def set_corner(self, pos):  # call when mouse move
        x, y = int(pos.x()+0.5), int(pos.y()+0.5)
        if self.select_type == 0:
            self.right, self.bottom = x, y
        elif self.select_type == 1:
            self.left += (x - self.last_x)
            self.top += (y - self.last_y)
        elif self.select_type == 2:
            self.right += (x - self.last_x)
            self.top += (y - self.last_y)
        elif self.select_type == 3:
            self.right += (x - self.last_x)
            self.bottom += (y - self.last_y)
        elif self.select_type == 4:
            self.left += (x - self.last_x)
            self.bottom += (y - self.last_y)
        elif self.select_type == 5:
            self.left += (x - self.last_x)
            self.top += (y - self.last_y)
            self.right += (x - self.last_x)
            self.bottom += (y - self.last_y)
        else:
            pass
        
        self.update_last_pos(x, y)
        # print(local_time(), f"select type {self.select_type}", self.left, self.top, self.right, self.bottom)
        return

    def mouse_release(self, pos):
        self.set_corner(pos)
        self.reformat()
        
        if self.too_small():  # reset
            self.left, self.top, self.right, self.bottom = None, None, None, None
        
        self.last_x, self.last_y = None, None
        self.select_type = None

        return
    

class Canvas(QtWidgets.QWidget):

    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)
    ctrlPress = QtCore.Signal(QtCore.QPointF, bool)
    shiftPress = QtCore.Signal(QtCore.QPointF, bool)
    mouseMove = QtCore.Signal(QtCore.QPointF)
    doubleClick = QtCore.Signal(QtCore.QPointF)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(
                    self.double_click
                )
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        
        self.st_pt, self.et_pt = None, None
        self.alpha_info = None
        self.is_draw_alpha = False
        
        self.roi_rect = ROIRect()
        return

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1:]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return
         
        if QtCore.Qt.AltModifier == int(ev.modifiers()) and QtCore.Qt.LeftButton == ev.buttons() and\
                self.st_pt is not None:
            self.et_pt = pos
            # self.altCropSize.emit(self.st_pt, self.et_pt)
            self.repaint()
        else:
            self.mouseMove.emit(pos)
        
        if QtCore.Qt.NoModifier == int(ev.modifiers()) and self.roi_rect.have_selected() and not self.pixmap.isNull():
            self.roi_rect.set_corner(pos)
            self.repaint()
            
        return

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())

        # added code
        # because of flipping of image, left button for right hand
        is_right = not (QtCore.Qt.RightButton == ev.buttons())
        if QtCore.Qt.ControlModifier == int(ev.modifiers()):
            self.ctrlPress.emit(pos, is_right)
            return
        if QtCore.Qt.ShiftModifier == int(ev.modifiers()):
            self.shiftPress.emit(pos, is_right)
            return
        if QtCore.Qt.AltModifier == int(ev.modifiers()) and QtCore.Qt.LeftButton == ev.buttons() and \
                not self.pixmap.isNull():
            self.st_pt, self.et_pt = pos, pos
            return
            # self.altPress.emit(pos)
        if QtCore.Qt.NoModifier == int(ev.modifiers()) and not self.pixmap.isNull():
            self.roi_rect.select(pos)
        # end
            
        return

    def mouseReleaseEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())
            
        # if QtCore.Qt.AltModifier == int(ev.modifiers()) and QtCore.Qt.LeftButton == ev.button() \
        #         and self.st_pt is not None:
        #     self.st_et = pos
        #     self.repaint()
        #     if abs(self.st_pt.x() - self.et_pt.x()) > 20 and abs(self.st_pt.y() - self.et_pt.y()) > 20:
        #         ret = QtWidgets.QMessageBox.question(self, "裁剪", "是否裁剪至绿框内的图像")
        #         if ret == QtWidgets.QMessageBox.Yes:
        #             self.altCrop.emit(self.st_pt, self.et_pt)
        #     self.st_pt, self.et_pt = None, None
        
        if QtCore.Qt.NoModifier == int(ev.modifiers()) and not self.pixmap.isNull():
            self.roi_rect.mouse_release(pos)
            self.repaint()
                    
        return

    def mouseDoubleClickEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())
        self.doubleClick.emit(pos)
        return
    
    def setScrollArea(self, scroll):
        self.scrollArea = scroll
        return
    
    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        ct = self.offsetToCenter()
        p.translate(ct)

        p.drawPixmap(0, 0, self.pixmap)
        
        if self.st_pt is not None and self.et_pt is not None:
            x1, y1 = self.st_pt.x(), self.st_pt.y()
            x2, y2 = self.et_pt.x(), self.et_pt.y()
            rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
            pen = QtGui.QPen(QtCore.Qt.green)
            pen.setWidth(max(1, int(round(2.0 / self.scale))))
            p.setPen(pen)
            p.drawRect(rect)
            
        if self.is_draw_alpha and self.alpha_info is not None and not self.pixmap.isNull():
            (l1, t1, r1, b1), (w1, h1) = self.alpha_info
            w2, h2 = self.pixmap.width()//2, self.pixmap.height()
            ratio = np.sqrt(w2/w1 * h2/h1)
            l2, t2 = l1*ratio, t1*ratio
            r2, b2 = r1*ratio, b1*ratio
            rect = QtCore.QRectF(l2, t2, r2-l2, b2-t2)
            pen = QtGui.QPen(QtCore.Qt.yellow)
            pen.setWidth(max(1, int(round(2.0 / self.scale))))
            pen.setStyle(QtCore.Qt.DotLine)
            p.setPen(pen)
            p.drawRect(rect)
            
        if self.roi_rect.is_efficient():
            rect = QtCore.QRectF(*self.roi_rect.get_rect())
            pen = QtGui.QPen(QtCore.Qt.green)
            pen.setWidth(max(1, int(round(2.0 / self.scale))))
            p.setPen(pen)
            p.drawRect(rect)
            
        p.end()
        return

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        # print("canva offsettocenter  area", str(area), "canva", str(self.size()), x, y)
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                # scroll
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (QtCore.Qt.ShiftModifier == int(mods))
                        else QtCore.Qt.Vertical,
                    )
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()
        return

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()
            
        if modifiers == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_Z:
            # ret = QtWidgets.QMessageBox.question(self, "恢复图像", "是否恢复至上一次裁剪前的图像")
            # if ret == QtWidgets.QMessageBox.Yes:
            # self.undo.emit()
            pass
        return

    def keyReleaseEvent(self, ev):
        return

    def loadPixmap(self, pixmap):
        self.pixmap = pixmap
        self.update()
        return
    
    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.update()
