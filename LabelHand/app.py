# -*- coding: utf-8 -*-

import functools
import math
import os
import os.path as osp
import re
import traceback
import webbrowser
import glob
import copy
import subprocess

import imgviz
import natsort
import setuptools.glob
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtGui import QImage, qRgb


from LabelHand import __appname__

from . import utils
from LabelHand.config import get_config
from LabelHand.label_file import LabelFile
from LabelHand.label_file import LabelFileError
from LabelHand.logger import logger
from LabelHand.widgets import Canvas
from LabelHand.widgets import FileDialogPreview
from LabelHand.widgets import ToolBar
from LabelHand.widgets import ZoomWidget
from LabelHand.widgets import HandPoseWidget, HandBetaWidget, HandGlobalWidget, HandKeypointWidget

import cv2
import numpy as np
from LabelHand.Alg.ManoTx import MANO, render_two_hand, local_time, recify_finger_pose, get_cube_from_bound
from LabelHand.widgets.View3D import View3D
import torch
import open3d as o3d


# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".


# LABEL_COLORMAP = imgviz.label_colormap()


class MainWindow(QtWidgets.QMainWindow):

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
        resolution=None
    ):
        if output is not None:
            logger.warning(
                "argument output is deprecated, use output_file instead"
            )
            if output_file is None:
                output_file = output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        self.right_mano = MANO("./Model/MANO", True, False, flat_hand_mean=True)
        self.left_mano = MANO("./Model/MANO", False, False, flat_hand_mean=True)
        if torch.sum(torch.abs(self.left_mano.shapedirs[:, 0, :] - self.right_mano.shapedirs[:, 0, :])) < 1:
            print('Fix shapedirs bug of MANO')
            self.left_mano.shapedirs[:, 0, :] *= -1
        
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
    
        self.resolution = resolution
        # Whether we need to save or not.
        self.dirty = False
        self.stackLabelFile = []
        self.clip_board = None

        # Main widgets and related state. 
        self.lastOpenDir = None

        self.global_hand = HandGlobalWidget(self.tr("global hand"), self)
        self.global_hand_dock = QtWidgets.QDockWidget(self.tr("global hand configure"), self)
        self.global_hand_dock.setObjectName(self.tr("global Hand"))
        self.global_hand_dock.setWidget(self.global_hand)

        self.right_hand_beta = HandBetaWidget(self.tr("right hand beta"), self, self.right_mano)
        self.right_hand_beta_dock = QtWidgets.QDockWidget(self.tr("right hand shape"), self)
        self.right_hand_beta_dock.setObjectName(self.tr("Right Hand Beta"))
        self.right_hand_beta_dock.setWidget(self.right_hand_beta)
         
        self.right_hand_pose = HandPoseWidget(self.tr("right hand pose"), self,
                                              self.right_mano, self.right_hand_beta, self.global_hand)
        self.right_hand_pose_dock = QtWidgets.QDockWidget(self.tr("right hand pose"), self)
        self.right_hand_pose_dock.setObjectName(self.tr("Right Hand Pose"))
        self.right_hand_pose_dock.setWidget(self.right_hand_pose)

        self.left_hand_beta = HandBetaWidget(self.tr("left hand beta"), self, self.left_mano)
        self.left_hand_beta_dock = QtWidgets.QDockWidget(self.tr("left hand shape"), self)
        self.left_hand_beta_dock.setObjectName(self.tr("Left Hand Beta"))
        self.left_hand_beta_dock.setWidget(self.left_hand_beta)
        
        self.left_hand_pose = HandPoseWidget(self.tr("left hand pose"), self,
                                             self.left_mano, self.left_hand_beta, self.global_hand)
        self.left_hand_pose_dock = QtWidgets.QDockWidget(self.tr("left hand pose"), self)
        self.left_hand_pose_dock.setObjectName(self.tr("Left Hand Pose"))
        self.left_hand_pose_dock.setWidget(self.left_hand_pose)

        self.right_hand_kp = HandKeypointWidget(self.tr("right_hand_keypoint"), self)
        self.right_hand_kp_dock = QtWidgets.QDockWidget(self.tr("right hand keypoint"), self)
        self.right_hand_kp_dock.setObjectName(self.tr("Right Hand Keypoint"))
        self.right_hand_kp_dock.setWidget(self.right_hand_kp)

        self.left_hand_kp = HandKeypointWidget(self.tr("left hand keypoint"), self)
        self.left_hand_kp_dock = QtWidgets.QDockWidget(self.tr("left hand keypoint"), self)
        self.left_hand_kp_dock.setObjectName(self.tr("Left Hand Keypoint"))
        self.left_hand_kp_dock.setWidget(self.left_hand_kp)

        self.global_view = View3D(self.tr("global_view"), self, self.tr("global 3d view"))
        self.global_view_dock = QtWidgets.QDockWidget(self.tr("global 3d view"), self)
        self.global_view_dock.setObjectName(self.tr("Global 3D View"))
        self.global_view_dock.setWidget(self.global_view)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        self.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)
        
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)

        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)
        self.canvas.ctrlPress.connect(self.set_hand_kp)
        self.canvas.shiftPress.connect(self.set_hand_pos)
        self.canvas.mouseMove.connect(self.show_pos)
        self.canvas.doubleClick.connect(self.select_kp)
        self.right_hand_pose.sig_config_changed.connect(self.hp_config_changed)
        self.right_hand_beta.sig_config_changed.connect(self.hp_config_changed)
        self.left_hand_pose.sig_config_changed.connect(self.hp_config_changed)
        self.left_hand_beta.sig_config_changed.connect(self.hp_config_changed)
        self.global_hand.sig_config_changed.connect(self.hp_config_changed)
        self.global_hand.sig_restore_template.connect(self.restore_from_template)
        self.global_hand.sig_3d_view.connect(self.open_3d_view)
        self.right_hand_kp.sig_config_changed.connect(self.hp_config_changed)
        self.left_hand_kp.sig_config_changed.connect(self.hp_config_changed)

        self.setCentralWidget(scrollArea)
        
        self.scrollArea = scrollArea
        self.canvas.setScrollArea(scrollArea)

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["right_hand_pose_dock", "right_hand_beta_dock", "left_hand_pose_dock", "left_hand_beta_dock",
                     "file_dock", "global_hand_dock", "right_hand_kp_dock", "left_hand_kp_dock", "global_view_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.global_view_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_hand_pose_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_hand_beta_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.left_hand_pose_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.left_hand_beta_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.global_hand_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_hand_kp_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.left_hand_kp_dock)
        
        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            "quit",
            self.tr("Quit application"),
        )
        open_ = action(
            self.tr("&Open"),
            self.openFile,
            shortcuts["open"],
            "open",
            self.tr("Open image or label file"),
        )
        opendir = action(
            self.tr("&Open Dir"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "open",
            self.tr("Open Dir"),
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevImg = action(
            self.tr("&Prev Image"),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        save = action(
            self.tr("&Save"),
            self.saveFile,
            shortcuts["save"],
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=True,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.saveAuto.setChecked(x),
            icon="save",
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text="Save With Image Data",
            slot=self.enableSaveImageWithData,
            tip="Save image data in label file",
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            "&Close",
            self.closeFile,
            shortcuts["close"],
            "close",
            "Close current file",
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep pevious annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])
 
        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="help",
            tip=self.tr("Show tutorial page"),
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(
                    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            functools.partial(self.addZoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            functools.partial(self.addZoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            functools.partial(self.setZoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )

        onlyLabelExist = action(
            self.tr("Only Label Exist"),
            self.only_show_label_exist,
            shortcuts["only_label_exist"],
            "only-label-exist",
            self.tr("only show label exist samples"),
            checkable=True,
            enabled=True,
        )
        
        # Group zoom controls into a list for easier toggling.
        zoomActions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        undo = action(
            self.tr("Undo"),
            self.recover_labelfile,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit"),
            enabled=False,
        )

        copyLabel = action(
            self.tr("Copy Label File"),
            self.copy_label_file,
            shortcuts["copy_label"],
            "copy-label",
            self.tr("copy label file"),
            enabled=True,
        )

        pasteLabel = action(
            self.tr("Paste Label File"),
            self.paste_label_file,
            shortcuts["paste_label"],
            "paste-label",
            self.tr("paste label file"),
            enabled=True,
        )

        copyPath = action(
            self.tr("Copy Path"),
            self.copy_path,
            shortcuts["copy_path"],
            "copy-path",
            self.tr("copy file path"),
            enabled=True,
        )

        crop = action(
            self.tr("Crop Image"),
            self.crop_image,
            shortcuts["crop"],
            "crop-image",
            self.tr("crop image"),
            checkable=True,
            enabled=True,
        )
         
        # Store actions for further handling.
        self.actions = utils.struct(
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            zoomActions=zoomActions,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
            fileMenuActions=(open_, opendir, save, saveAs, close, quit),
            tool=(),
            undo=undo,
            copyLabel=copyLabel,
            pasteLabel=pasteLabel,
            copyPath=copyPath,
            crop=crop,
            onlyLabelExist=onlyLabelExist,
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                None,
                undo,
                copyLabel,
                pasteLabel,
                copyPath,
                crop,
            ),
        )

        self.menus = utils.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
        )

        utils.addActions(
            self.menus.file,
            (
                open_,
                openNextImg,
                openPrevImg,
                opendir,
                self.menus.recentFiles,
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help,))
        utils.addActions(
            self.menus.view,
            (
                self.right_hand_pose_dock.toggleViewAction(),
                self.right_hand_beta_dock.toggleViewAction(),
                self.left_hand_pose_dock.toggleViewAction(),
                self.left_hand_beta_dock.toggleViewAction(),
                self.global_hand_dock.toggleViewAction(),
                self.right_hand_kp_dock.toggleViewAction(),
                self.left_hand_kp_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                self.global_view_dock.toggleViewAction(),
                None,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                onlyLabelExist,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)
        self.actions.undo.setEnabled(True)
        utils.addActions(self.menus.edit, self.actions.editMenu)

        self.tools = self.toolbar("Tools")
        # Menu buttons on Left
        self.actions.tool = (
            open_,
            opendir,
            openNextImg,
            openPrevImg,
            save,
            deleteFile,
            None,
            zoom,
            fitWidth,
        )

        self.statusBar().showMessage(str(self.tr("%s started.")) % __appname__)
        self.statusBar().show()

        if output_file is not None and self._config["auto_save"]:
            logger.warn(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        # self.image = QtGui.QImage()
        self.image = None
        self.imagePath = None
        self.recentFiles = []
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("LabelHand", "LabelHand")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(state)
        self.setFocusPolicy(Qt.StrongFocus)
 
        if filename is not None:
            if osp.isdir(filename):
                self.importDirImages(filename, load=False)
            else:
                self.filename = filename
                self.queueEvent(functools.partial(self.loadFile, self.filename))  # runs in the background
                self.lastOpenDir = self.settings.value("lastOpenDir", None)
        else:
            self.filename = self.settings.value("filename", None)
            self.lastOpenDir = self.settings.value("lastOpenDir", None)

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        # if self.filename is not None:
        #     self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()
        return

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar

    # Support Functions
    def setDirty(self, is_undo=False):
        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.saveLabels(label_file, is_undo)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}*".format(title, self.filename)
        self.setWindowTitle(title)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}".format(title, self.filename)
        self.setWindowTitle(title)
        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)
        return
    
    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        return

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks 
    def tutorial(self):
        url = "www.google.com"  # NOQA
        webbrowser.open(url)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, "&%d %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)
        return

    def fileSearchChanged(self):
        self.importDirImages(self.lastOpenDir, pattern=self.fileSearch.text(), load=False)
        return

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.mayContinue():
            return

        currIndex = self.imageList.index(str(item.text()))
        if currIndex < len(self.imageList):
            filename = self.imageList[currIndex]
            if filename:
                self.loadFile(filename)
        return

    def saveLabels(self, filename, is_undo=False):
        # print(local_time(), "save label")
        lf = LabelFile()
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.shape[0],
                imageWidth=self.image.shape[1],
                right_hand_param=self.save_hand_param(True),
                left_hand_param=self.save_hand_param(False),
                global_hand_param=self.global_hand.param.to_dict(),
                right_hand_kp_param=self.right_hand_kp.param.to_dict(),
                left_hand_kp_param=self.left_hand_kp.param.to_dict()
            )
            if self.labelFile and not is_undo:
                self.stackLabelFile.append(self.labelFile)
                self.stackLabelFile = self.stackLabelFile[-50:]
            self.labelFile = lf
            items = self.fileListWidget.findItems(self.imagePath, Qt.MatchExactly)
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    # Callback functions: 
    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)
        return
    
    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        # print("setZoom1", value)
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        # print("setZoom2", value)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def zoomRequest(self, delta, pos):
        if self.image is None:
            return
        
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        bar = self.scrollBars[Qt.Horizontal]
        
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )
        return

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)
        return

    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if filename in self.imageList and (
            self.fileListWidget.currentRow() != self.imageList.index(filename)
        ):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        # assumes same name, but json extension
        self.status(
            str(self.tr("Loading %s...")) % osp.basename(str(filename))
        )
        label_file = osp.splitext(filename)[0] + ".json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                if self.labelFile:
                    self.stackLabelFile.append(self.labelFile)
                    self.stackLabelFile = self.stackLabelFile[-50:]
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            # self.imageData = self.labelFile.imageData
            self.imagePath = osp.join(osp.dirname(label_file), self.labelFile.imagePath)
            self.otherData = self.labelFile.otherData
        else:
            self.labelFile = None
        
        self.imageData = LabelFile.load_image_file(filename)
        if self.imageData:
            self.imagePath = filename
        
        # origin code
        # image = QtGui.QImage.fromData(self.imageData)
        # new code:
        image = cv2.imdecode(np.frombuffer(self.imageData, np.uint8), cv2.IMREAD_COLOR)
        image = np.ascontiguousarray(image[:, :, ::-1])
        # end
        
        if image is None:
            formats = [
                "*.{}".format(fmt.data().decode())
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % filename)
            return False
        
        self.image = image
        self.filename = filename
        if self._config["keep_prev"]:
            # prev_shapes = self.canvas.shapes
            pass
        self.set_initial_roi()

        if self.labelFile:
            self.restore_param(self.labelFile)
        self.display_param()
        self.set_canva()

        # if self._config["keep_prev"] and self.noShapes():
        #     # self.loadShapes(prev_shapes, replace=False)
        #     self.setDirty()
        # else:
        #     self.setClean()
        
        self.setClean()
        self.canvas.setEnabled(True)
        
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
         
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()
        self.status(str(self.tr("Loaded %s")) % osp.basename(str(filename)))
        self.show_pos(QtCore.QPointF(0, 0))
        return True

    def resizeEvent(self, event):
        if (
            self.canvas
            and self.image is not None
            and self.zoomMode != self.MANUAL_ZOOM
        ):
            self.adjustScale()
        # print(local_time(), "reiszeEvent canva size", self.canvas.size(), self.scrollArea.size())
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert self.image is not None, "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("lastOpenDir", self.lastOpenDir if self.lastOpenDir else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        # ask the use for where to save the labels
        self.settings.setValue('window/geometry', self.saveGeometry())
        return
    
    def dragEnterEvent(self, event):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.mayContinue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def openPrevImg(self, _value=False):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        if self.filename is None:
            return

        currIndex = self.imageList.index(self.filename)
        if currIndex - 1 >= 0:
            filename = self.imageList[currIndex - 1]
            if filename:
                self.loadFile(filename)

        self._config["keep_prev"] = keep_prev

    def openNextImg(self, _value=False, load=True):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.imageList[0]
        else:
            currIndex = self.imageList.index(self.filename)
            if currIndex + 1 < len(self.imageList):
                filename = self.imageList[currIndex + 1]
            else:
                filename = self.imageList[-1]
        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

        self._config["keep_prev"] = keep_prev

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + ["*%s" % LabelFile.suffix]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setDirectory(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self.loadFile(fileName)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(
                self.imageList.index(current_filename)
            )
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        assert self.image is not None, "cannot save empty image"
        if self.labelFile:
            # DL20180323 - overwrite when in directory
            self._saveFile(self.labelFile.filename)
        elif self.output_file:
            self._saveFile(self.output_file)
            self.close()
        else:
            self._saveFile(self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image is not None, "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.output_dir, filters
            )
        else:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.currentPath(), filters
            )
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def getLabelFile(self):
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"
        return label_file

    def deleteFile(self):
        label_file = self.getLabelFile()
        if not osp.exists(label_file):
            mb = QtWidgets.QMessageBox
            msg = self.tr("不存在标注结果")
            mb.information(self, self.tr("Attention"), msg)
            return
            
        mb = QtWidgets.QMessageBox
        msg = self.tr("是否永久删除标注结果，不可恢复")
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return
        
        os.remove(label_file)
        logger.info("Label file is removed: {}".format(label_file))

        item = self.fileListWidget.currentItem()
        item.setCheckState(Qt.Unchecked)

        self.resetState()
        
        return
 
    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(
            self.filename
        )
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]
 
    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        # defaultOpenDirPath = dirpath if dirpath else "."
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = (osp.dirname(self.filename) if self.filename else ".")

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )
        self.importDirImages(targetDirPath)
        return

    @property
    def imageList(self):
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            lst.append(item.text())
        return lst

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
                
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)

        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)

        self.openNextImg()

    def importDirImages(self, dirpath, pattern=None, load=True):
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)

        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.filename = None
        self.fileListWidget.clear()
        for filename in self.scanAllImages(dirpath):
            if pattern and pattern not in filename:
                continue
            if osp.splitext(filename)[0].endswith("_mask"):
                continue
            
            label_file = osp.splitext(filename)[0] + ".json"
            if self.actions.onlyLabelExist.isChecked() and not osp.exists(label_file):
                continue
            
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(filename)
            item.setToolTip(os.path.basename(filename))
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)
        self.openNextImg(load=load)
        return

    def scanAllImages(self, folderPath):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        images = []
        # for root, dirs, files in os.walk(folderPath):
        #     for file in files:
        #         if file.lower().endswith(tuple(extensions)):
        #             relativePath = osp.join(root, file)
        #             images.append(relativePath)
        for file in glob.glob(os.path.join(folderPath, "*")):
            if file.lower().endswith(tuple(extensions)):
                images.append(file)
        images = natsort.os_sorted(images)
        return images

    def toQImage(self, image):
        qimg = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        return qimg
    
    def restore_param(self, label_file):
        self.restore_hand_param(label_file.right_hand_param, True)
        self.restore_hand_param(label_file.left_hand_param, False)
        self.global_hand.param.from_dict(label_file.global_hand_param)
        self.right_hand_kp.param.from_dict(label_file.right_hand_kp_param)
        self.left_hand_kp.param.from_dict(label_file.left_hand_kp_param)
        return
    
    def display_param(self):
        self.right_hand_pose.display_param()
        self.right_hand_beta.display_param()
        self.left_hand_pose.display_param()
        self.left_hand_beta.display_param()
        self.global_hand.display_param()
        self.right_hand_kp.display_param()
        self.left_hand_kp.display_param()
        return
    
    def recover_labelfile(self):
        if len(self.stackLabelFile) == 0:
            return
        
        lf = self.stackLabelFile.pop()
        if self.labelFile.filename != lf.filename:
            print(local_time(), "ignore recovered label file from different file")
            return
        
        print(local_time(), "rescore_labelfile")
        self.restore_param(lf)
        self.display_param()
        self.set_canva()
        self.setDirty(True)
        return
     
    def set_hand_pos(self, pos, is_right):
        if self.image is None:
            return
        obj = self.right_hand_pose if is_right else self.left_hand_pose
        roi_width, roi_height = self.global_hand.param.get_roi_size()
        x, y = int(pos.x()), int(pos.y())
        x = x % roi_width
        obj.set_hand_pose(x, y)
        obj.config_changed()
        return
     
    def hp_config_changed(self, is_dirty=True):
        if self.image is not None:
            self.set_canva()
            if is_dirty:
                self.setDirty()
        return
    
    def save_hand_param(self, is_rhand):
        hand_pose = self.right_hand_pose if is_rhand else self.left_hand_pose
        hand_beta = self.right_hand_beta if is_rhand else self.left_hand_beta
        pose_dict = hand_pose.param.to_dict()
        beta_dict = hand_beta.param.to_dict()
        param_dict = {**pose_dict, **beta_dict}
        return param_dict
    
    def restore_hand_param(self, hand_param, is_rhand):
        hand_pose = self.right_hand_pose if is_rhand else self.left_hand_pose
        hand_beta = self.right_hand_beta if is_rhand else self.left_hand_beta
        hand_pose.param.from_dict(hand_param)
        hand_beta.param.from_dict(hand_param)
        return
     
    def get_mano_param(self, is_rhand):
        hand_pose = self.right_hand_pose if is_rhand else self.left_hand_pose
        hand_beta = self.right_hand_beta if is_rhand else self.left_hand_beta
        
        xy, depth, root_pose, finger_pose = hand_pose.param.to_mano_param(hand_pose.mano.is_rhand)
        beta, recify_rot, ori_root_pos, ori_finger_root_pos = hand_beta.param.to_mano_param()
        finger_pose = recify_finger_pose(finger_pose, recify_rot)
        return xy, depth, root_pose, finger_pose, beta, ori_root_pos
    
    def set_canva(self):
        right_hand_param = self.get_mano_param(True)
        right_draw_config = self.right_hand_pose.get_draw_config()

        left_hand_param = self.get_mano_param(False)
        left_draw_config = self.left_hand_pose.get_draw_config()
        
        global_param = self.global_hand.param.to_mano_param()
        img, canva, _, geo = render_two_hand(self.image, self.right_mano, self.left_mano, right_hand_param,
                                             left_hand_param, global_param, right_draw_config, left_draw_config)

        img[:, -1], canva[:, 0] = 255, 255
        img = self.right_hand_kp.draw_hand_kp(img, 0)
        img = self.left_hand_kp.draw_hand_kp(img, 21)
        
        self.set_global_view(geo)

        img_merge = np.hstack([img, canva])
        qimg = self.toQImage(img_merge)
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimg))
        return
    
    
    def set_global_view(self, geo):
        merge_mesh = o3d.geometry.TriangleMesh()
        for item in geo:
            mesh = item["geometry"]
            # vertices = np.asarray(mesh.vertices)
            # ct = vertices.mean(axis=0)
            # max_size = (vertices.max(axis=0) - vertices.min(axis=0)).max()
            # axes = o3d.geometry.TriangleMesh().create_coordinate_frame(size=max_size//2, origin=ct)
            # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(100).translate(mesh.get_center())
            merge_mesh = merge_mesh + mesh

        box_mesh = self.create_bounding_box(merge_mesh)
        
        self.global_view.set_mesh(merge_mesh+box_mesh)
        return
    
    @staticmethod
    def create_bounding_box(mesh):
        max_val = mesh.get_max_bound()
        min_val = mesh.get_min_bound()
        radius = (max_val - min_val)*0.1
        max_val, min_val = max_val + radius, min_val - radius
        
        bound = np.hstack([min_val[..., None], max_val[..., None]])
        
        vertices, edges, faces = get_cube_from_bound(bound)

        box = o3d.geometry.TriangleMesh()
        for edge in edges:
            st, et = vertices[edge[0]], vertices[edge[1]]
            length = np.linalg.norm(et-st)
            if np.allclose(length, 0):
                continue
            line = o3d.geometry.TriangleMesh().create_cylinder(1, length).translate(np.array([0, 0, length//2]))
            
            if abs(st[0]-et[0]) > 1E-6:
                color = np.array([0.8, 0, 0])
                rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 1*np.pi/2, 0]))
                box += copy.deepcopy(line).rotate(rotation, center=(0, 0, 0)).translate(st).paint_uniform_color(color)
            elif abs(st[1]-et[1]) > 1E-6:
                color = np.array([0, 0.8, 0])
                rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([-1*np.pi/2, 0, 0]))
                box += copy.deepcopy(line).rotate(rotation, center=(0, 0, 0)).translate(st).paint_uniform_color(color)
            elif abs(st[2]-et[2]) > 1E-6:
                color = np.array([0, 0, 0.8])
                box += copy.deepcopy(line).translate(st).paint_uniform_color(color)
            else:
                pass
        box.compute_vertex_normals()
        box.compute_triangle_normals()
        return box
    
    def set_hand_kp(self, pos, is_rhand):
        x, y = int(pos.x()+0.5), int(pos.y()+0.5)
        roi_width, roi_height = self.global_hand.param.get_roi_size()
        if x >= roi_width or y >= roi_height:
            return
        hand_kp = self.right_hand_kp if is_rhand else self.left_hand_kp
        hand_kp.set_keypoint(x, y)
        hand_kp.config_changed()
        return
     
    def show_pos(self, pos):
        if self.image is not None:
            roi_width, roi_height = self.global_hand.param.get_roi_size()
            x, y = int(pos.x()+0.5), int(pos.y()+0.5)
            if roi_width != 0:
                x = x % roi_width
            self.status("size=%dx%d x=%d y=%d" % (roi_width, roi_height, x, y))
        return
     
    def select_kp(self, pos):
        x, y = int(pos.x()+0.5), int(pos.y()+0.5)
        self.right_hand_kp.select_keypoint(x, y)
        self.left_hand_kp.select_keypoint(x, y)
        self.hp_config_changed(False)
        return
    
    def set_initial_roi(self):
        self.global_hand.set_image_size(self.image.shape[1], self.image.shape[0])
        left, top, right, bottom = 0, 0, self.image.shape[1], self.image.shape[0]
        self.global_hand.param.spin_left = left
        self.global_hand.param.spin_top = top
        self.global_hand.param.spin_right = right
        self.global_hand.param.spin_bottom = bottom
        return
     
    def restore_from_template(self, path_or_label_file):
        def get_values(dt, keys):
            values = []
            for key in keys:
                values.append(dt[key])
            return values

        # print(local_time(), self.left_hand_pose.param.to_dict())
        if isinstance(path_or_label_file, str):
            lf = LabelFile(path_or_label_file)
        else:
            lf = path_or_label_file
        
        self.left_hand_pose.param.from_dict(lf.left_hand_param)
        self.right_hand_pose.param.from_dict(lf.right_hand_param)
        
        keys = ["spin_left", "spin_top", "spin_right", "spin_bottom"]
        tp_left, tp_top, tp_right, tp_bottom = get_values(lf.global_hand_param, keys)
        curr_left, curr_top, curr_right, curr_bottom = self.global_hand.param.get_roi_rect()
        x_ratio = (curr_right-curr_left)/(tp_right-tp_left)
        y_ratio = (curr_bottom-curr_top)/(tp_bottom-tp_top)
        ratio = np.sqrt(x_ratio*y_ratio)
        arr_ratio = np.array([ratio, ratio, 1.0/ratio])
        
        left_coord = np.array(get_values(lf.left_hand_param, ["spin_x", "spin_y", "sld_depth"]))
        right_coord = np.array(get_values(lf.right_hand_param, ["spin_x", "spin_y", "sld_depth"]))
        scale_left_coord, scale_right_coord = (left_coord*arr_ratio).tolist(), (right_coord*arr_ratio).tolist()
         
        keys = ["image_width", "image_height"]
        tp_img_width, tp_img_height = get_values(lf.global_hand_param, keys)
        curr_img_height, curr_img_width = self.image.shape[:2]
        img_ratio = np.sqrt(curr_img_width*curr_img_height/(tp_img_width*tp_img_height))
        self.canvas.roi_rect.set_rect(tp_left, tp_top, tp_right, tp_bottom, img_ratio)
        
        # override coord 
        self.left_hand_pose.param.set_coord(*scale_left_coord)
        self.right_hand_pose.param.set_coord(*scale_right_coord)
        # print(local_time(), self.left_hand_pose.param.to_dict())

        self.display_param()
        self.hp_config_changed()
        return
    
    def crop_image(self):
        if not self.canvas.roi_rect.is_efficient():
            return
        left, top, width, height = self.canvas.roi_rect.get_rect()
        image_height, image_width = self.image.shape[:2]
        left, right = left % image_width, (left+width) % image_width
        bottom = top + height
        self.canvas.roi_rect.reset()
        self.global_hand.set_roi(left, top, right, bottom)
        self.global_hand.config_changed(True)
        return
    
    def copy_label_file(self):
        self.clip_board = copy.deepcopy(self.labelFile)
        return
    
    def paste_label_file(self):
        if self.image is not None and self.clip_board is not None:
            self.restore_from_template(self.clip_board)
        return
    
    def open_3d_view(self):
        if self.image is None:
            return
        return
    
    def only_show_label_exist(self, value):
        self.actions.onlyLabelExist.setChecked(value)
        self.importDirImages(self.lastOpenDir, load=True)
        return
    
    def copy_path(self):
        command = 'echo ' + self.filename.strip() + '| clip'
        os.system(command)
        return