import base64
import contextlib
import io
import json
import os.path as osp

import PIL.Image

from LabelHand import __version__
from LabelHand.logger import logger
from LabelHand import PY2
from LabelHand import QT4
from LabelHand import utils
import copy



PIL.Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    if PY2:
        mode += "b"
        encoding = None
    else:
        encoding = "utf-8"
    yield io.open(name, mode, encoding=encoding)
    return


class LabelFileError(Exception):
    pass


class LabelFile(object):

    suffix = ".json"

    def __init__(self, filename=None):
        self.imagePath = None
        self.imageData = None
        self.right_hand_param = {}
        self.left_hand_param = {}
        self.global_hand_param = {}
        self.right_hand_kp_param = {}
        self.left_hand_kp_param = {}
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error("Failed opening image file: {}".format(filename))
            return

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if PY2 and QT4:
                format = "PNG"
            elif ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "imageHeight",
            "imageWidth",
            "right_hand_param",
            "left_hand_param",
            "global_hand_param",
            "right_hand_kp_param",
            "left_hand_kp_param",
        ]
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            version = data.get("version")
            if version is None:
                logger.warning(
                    "Loading JSON file ({}) of unknown version".format(
                        filename
                    )
                )
            elif version.split(".")[0] != __version__.split(".")[0]:
                logger.warning(
                    "This JSON file ({}) may be incompatible with "
                    "current version. version in file: {}, "
                    "current version: {}".format(
                        filename, version, __version__
                    )
                )

            if data["imageData"] is not None:
                imageData = base64.b64decode(data["imageData"])
                if PY2 and QT4:
                    imageData = utils.img_data_to_png_data(imageData)
            else:
                # relative path from label file to relative path from cwd
                # imagePath = osp.join(osp.dirname(filename), data["imagePath"])
                # imageData = self.load_image_file(imagePath)
                imageData = None
            imagePath = data["imagePath"]
            right_hand_param = data.get("right_hand_param", {})
            left_hand_param = data.get("left_hand_param", {})
            global_hand_param = data.get("global_hand_param", {})
            right_hand_kp_param = data.get("right_hand_kp_param", {})
            left_hand_kp_param = data.get("left_hand_kp_param", {})

            # self._check_image_height_and_width(base64.b64encode(imageData).decode("utf-8"), 
            #                                    data.get("imageHeight"), data.get("imageWidth"))
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded 
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData
        self.right_hand_param = right_hand_param
        self.left_hand_param = left_hand_param
        self.global_hand_param = global_hand_param
        self.right_hand_kp_param = right_hand_kp_param
        self.left_hand_kp_param = left_hand_kp_param
        return

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error(
                "imageHeight does not match with imageData or imagePath, "
                "so getting imageHeight from actual image."
            )
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error(
                "imageWidth does not match with imageData or imagePath, "
                "so getting imageWidth from actual image."
            )
            imageWidth = img_arr.shape[1]
        return imageHeight, imageWidth

    def save(
        self,
        filename,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        right_hand_param=None,
        left_hand_param=None,
        global_hand_param=None,
        right_hand_kp_param=None,
        left_hand_kp_param=None,
    ):
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode("utf-8")
            imageHeight, imageWidth = self._check_image_height_and_width( imageData, imageHeight, imageWidth)
        
        self.imagePath = imagePath
        self.imageData = imageData
        self.right_hand_param = copy.deepcopy(right_hand_param)
        self.left_hand_param = copy.deepcopy(left_hand_param)
        self.global_hand_param = copy.deepcopy(global_hand_param)
        self.right_hand_kp_param = copy.deepcopy(right_hand_kp_param)
        self.left_hand_kp_param = copy.deepcopy(left_hand_kp_param)
        
        data = dict(
            version=__version__,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
            right_hand_param=right_hand_param,
            left_hand_param=left_hand_param,
            global_hand_param=global_hand_param,
            right_hand_kp_param=right_hand_kp_param,
            left_hand_kp_param=left_hand_kp_param
        )
        try:
            with open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
