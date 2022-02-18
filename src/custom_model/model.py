import pathlib
import sys

import numpy as np

_parentdir = pathlib.Path("../../mmdetection/").parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
# trunk-ignore(flake8/E402)
from mmdetection.mmdet.apis import inference_detector, init_detector

sys.path.remove(str(_parentdir))


class Model(object):
    """
    Represents a object detection model from mmdetection
    """

    def __init__(self, params: dict) -> None:

        self.model = init_detector(
            params["config"], params["check_pnt"], device="cuda:0"
        )
        self.classes = self.model.CLASSES

        print("Available classes:")
        for i, j in enumerate(self.classes):
            print(i, "-", j)

    def predict(self, img):
        """Detects objects in a given image"""
        result = inference_detector(self.model, img)

        boxes, confidences, labels = self.prepare_result(result)

        return zip(boxes, confidences, labels)

    def prepare_result(self, result) -> tuple:
        """Prepare the predictions to be used"""
        boxes, confs, labels = [], [], []
        # for idx in range(len(result)):
        for idx in [2, 13]:
            for content in result[idx]:
                ax, ay, cx, cy, conf = content
                bx, by, dx, dy = cx, ay, ax, cy
                box = np.array([[ax, ay], [bx, by], [cx, cy], [dx, dy]])
                boxes.append(box)
                confs.append(conf)
                labels.append(idx)

        return boxes, confs, labels
