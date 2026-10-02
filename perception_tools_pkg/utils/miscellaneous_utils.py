import cv2
import numpy as np


def get_bb(image, task: str = "hi"):
    x, y, w, h = cv2.selectROI(task, image, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow(task)
    input_box = np.array([x, y, x + w, y + h])
    print(input_box)
    return input_box
