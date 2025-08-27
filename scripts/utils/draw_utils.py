from PIL import Image, ImageDraw, ImageFont
from PIL import Image
import numpy as np
from typing import List
import cv2
from numpy.typing import ArrayLike, NDArray


def draw_annotated_image_box(
        image: Image.Image,
        predicted_domain: str,
        box: ArrayLike
) -> Image.Image:
    image = image.convert('RGB')
    screenshot_img_arr = np.asarray(image)
    screenshot_img_arr = np.flip(screenshot_img_arr, -1)
    screenshot_img_arr = screenshot_img_arr.astype(np.uint8)

    if box is not None:
        cv2.rectangle(screenshot_img_arr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (69, 139, 0), 2)
        cv2.putText(screenshot_img_arr, 'Predicted phishing target: '+ predicted_domain, (int(box[0]), int(box[3])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
    else:
        cv2.putText(screenshot_img_arr, 'Predicted phishing target: ' + predicted_domain, (int(10), int(10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2)
    screenshot_img_arr = np.flip(screenshot_img_arr, -1)
    image = Image.fromarray(screenshot_img_arr)
    return image
