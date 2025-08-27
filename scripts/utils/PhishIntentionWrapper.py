from ..phishintention.modules.awl_detector import pred_rcnn, vis, find_element_type
from ..phishintention.modules.logo_matching import ocr_main, l2_norm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from typing import Tuple, Union
from numpy.typing import ArrayLike, NDArray

class LayoutDetector(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, screenshot_path: str) -> Tuple[NDArray, NDArray]:
        # Run detection with RCNN predictor
        pred_boxes, pred_classes, _ = pred_rcnn(
            im=screenshot_path,
            predictor=self.predictor
        )
        pred_boxes = pred_boxes.numpy()
        pred_classes = pred_classes.numpy()
        return pred_boxes, pred_classes

class LogoDetector(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, screenshot_path: str) -> NDArray:
        # Run detection with RCNN predictor
        pred_boxes, pred_classes, _ = pred_rcnn(
            im=screenshot_path,
            predictor=self.predictor
        )

        # Filter to "logo" class
        logo_pred_boxes, _ = find_element_type(
            pred_boxes, pred_classes, bbox_type="logo"
        )
        logo_pred_boxes = logo_pred_boxes.numpy()
        return logo_pred_boxes

class LogoEncoder(nn.Module):
    def __init__(self, siamese_model, ocr_model, matching_threshold, img_size: int = 224):
        super().__init__()
        self.siamese_model = siamese_model
        self.ocr_model = ocr_model
        self.img_size = img_size
        self.matching_threshold = matching_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transformation pipeline
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def preprocess_image(self, img: Union[str, Image.Image]) -> Image.Image:
        img = Image.open(img) if isinstance(img, str) else img
        img = img.convert("RGBA").convert("RGB")
        # Pad to square
        pad_color = (255, 255, 255)
        img = ImageOps.expand(
            img,
            (
                (max(img.size) - img.size[0]) // 2,
                (max(img.size) - img.size[1]) // 2,
                (max(img.size) - img.size[0]) // 2,
                (max(img.size) - img.size[1]) // 2,
            ),
            fill=pad_color,
        )
        # Resize
        img = img.resize((self.img_size, self.img_size))
        return img

    def forward(self, img: Image.Image) -> NDArray:
        img = self.preprocess_image(img)

        ocr_emb = ocr_main(image_path=img, model=self.ocr_model, height=None, width=None)[0]
        ocr_emb = ocr_emb[None, ...].to(self.device)
        img_tensor = self.img_transforms(img)[None, ...].to(self.device)
        logo_feat = self.siamese_model.features(img_tensor, ocr_emb)

        # L2 normalize
        logo_feat = l2_norm(logo_feat).squeeze(0).detach().cpu().numpy()
        return logo_feat