from dataclasses import dataclass
from typing import Tuple, Dict
import torch
from PIL import Image


@dataclass
class CropParameters:
    """Parameters describing how an image was cropped."""
    fov_scale: float
    cx_offset: float
    cy_offset: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.fov_scale, self.cx_offset, self.cy_offset)

    @classmethod
    def identity(cls) -> 'CropParameters':
        return cls(fov_scale=1.0, cx_offset=0.0, cy_offset=0.0)


@dataclass
class ProcessedImage:
    """Processed image and associated data."""
    image: Image.Image
    crop_params: CropParameters

    @classmethod
    def from_image(cls, image: Image.Image) -> 'ProcessedImage':
        return cls(image=image, crop_params=CropParameters.identity())


@dataclass
class CameraPose:
    """Camera pose parameters."""
    extrinsic: torch.Tensor  # 4x4 matrix
    intrinsic: torch.Tensor  # 3x3 matrix

    def de_crop(self, crop_params: CropParameters) -> 'CameraPose':
        """Adjust the camera pose to account for image cropping."""
        scale, x_offset, y_offset = crop_params.as_tuple()
        intrinsic = self.intrinsic.clone()
        intrinsic[0, 0] *= scale
        intrinsic[1, 1] *= scale
        intrinsic[0, 2] = intrinsic[0, 2] * scale + x_offset
        intrinsic[1, 2] = intrinsic[1, 2] * scale + y_offset
        return CameraPose(extrinsic=self.extrinsic, intrinsic=intrinsic)

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            'extrinsic': self.extrinsic,
            'intrinsic': self.intrinsic,
        }