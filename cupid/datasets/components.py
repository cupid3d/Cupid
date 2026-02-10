from typing import *
from abc import abstractmethod
import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from ..utils.pose_utils import camera_parameters_from_frame_data


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
        min_aesthetic_score (float): minimum aesthetic score for the dataset
        is_eval (bool): whether to evaluate all the views of the dataset
        max_num_retry (int): maximum number of retries for each instance when
            exception occurs, for example, no valid coords for instance and view
        instances_sha256 (List[str]): list of sha256 of the instances to be included in the dataset
    """

    def __init__(self,
        roots: str,
        min_aesthetic_score: float = 5.0,
        is_eval: bool = False,
        max_num_retry: int = 10,
        instances_sha256: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__()
        self.roots = roots.split(',')
        self.instances = []
        self.metadata = pd.DataFrame()
        self.min_aesthetic_score = min_aesthetic_score
        self.is_eval = is_eval
        self.max_num_retry = max_num_retry
        self.instances_sha256 = instances_sha256
        
        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
            self._stats[key]['Total'] = len(metadata)
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)
            self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
            metadata.set_index('sha256', inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])

        # filter instances by sha256
        if self.instances_sha256 is not None:
            self.instances = [instance for instance in self.instances if instance[1] in self.instances_sha256]
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    def get_instance(self, root: str, instance: str, **kwargs) -> Dict[str, Any]:
        return {}
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        num_retry = 0
        while True:
            try:
                root, instance = self.instances[index]
                return self.get_instance(root, instance)
            except Exception as e:
                print(e)
                num_retry += 1
                if num_retry >= self.max_num_retry:
                    raise e
                index = np.random.randint(0, len(self))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class TextConditionedMixin:
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        self.captions = {}
        for instance in self.instances:
            sha256 = instance[1]
            self.captions[sha256] = json.loads(self.metadata.loc[sha256]['captions'])
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['captions'].notna()]
        stats['With captions'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance, **kwargs):
        pack = super().get_instance(root, instance, **kwargs)
        text = np.random.choice(self.captions[instance])
        pack['cond'] = text
        return pack
    
    
class ImageConditionedMixin:
    def __init__(
        self, 
        roots, 
        *, 
        image_size : int = 518, 
        crop_size_ratio : Optional[Union[float, Tuple[float, float]]] = None,
        crop_prob : float = 1.0,
        crop_in_eval : bool = True,
        crop_num_discretization_buckets: Optional[int] = None,
        crop_allow_padding: bool = False,
        **kwargs
    ):
        self.image_size = image_size
        self.crop_size_ratio = crop_size_ratio
        self.crop_prob = crop_prob
        self.crop_in_eval = crop_in_eval
        self.crop_num_discretization_buckets = crop_num_discretization_buckets
        self.crop_allow_padding = crop_allow_padding
        self.override_buekct_idx = None
        self.last_crop_ratio = None
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats

    def get_crop_bbox(self, image: Image.Image) -> Tuple[int, int, int, int]:
        if isinstance(self.crop_size_ratio, float):
            ratio = self.crop_size_ratio
        elif isinstance(self.crop_size_ratio, (tuple, list)):
            min_ratio, max_ratio = self.crop_size_ratio[:2]
            if self.is_eval:  # Decay to midpoint for eval mode
                ratio = (min_ratio + max_ratio) / 2
            elif len(self.crop_size_ratio) == 2:  # Uniform distribution
                ratio = np.random.uniform(min_ratio, max_ratio)
            elif len(self.crop_size_ratio) == 3:  # Beta distribution
                k = self.crop_size_ratio[2]
                ratio = min_ratio + (max_ratio - min_ratio) * np.random.beta(k, k)
            else:
                raise ValueError(f'Invalid crop_size_ratio tuple len: {self.crop_size_ratio}')

            if self.crop_num_discretization_buckets is not None:
                if self.override_buekct_idx is not None:
                    bucket_idx = self.override_buekct_idx
                else:
                    bucket_t = (ratio - min_ratio) / (max_ratio - min_ratio)
                    bucket_idx = int(bucket_t * self.crop_num_discretization_buckets)
                    bucket_idx = min(max(bucket_idx, 0), self.crop_num_discretization_buckets - 1)
                bucket_t = (bucket_idx + 0.5) / self.crop_num_discretization_buckets
                ratio = min_ratio + (max_ratio - min_ratio) * bucket_t
        else:
            raise ValueError(f'Invalid crop_size_ratio: {self.crop_size_ratio}')
        self.last_crop_ratio = ratio

        assert image.mode == 'RGBA', 'Image must be in RGBA mode for cropping to work'
        width, height = image.size
        ys, xs = np.array(image.getchannel(3)).nonzero()
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())

        cx = int(round((x0 + (x1 + 1)) / 2))
        cy = int(round((y0 + (y1 + 1)) / 2))
        obj_hsize = 0.5 * max(x1 - x0 + 1, y1 - y0 + 1)
        aug_hsize = int(round(obj_hsize * ratio))
        if not self.crop_allow_padding:
            max_hsize = min(cx, cy, width - cx, height - cy)
            aug_hsize = min(aug_hsize, max_hsize)
        aug_size = aug_hsize * 2

        if not self.crop_allow_padding:
            left = max(0, min(cx - aug_hsize, width - aug_size))
            top  = max(0, min(cy - aug_hsize, height - aug_size))
        else:
            left = cx - aug_hsize
            top  = cy - aug_hsize
        right  = left + aug_size
        bottom = top + aug_size
        
        return (left, top, right, bottom)
    
    def crop_image(self, image: Image.Image, depth_image: Optional[Image.Image], pack: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Image.Image]:
        image_size = torch.tensor(image.size, dtype=torch.float32)
        crop_bbox = self.get_crop_bbox(image)
        image = image.crop(crop_bbox)
        # if self.return_depth_map:
        #    depth_image = depth_image.crop(crop_bbox)
        crop_bbox = torch.tensor(crop_bbox, dtype=torch.float32).reshape(2, 2)
        crop_bbox = crop_bbox / image_size

        if 'uv_volume' in pack:
            bbox_min = crop_bbox[0][:, None, None, None]
            bbox_max = crop_bbox[1][:, None, None, None]
            crop_uvs = (pack['uv_volume'] - bbox_min) / (bbox_max - bbox_min)
            pack['uv_volume'] = crop_uvs.clamp_(0.0, 1.0)

        if 'uvs' in pack:
            bbox_min = crop_bbox[0][None, :]
            bbox_max = crop_bbox[1][None, :]
            crop_uvs = (pack['uvs'] - bbox_min) / (bbox_max - bbox_min)
            pack['uvs'] = crop_uvs.clamp_(0.0, 1.0)

        if 'ssuv' in pack and self.const_ssuv == 'crop':
            in_image_mask = torch.logical_and(pack['uv_volume'] >= 0.01, pack['uv_volume'] <= 0.99).all(dim=0)
            ssuv = torch.zeros(1, self.resolution, self.resolution, self.resolution, dtype=torch.int)
            ssuv[:, in_image_mask] = 1
            pack['ssuv'] = ssuv

        if 'intrinsics' in pack:
            intrinsics = pack.pop('intrinsics')
            # if self.return_raw_cond:
                # pack['raw_intrinsics'] = intrinsics.clone()
            scale = crop_bbox[1] - crop_bbox[0]
            intrinsics[0,0] = intrinsics[0,0] / scale[0]
            intrinsics[0,2] = (intrinsics[0,2] - crop_bbox[0, 0]) / scale[0]
            intrinsics[1,1] = intrinsics[1,1] / scale[1]
            intrinsics[1,2] = (intrinsics[1,2] - crop_bbox[0, 1]) / scale[1]
            pack['intrinsics'] = intrinsics

        pack['crop_bbox'] = crop_bbox

        return image, depth_image

    def get_image_cond(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        return image
    

    def get_instance(self, root, instance, frame_data=None, **kwargs):
        image_root = os.path.join(root, 'renders_cond', instance)
        if frame_data is None:
            with open(os.path.join(image_root, 'transforms.json')) as f:
                metadata = json.load(f)
                view_idx = np.random.randint(len(metadata['frames']))
                frame_data = metadata['frames'][view_idx]
        if 'render_dir' in frame_data:
            image_root = frame_data['render_dir']

        pack = super().get_instance(root, instance, frame_data=frame_data, **kwargs)
        
        image_path = os.path.join(image_root, frame_data['file_path'])
        image = Image.open(image_path)
        depth_image = None

        if self.crop_size_ratio is not None and \
            (not self.is_eval or self.crop_in_eval) and \
            (self.is_eval or np.random.rand() < self.crop_prob):
            image, depth_image = self.crop_image(image, depth_image, pack)
        
        pack['cond'] = self.get_image_cond(image)
       
        return pack
    

class TransformConditionedMixin:
    """
    Mixin for datasets that are conditioned on camera transforms.
    
    Args:
        num_ref_views (int): number of reference views to use per instance
        num_cond_views (int): number of conditional views to use per instance
        eval_view_sub_sample (int): split number of views to subsample for evaluation
        eval_view_sub_sample_offset (int): offset for evaluation view subsampling
    """
    def __init__(self, 
        roots, 
        *, 
        num_ref_views: int = 0, 
        num_cond_views: int = 24, 
        eval_view_sub_sample: int = 1,
        eval_view_sub_sample_offset: int = 0,
        **kwargs
    ):
        self.num_ref_views = num_ref_views
        self.num_cond_views = num_cond_views
        self.eval_view_sub_sample = eval_view_sub_sample
        self.eval_view_sub_sample_offset = eval_view_sub_sample_offset
        self.eval_total_views = (self.num_ref_views + self.num_cond_views) // self.eval_view_sub_sample
        super().__init__(roots, **kwargs)

    def _get_frame_data(self, root, instance, view_idx):
        if view_idx < self.num_cond_views:
            with open(os.path.join(root, 'renders_cond', instance, 'transforms.json')) as f:
                metadata = json.load(f)
            frame_data = metadata['frames'][view_idx]
            frame_data['render_dir'] = os.path.join(root, 'renders_cond', instance)
        else:
            with open(os.path.join(root, 'renders', instance, 'transforms.json')) as f:
                metadata = json.load(f)
            frame_data = metadata['frames'][view_idx - self.num_cond_views]
            frame_data['render_dir'] = os.path.join(root, 'renders', instance)
        return frame_data

    def get_instance(self, root, instance, frame_data, **kwargs):
        """Get base data, and camera transforms for an instance."""
        pack = super().get_instance(root, instance, frame_data=frame_data, **kwargs)
        extrinsic, intrinsic = camera_parameters_from_frame_data(frame_data)
        pack['intrinsics'] = intrinsic
        pack['extrinsics'] = extrinsic
        return pack

    def get_instance_view_index(self, index):
        assert self.is_eval, "get_instance_view_index should be only used in eval mode"
        instance_idx, view_idx = divmod(index, self.eval_total_views)
        view_idx = view_idx * self.eval_view_sub_sample + self.eval_view_sub_sample_offset
        return instance_idx, view_idx

    def __getitem__(self, index):
        if self.is_eval:
            instance_idx, view_idx = self.get_instance_view_index(index)
            root, instance = self.instances[instance_idx]
            frame_data = self._get_frame_data(root, instance, view_idx)
            return self.get_instance(root, instance, frame_data=frame_data)
        else:
            num_retry = 0
            while True:
                try:
                    root, instance = self.instances[index]
                    view_idx = np.random.randint(self.num_ref_views + self.num_cond_views)
                    frame_data = self._get_frame_data(root, instance, view_idx=view_idx)
                    return self.get_instance(root, instance, frame_data=frame_data)
                except Exception as e:
                    print(e)
                    num_retry += 1
                    if num_retry >= self.max_num_retry:
                        raise e
                    index = np.random.randint(0, len(self))

    def __len__(self):
        if self.is_eval:
            return len(self.instances) * self.eval_total_views
        else:
            return len(self.instances)

