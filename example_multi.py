"""
Example: Multi-Object 3D Reconstruction with Cupid3D
"""
import os
import glob

os.environ['SPCONV_ALGO'] = 'native'  # Skip benchmarking for one-off runs

from cupid.pipelines import Cupid3DPipeline
from cupid.utils.align_utils import Aligner
from cupid.utils import sample_utils


# Load pipeline
pipeline = Cupid3DPipeline.from_pretrained("hbb1/Cupid")
pipeline.cuda()

# Load image and segmentation masks
image = sample_utils.load_image("assets/real_scenes/pexels-karolina-grabowska-7193650.jpg")
masks = [sample_utils.load_image(p) for p in sorted(glob.glob("assets/multi_no_occ/pexels-anna-nekrashevich-7214602/segmentation_*.png"))]

# Run 3D reconstruction for each object
objects = [pipeline.run(image, mask=mask) for mask in masks]

# Align and export individual meshes
aligner = Aligner(image, objects=objects)
aligner.export_meshes("result")

# Compose all meshes into a single file (optional)
Aligner.compose_mesh_from_metadata(
    meta_file="result/metadata.json",
    output_path="result/composed.glb"
)

# Optional: convert to Blender file (with camera)
# python convert_blender.py --meta_file result/metadata.json --output_path result/scene.blend --save_file