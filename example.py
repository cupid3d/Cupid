"""
Example: Single Object 3D Reconstruction with Cupid3D
"""

import os
os.environ['SPCONV_ALGO'] = 'native'  # Use 'auto' for faster repeated runs

import imageio
import numpy as np
from PIL import Image
from cupid.pipelines import Cupid3DPipeline
from cupid.utils import render_utils, sample_utils
from cupid.utils.align_utils import save_mesh


# Load pipeline
pipeline = Cupid3DPipeline.from_pretrained("hbb1/Cupid")
pipeline.cuda()

# Load input image and run reconstruction
image = sample_utils.load_image("assets/example_image/typical_creature_dragon.png")
outputs = pipeline.run(image)

# outputs contains:
#   - 'gaussian': 3D Gaussians
#   - 'radiance_field': radiance fields
#   - 'mesh': meshes
#   - 'pose': camera extrinsic and intrinsic parameters

# Save side-by-side comparison (input vs rendered)
render_rgb = render_utils.render_pose(outputs['gaussian'][0], outputs['pose'][0])['color'][0]
input_rgb = Image.alpha_composite(Image.new("RGBA", image.size, (0, 0, 0, 255)), image)
input_rgb = np.array(input_rgb.resize((512, 512), Image.Resampling.LANCZOS).convert('RGB'))
imageio.imwrite('sample.png', np.concatenate([input_rgb, render_rgb], axis=1))

# Save mesh and camera pose
save_mesh(
    all_outputs=outputs,
    poses=outputs.pop('pose'),
    output_dir='output'
)

# Optional: convert to Blender file
# python convert_blender.py --meta_file output/metadata.json --output_path output/scene.blend --save_file

# Optional: render turntable video
# video_rgb = render_utils.render_video(outputs['gaussian'][0])['color']
# video_normal = render_utils.render_video(outputs['mesh'][0])['normal']
# video_combined = [np.concatenate([r, n], axis=1) for r, n in zip(video_rgb, video_normal)]
# imageio.mimsave('sample.mp4', video_combined, fps=30)
