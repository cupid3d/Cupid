import argparse
import sys
import blendertoolbox as bt 
import bpy
import os
import numpy as np
import json
from pathlib import Path


def main(args):
    outputPath = os.path.abspath(args.output_path)

    with open(args.meta_file, 'r') as f:
        meta_data = json.load(f)

    imgRes_x = args.resolution[0]
    imgRes_y = args.resolution[1]
    numSamples = args.num_samples
    exposure = 1.0
    
    bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

    scene = bpy.context.scene
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.sequencer_colorspace_settings.name = 'sRGB'

    location = (0, 0, 0)
    rotation = (0, 0, 0)
    scale = (1, 1, 1)
    
    meshes = []
    first_extrinsic = None
    z_min = 100
    
    for i, _ in enumerate(meta_data['glb_path']):
        meshPath = (Path(args.meta_file).parent / f'mesh{i}.glb').as_posix()
        mesh_obj = bt.readMesh(meshPath, location, rotation, scale)

        extrinsic = np.array(meta_data['pose'][i]['extrinsic'])
        if 'model_scale' in meta_data['pose'][i]:
            mesh_scale = np.array(meta_data['pose'][i]['model_scale'])
        else:
            mesh_scale = 1.0

        verts_transformed = np.array([v.co for v in mesh_obj.data.vertices])
        verts_transformed = verts_transformed * mesh_scale
        
        if i == 0:
            first_extrinsic = extrinsic
        else:
            extrinsic = np.linalg.inv(first_extrinsic) @ extrinsic
            verts_transformed = verts_transformed @ extrinsic[:3, :3].T + extrinsic[:3, 3:4].T
            
        z_min = min(verts_transformed[:, -1].min(), z_min)

        for j, v in enumerate(mesh_obj.data.vertices):
            v.co = verts_transformed[j]

        meshes.append(mesh_obj)

    intrinsic = np.array(meta_data['pose'][0]['intrinsic'])[None]
    extrinsic = first_extrinsic[None]
    
    bpy.ops.object.shade_smooth() 

    cam = bt.setCamera_from_extrinsic_intrinsic(extrinsic[0], intrinsic[0], imgRes_x, imgRes_y)

    if args.creat_cam_vis:
        # visualize the camera frustums
        if extrinsic.ndim == 3 and intrinsic.ndim == 3:
            for i in range(extrinsic.shape[0]):
                color = (np.random.rand(), np.random.rand(), np.random.rand(), 1.0)
                bt.create_camera_visualization(extrinsic[i], intrinsic[i], scale=0.25, name=f"CameraVisualization_{i}", color=color)
        else:
            bt.create_camera_visualization(extrinsic[0], intrinsic[0], scale=0.25)

    for mesh in meshes:
        bt.setMat_unlit(mesh, color_type='texture')
    
    bt.invisibleGround(shadowBrightness=0.9, location=(0, 0, z_min))

    lightAngle = (6, -30, -155) 
    strength = 2
    shadowSoftness = 0.3
    sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

    bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1)) 

    bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')

    if args.save_file:
        bpy.ops.wm.save_mainfile(filepath=os.path.dirname(outputPath) + '/scene.blend')


    # bt.renderImage(outputPath, cam)
    # you can define trajectory and render video here
    # if extrinsic.ndim == 3 and intrinsic.ndim == 3:
    #     for i in range(extrinsic.shape[0]):
    #         cam = bt.setCamera_from_extrinsic_intrinsic(extrinsic[i], intrinsic[i], imgRes_x, imgRes_y)
    #         output_path_i = os.path.splitext(outputPath)[0] + f'_{i:03d}.png'
    #         bt.renderImage(output_path_i, cam)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a mesh using BlenderToolbox.')
    parser.add_argument('--meta_file', type=str, required=True, help='Path to the metadata JSON file.')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the rendered image.')
    parser.add_argument('--resolution', type=int, nargs=2, default=[512, 512], help='Image resolution (width height).')
    parser.add_argument('--num_samples', type=int, default=300, help='Number of samples for rendering.')
    parser.add_argument('--save_file', action='store_true', help='Whether to save the Blender file.')
    parser.add_argument('--creat_cam_vis', action='store_true', help='Whether to create camera visualization.')

    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = str(Path(args.meta_file).parent / 'unlit.png')
    main(args)