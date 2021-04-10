import bpy
import argparse
import sys
import random
import mathutils
# import matlib

sys.path.append('C:/Users/qbhan/Desktop/blender-git/build_windows_x64_vc16_Release/bin/Release/2.93/scripts/scenegen')
# import rand_util
from material import *


# blender 
# -b C:\Users\qbhan\Downloads\hot-dogs.blend 
# -P 2.93\scripts\scenegen\render.py

# TODO
# 1. Make file path respect to the input blender file (might use arguments)
# 2. Modulatation
# 3. File for camera settings
# 4. Material settings
# 5. Change feature file names

def update_camera(camera, focus_point=mathutils.Vector((0.0, 0.0, 0.0)), distance=1.0):
    """
    Focus the camera to a focus point and place the camera at a specific distance from that
    focus point. The camera stays in a direct line with the focus point.

    :param camera: the camera object
    :type camera: bpy.types.object
    :param focus_point: the point to focus on (default=``mathutils.Vector((0.0, 0.0, 0.0))``)
    :type focus_point: mathutils.Vector
    :param distance: the distance to keep to the focus point (default=``10.0``)
    :type distance: float
    """
    looking_direction = camera.location - focus_point
    rot_quat = looking_direction.to_track_quat('Z', 'Y')

    # rot_quat.

    camera.rotation_euler = rot_quat.to_euler()
    # Use * instead of @ for Blender <2.8
    camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, distance))


def rand_cam(active_cam):
    
    # Location
    loc = active_cam.location[0:3]
    d_loc = map(lambda x: 0.5 * x, rand_3d())
    # print(d_loc)
    active_cam.location = tuple(map(lambda i, j: i - j, loc, d_loc))

    # Rotation
    rot = active_cam.rotation_euler[0:3]
    print(rot[0:3])
    d_rot = map(lambda x: 0.05 * x, rand_3d())
    # active_cam.rotation_euler = tuple(map(lambda i, j: i - j, rot, d_rot))

    # update_camera(active_cam)

    # DOF settings
    # dof_set = active_cam.data.dof

    # FOV
    # print(active_cam)
    # dof_set.angle_x
    # dof_set.angle_y
        

# change scene names according to the blender file
# scenename = sys.argv[-1]
scenename = 'pontiac'

noisy_base_path = 'C:/Users/qbhan/rendered/' + scenename + '_noisy/'
gt_base_path = 'C:/Users/qbhan/rendered/' + scenename + '_gt/'

# variables
C, D = bpy.context, bpy.data
computation_type = 'CUDA'
gpu_id = (0,)

# Select the computing device.
C.scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
devices = prefs.get_devices()
if computation_type == 'CUDA':
    bpy.context.scene.cycles.device = 'GPU'
    prefs.compute_device_type = 'CUDA'
for i, gpu in enumerate(devices[0]):
    gpu.use = (i in gpu_id)

# Turn off OptiX Denoising
D.scenes["Scene"].cycles.use_denoising = False

# Samples Per Pixels
D.scenes["Scene"].cycles.samples = 16
D.scenes["Scene"].cycles.max_bounces = 5

scene = bpy.context.scene
scene.render.resolution_x = 1280
scene.render.resolution_y = 720

# rand_cam(D.objects["Camera"])
# rand_mat_lib(D)
rand_mat(D)
# print(C.scene.matlib)

tree = bpy.context.scene.node_tree
links = tree.links

# Clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)
 
C.scene.render.image_settings.file_format = 'OPEN_EXR'
C.scene.render.image_settings.color_depth = '32'

# Create input view layer node.
view_layers = tree.nodes.new('CompositorNodeRLayers')

# Denoising Normal
normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = 'Normal Output'
links.new(view_layers.outputs['Denoising Normal'], normal_file_output.inputs[0])
normal_file_output.format.file_format = "OPEN_EXR"

# Denoisng Depth
depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
links.new(view_layers.outputs['Denoising Depth'], depth_file_output.inputs[0])
depth_file_output.format.file_format = "OPEN_EXR"

# Denoising Albedo
albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = 'Albedo Output'
links.new(view_layers.outputs['Denoising Albedo'], albedo_file_output.inputs[0])
albedo_file_output.format.file_format = "OPEN_EXR"

# Denoising Shadowing
shadow_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
shadow_file_output.label = 'Shadowing Output'
links.new(view_layers.outputs['Denoising Shadowing'], shadow_file_output.inputs[0])
shadow_file_output.format.file_format = "OPEN_EXR"

# Denoising Variance
variance_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
variance_file_output.label = 'Variance Output'
links.new(view_layers.outputs['Denoising Variance'], variance_file_output.inputs[0])
variance_file_output.format.file_format = "OPEN_EXR"

# Denoising Intensity
intensity_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
intensity_file_output.label = 'Intensity Output'
links.new(view_layers.outputs['Denoising Intensity'], intensity_file_output.inputs[0])
intensity_file_output.format.file_format = "OPEN_EXR"

# File path for train noisy
normal_file_output.base_path = str(noisy_base_path + "normal/")
depth_file_output.base_path = str(noisy_base_path + 'depth/')
albedo_file_output.base_path = str(noisy_base_path + 'albedo/')
shadow_file_output.base_path = str(noisy_base_path + 'shadow/')
variance_file_output.base_path = str(noisy_base_path + 'variance/')
intensity_file_output.base_path = str(noisy_base_path + 'intensity/')
D.scenes[0].render.filepath = noisy_base_path + scenename

# Render & Write files
bpy.ops.render.render(write_still=True)

# File path for train gt
D.scenes["Scene"].cycles.samples = 128
normal_file_output.base_path = str(gt_base_path + "normal/")
depth_file_output.base_path = str(gt_base_path + 'depth/')
albedo_file_output.base_path = str(gt_base_path + 'albedo/')
shadow_file_output.base_path = str(gt_base_path + 'shadow/')
variance_file_output.base_path = str(gt_base_path + 'variance/')
intensity_file_output.base_path = str(gt_base_path + 'intensity/')
D.scenes[0].render.filepath = gt_base_path + scenename
bpy.ops.render.render(write_still=True)
# print(noisy_base_path)