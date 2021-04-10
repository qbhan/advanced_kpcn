import argparse
import os
import subprocess
# import bpy

blender_path = 'C:/Users/qbhan/Desktop/blender-git/build_windows_x64_vc16_Release/bin/Release/blender.exe'
blender_scenes = 'C:/Users/qbhan/blender_scenes'
# scene_path = 'C:/Users/qbhan/Downloads/hot-dogs.blend'
script_path = 'C:/Users/qbhan/Desktop/blender-git/build_windows_x64_vc16_Release/bin/Release/2.93/scripts/scenegen/render.py'

def blender_files():
    scene_list = os.listdir(blender_scenes)
    new_scene_list = []
    for scene in scene_list:
        ext = scene.split('.')[-1]
        if ext == 'blend':
            new_scene_list.append(scene)
    return new_scene_list

def run_cmd():
    # os.system('cd ../../../')
    scene_list = blender_files()
    print(scene_list)
    for scene in scene_list:
        scene_path = blender_scenes + '/' + scene
        scenename = scene.split('.')[0]
        cmd = blender_path + ' -b ' + scene_path + ' -P ' + script_path + ' -- ' + scenename
        print(cmd)
        subprocess.call(cmd)

def run_cmd_times():
    scene_path = blender_scenes + '/' + 'hotdogs.blend'
    scenename = 'hotdogs'
    for i in range(10):
        cmd = blender_path + ' -b ' + scene_path + ' -P ' + script_path + ' -- ' + scenename +'_' + str(i+1)
        print(cmd)
        subprocess.call(cmd)


def run_cmd_single():
    scene_path = blender_scenes + '/' + 'pontiac.blend'
    scenename = 'pontiac'
    cmd = blender_path + ' -b ' + scene_path + ' -P ' + script_path + ' -- ' + scenename
    print(cmd)
    subprocess.call(cmd)


if __name__=="__main__":
    # run_cmd()
    run_cmd_single()
    # print(bpy.data.scenes["Scene"].matlib)
    # run_cmd_times()