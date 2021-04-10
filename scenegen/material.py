import bpy
import random
import sys
sys.path.append('C:/Users/qbhan/Desktop/blender-git/build_windows_x64_vc16_Release/bin/Release/2.93/scripts/scenegen')
from rand_util import *


global C, D

def jitter_rgb(node):
    if node.name == "RGB":
        r, g, b, a = node.outputs[0].default_value[0:4]
        print("Original color : {}, {}, {}, {}".format(r,g,b,a))
        node.outputs[0].default_value = (rand_jitter_clip(r, 0.1, 0, 1), rand_jitter_clip(g, 0.1, 0, 1), rand_jitter_clip(b, 0.1, 0, 1), rand_jitter_clip(a, 0.1, 0, 1))
        print(node.outputs[0].default_value[0:4])

def rand_mat_case(node):
    
    # BSDFs
    if node.name == 'Diffuse BSDF':
        pass
        node.inputs[0].default_value = pos_rand_4d()    # color
        node.inputs[1].default_value = random.random()

    elif node.name == 'Glossy BSDF' :
        pass
        # node.inputs[1].default_value = random.random()
        # node.distribution

    elif node.name == 'Glass BSDF':
        pass
        if len(node.inputs[0].links) == 0:  # color
            node.inputs[0].default_value = pos_rand_4d()    
        # node.inputs[1].default_value = random.random()
        # node.inputs[2].default_value = random.random()
        # node.distribution

    elif node.name == 'Anisotropic BSDF':
        pass
        if len(node.inputs[0].links) == 0:  # color
            node.inputs[0].default_value = pos_rand_4d()
        node.inputs[1].default_value = random.random()  # roughness
        node.inputs[2].default_value = random.random()  # anisotropy
        node.inputs[3].default_value = random.random()  # rotation

    elif node.name == 'Velvet BSDF':
        pass
        if len(node.inputs[0].links) == 0:  # color
            node.inputs[0].default_value = pos_rand_4d()
        node.inputs[1].default_value = random.random()  # sigma

    elif node.name == 'Principled BSDF':
        pass
        if len(node.inputs[0].links) == 0:  # color
            node.inputs[0].default_value = pos_rand_4d()
        node.inputs[1].default_value = random.random()  # subsurface
        # node.inputs[2].default_value = random.random()  # subsurface radius
        node.inputs[3].default_value = pos_rand_4d()    # subsurface color
        node.inputs[4].default_value = random.random()  # metallic
        node.inputs[5].default_value = random.random()  # specular
        node.inputs[6].default_value = random.random()  # specular tint
        node.inputs[7].default_value = random.random()  # roughness
        node.inputs[8].default_value = random.random()  # anisotropic
        node.inputs[9].default_value = random.random()  # anisotropic rotation
        node.inputs[10].default_value = random.random()  # sheen
        node.inputs[11].default_value = random.random()  # sheen
        node.inputs[12].default_value = random.random()  # clearcoat
        node.inputs[13].default_value = random.random()  # clearcoat roughness
        node.inputs[14].default_value = random.random() * 2 # IOR
        node.inputs[15].default_value = random.random()  # transmission
        node.inputs[16].default_value = random.random()  # transmission roughness
        node.inputs[17].default_value = pos_rand_4d()  # emission
        node.inputs[18].default_value = random.random()  # alpha
        

    # Operators
    elif node.name == 'Mix Shader' :
        pass
        if len(node.inputs[0].links) == 0:
            node.inputs[0].default_value = random.random()  # fac (percentage to mix)

    elif node.name == 'Layer Weight':
        pass
        node.inputs[0].default_value = random.random()  # blender

    elif node.name == 'Mix':
        pass

    elif node.name == 'Add':
        pass

    elif node.name == 'Multiply':
        pass

    # Single Entities
    elif node.name == 'Gamma':
        pass
        node.inputs[1].default_value = random.random()

    elif node.name == 'Hue Saturation Value':
        pass
        node.inputs[0].default_value = random.random()  # hue
        node.inputs[1].default_value = random.random() * 10 # saturation
        node.inputs[2].default_value = random.random()  # value
        node.inputs[3].default_value = random.random()  # fac
        if len(node.inputs[4].links) == 0:  # color
            node.inputs[4].default_value = pos_rand_4d()   

    elif node.name == 'RGB':
        pass
        node.outputs[0].default_value = pos_rand_4d()
        # print(node.inputs[0].default_value)

    elif node.name == 'Invert':
        pass
        node.inputs[0].default_value = random.random()  # fac
        if len(node.inputs[1].links) == 0:  # color
            node.inputs[1].default_value = pos_rand_4d()


    elif node.name == 'ColorRamp':
        pass

    elif node.name == 'Attribute':
        pass

    elif node.name == 'Geometry':
        pass

    elif node.name == 'Emission':
        pass
        node.inputs[0].default_value = pos_rand_4d()    # color
        node.inputs[1].default_value = random.random()

    else:
        pass


def rand_mat_new(slot):
    new_mat = D.materials.new(name="new_mat_" + "i")
    new_mat.diffuse_color = pos_rand_4d()
    new_mat.specular_color = pos_rand_3d()
    slot.material = new_mat
    

def traverse_nodes(node_in):
    # print(len(node_in.inputs))
    for n_inputs in node_in.inputs:
        # print(len(n_inputs.links))
        for node_links in n_inputs.links:
            # print("going to " + node_links.from_node.name)
            # rand_mat_case(node_links.from_node)
            jitter_rgb(node_links.from_node)
            traverse_nodes(node_links.from_node)


def rand_mat(D):
    
    objects = D.objects
    # print(objects[0:61])
    for obj in objects:
        i = 0
        for slot in obj.material_slots:
            print(slot.material.node_tree.nodes[0:-1])
            for mat_node in slot.material.node_tree.nodes:
                if mat_node.type == 'OUTPUT_MATERIAL':
                    print("Starting at " + mat_node.name + ' for material ' + slot.material.name)
                    traverse_nodes(mat_node)
            

def rand_mat_lib(D):
    for obj in D.objects:
        obj.select_set(True)
        D.scenes["Scene"].matlib.mat_index = random.randrange(32)
        bpy.ops.matlib.operator(cmd="APPLY")
        obj.select_set(False)