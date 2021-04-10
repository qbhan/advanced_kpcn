import bpy
import random

def pos_rand_3d():
    return (random.random(), random.random(), random.random())

def pos_rand_4d():
    return (random.random(), random.random(), random.random(), random.random())

def tuple_scale(vec, n):
    return map(lambda n, x: n * x, n , vec)
    
def rand_3d():
    return (random.random() - 0.5, random.random() - 0.5, random.random() - 0.5)

def rand_4d():
    return (random.random() - 0.5, random.random() - 0.5, random.random() - 0.5, random.random() - 0.5)

def rand_jitter(s):
    return (random.random() - 0.5) * 2 * s

def rand_jitter_clip(v, s, lb, ub):
    ret = v + rand_jitter(s)
    if ret >= lb and ub >= ret:
        return ret
    elif ret < lb:
        return lb
    else:
        return ub