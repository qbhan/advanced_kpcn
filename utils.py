import torch
import sys
import gc


def to_torch_tensors(data):
  if isinstance(data, dict):
    for k, v in data.items():
      if not isinstance(v, torch.Tensor):
        data[k] = torch.from_numpy(v)
  elif isinstance(data, list):
    for i, v in enumerate(data):
      if not isinstance(v, torch.Tensor):
        data[i] = to_torch_tensors(v)
    
  return data


def send_to_device(data):
  if isinstance(data, dict):
    for k, v in data.items():
      if isinstance(v, torch.Tensor):
        data[k] = v.to(device)
  elif isinstance(data, list):
    for i, v in enumerate(data):
      if isinstance(v, torch.Tensor):
        data[i] = v.to(device)
    
  return data


def getsize(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object reffered to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz