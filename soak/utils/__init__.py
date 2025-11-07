import json

from pydantic import BaseModel


def debug_serialization(obj, path="root", seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        print(f"↩️  {path}: already seen {type(obj)} – stopping recursion")
        return
    seen.add(obj_id)

    if isinstance(obj, BaseModel):
        for name, value in obj.__dict__.items():
            try:
                debug_serialization(value, f"{path}.{name}", seen)
            except Exception as e:
                print(f"❌ {path}.{name}: {type(value)} -> {e}")
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                debug_serialization(v, f"{path}[{k!r}]", seen)
            except Exception as e:
                print(f"❌ {path}[{k!r}]: {type(v)} -> {e}")
        return

    if isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            try:
                debug_serialization(v, f"{path}[{i}]", seen)
            except Exception as e:
                print(f"❌ {path}[{i}]: {type(v)} -> {e}")
        return

    try:
        json.dumps(obj)
    except TypeError as e:
        print(f"❌ {path}: not serialisable ({type(obj)}): {e}")
