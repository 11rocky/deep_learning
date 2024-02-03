import importlib
import inspect
import os


def get_cls_in_module(module, name, parent_cls=object):
    if isinstance(module, str):
        module = importlib.import_module(module)
    for _, member in inspect.getmembers(module):
        if inspect.isclass(member) and issubclass(member, parent_cls) and member.__name__ == name:
            return member
    return None


def get_cls_in_package(package: str, name, parent_cls=object):
    pkg = importlib.import_module(package)
    items = os.listdir(os.path.dirname(pkg.__file__))
    for item in items:
        if item.startswith("_") or not item.endswith(".py"):
            continue
        mod = os.path.splitext(os.path.basename(item))[0]
        module = importlib.import_module("{}.{}".format(package, mod))
        cls = get_cls_in_module(module, name, parent_cls)
        if cls is not None:
            return cls
    return None

