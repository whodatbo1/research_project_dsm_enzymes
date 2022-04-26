import importlib.util


def get_instance_info(i):
    fileName = 'FJSP_' + str(i)
    spec = importlib.util.spec_from_file_location('instance', "../instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod