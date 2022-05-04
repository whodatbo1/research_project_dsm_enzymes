import importlib.util


def get_instance_info(i):
    fileName = 'FJSP_' + str(i)
    spec = importlib.util.spec_from_file_location('instance', "../instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def calculate_problem_size(instance):
    init = 1
    for j in instance.jobs:
        for op in instance.operations[j]:
            init *= len(instance.machineAlternatives[j, op])
    return init

print(calculate_problem_size(get_instance_info(0)))