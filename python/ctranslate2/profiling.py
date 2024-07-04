from ctranslate2 import _ext, Device
import sys

def init_profiler(device = Device.cpu, num_threads = 1):
    _ext.init_profiling(device, num_threads)


def dump_profiler():
    profiling_data = _ext.dump_profiling()
    sys.stdout.write(profiling_data)
