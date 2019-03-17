from collections import namedtuple

Device = namedtuple("Device",
        ("device_name", "ssh_address", 'backends', "num_threads"))

devices = [
    Device('rpi3b',    'pi',     ['arm_cpu'], 4),
    Device('rk3399',   'fire',   ['arm_cpu', 'mali'], 2),

    Device('1080ti',   'aquarium',  ['cuda', 'opencl'], -1),
    Device('titanx',   'sharetea',  ['cuda', 'opencl'], -1),
]

