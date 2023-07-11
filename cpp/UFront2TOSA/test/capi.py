import argparse
import ctypes

capi = ctypes.CDLL("../build/lib/libUfrontCAPI.so")
capi.ufront_to_tosa.restype = ctypes.c_char_p
capi.ufront_to_tosa.argtypes = [ctypes.c_char_p]

parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="?", default=None, help="Ufront file to parse")

args = parser.parse_args()
with open(args.file) as f:
    ufront = f.read()

print(capi.ufront_to_tosa(ufront.encode("utf-8")))
