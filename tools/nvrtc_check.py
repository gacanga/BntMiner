# nvrtc_check.py - compiles the device part of gpu/bnt_gpu.cu (up to "host side") with NVRTC
# for sm_61 and runs ptxas -v (registers, spills), on Linux without a CUDA toolkit.
#   pip download nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 --no-deps, unzip both wheels into one
#   folder, then:  CUDA_PIP_ROOT=<folder>/nvidia python3 tools/nvrtc_check.py gpu/bnt_gpu.cu
import ctypes, sys, subprocess, os, re, tempfile
C = os.environ.get('CUDA_PIP_ROOT', 'cuda/nvidia')
lib = ctypes.CDLL(C + '/cuda_nvrtc/lib/libnvrtc.so.12')
src = open(sys.argv[1]).read()
src = src.split('// host side')[0]
src = re.sub(r'#include [<"][^>"]*[>"]', '', src)
src = 'typedef unsigned long long uint64_t; typedef unsigned int uint32_t; typedef unsigned char uint8_t;\n' + src
prog = ctypes.c_void_p()
lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"k.cu", 0, None, None)
opts = [b"--gpu-architecture=compute_61", b"-std=c++17"] + [o.encode() for o in sys.argv[2:]]
arr = (ctypes.c_char_p * len(opts))(*opts)
r = lib.nvrtcCompileProgram(prog, len(opts), arr)
n = ctypes.c_size_t(); lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
log = ctypes.create_string_buffer(n.value); lib.nvrtcGetProgramLog(prog, log)
print(log.value.decode())
if r != 0: print('NVRTC FAILED', r); sys.exit(1)
lib.nvrtcGetPTXSize(prog, ctypes.byref(n)); ptx = ctypes.create_string_buffer(n.value); lib.nvrtcGetPTX(prog, ptx)
open(os.path.join(tempfile.gettempdir(), 'k.ptx'), 'wb').write(ptx.value)
out = subprocess.run([C + '/cuda_nvcc/bin/ptxas', '-arch=sm_61', '-v', '-O3', os.path.join(tempfile.gettempdir(), 'k.ptx'), '-o', os.path.join(tempfile.gettempdir(), 'k.cubin')], capture_output=True, text=True)
print(out.stdout, out.stderr)
sys.exit(out.returncode)
