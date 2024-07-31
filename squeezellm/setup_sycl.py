from setuptools import setup
#import torch
import intel_extension_for_pytorch
#from torch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension
from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension
#from intel_extension_for_pytorch.utils.cpp_extension import DPCPPExtension, DpcppBuildExtension
setup(
    name='quant_sycl',
    ext_modules=[
        DPCPPExtension(
           'quant_sycl', ['quant_cuda.cpp', 'quant_cuda_kernel.dp.cpp']
        )
    ],
    cmdclass={'build_ext': DpcppBuildExtension}
)
