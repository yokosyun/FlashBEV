import os
import platform
import sys
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import torch


def make_cuda_ext(name, module, sources, extra_args=[]):
    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CUDAExtension
        raise EnvironmentError('CUDA is required to compile FlashBEVPool!')

    # Build full paths to source files
    source_paths = []
    for source in sources:
        if os.path.isabs(source):
            source_paths.append(source)
        else:
            source_paths.append(os.path.join(*module.split("."), source))
    
    return extension(
        name="{}.{}".format(module, name),
        sources=source_paths,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="flashbevpool",
        version="0.1.0",
        description="FlashBEVPool: Fast BEV Pooling with Fused CUDA Kernels",
        long_description="A high-performance BEV pooling implementation with fused CUDA kernels for 3D object detection.",
        author="FlashBEV Contributors",
        keywords="computer vision, 3D object detection, BEV pooling, CUDA",
        url="https://github.com/yourusername/flashbevpool",
        packages=find_packages(),
        include_package_data=True,
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        license="Apache License 2.0",
        install_requires=["torch>=1.8.0"],
        ext_modules=[
            make_cuda_ext(
                name="flashbevpool_ext",
                module="flashbevpool",
                sources=[
                    "src/flash_bevpool.cpp",
                    "src/flash_bevpool_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="sampling_vt_ext",
                module="flashbevpool",
                sources=[
                    "src/sampling_vt.cpp",
                    "src/sampling_vt_cuda.cu",
                ],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
