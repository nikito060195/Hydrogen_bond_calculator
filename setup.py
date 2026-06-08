# setup.py
import pybind11
from setuptools import setup, Extension

ext_modules = [
    Extension(
        "hbond_core",                      # O nome do módulo que será importado no Python
        ["hbond_core.cpp"],                # O nosso arquivo fonte em C++
        include_dirs=[pybind11.get_include()],
        language='c++',
        # Flags essenciais para alta performance:
        # -O3: Otimização agressiva do compilador
        # -march=native: Otimiza as instruções para a arquitetura exata da sua CPU
        # -fopenmp: Habilita o multiprocessamento nativo do C++
        extra_compile_args=['-O3', '-march=native', '-fopenmp'], 
        extra_link_args=['-fopenmp']
    ),
]

setup(
    name="hbond_core",
    ext_modules=ext_modules,
)
