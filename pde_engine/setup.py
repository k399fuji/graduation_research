from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    # 熱方程式モジュール
    Pybind11Extension(
        "heat1d_cpp",
        [
            "cpp/heat1d_module.cpp",
            "cpp/heat1d_solver.cpp",
        ],
        include_dirs=["cpp"],
        cxx_std=17,
    ),
    # 波動方程式モジュール
    Pybind11Extension(
        "wave1d_cpp",
        [
            "cpp/wave1d_module.cpp",
            "cpp/wave1d_solver.cpp",
        ],
        include_dirs=["cpp"],
        cxx_std=17,
    ),

    # 2D 熱方程式モジュール
    Pybind11Extension(
        "heat2d_cpp",
        [
            "cpp/heat2d_module.cpp",
            "cpp/heat2d_solver.cpp",
        ],
        include_dirs=["cpp"],
        cxx_std=17,
    )
]

setup(
    name="pde_solvers",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
