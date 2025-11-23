// cpp/wave1d_module.cpp
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "wave1d_solver.hpp"

namespace py = pybind11;

// ラッパ関数: Wave1DSolver を内部で生成して steps 回まわし、(x, u) を返す
std::pair<std::vector<double>, std::vector<double>>
solve_wave_1d(int    Nx    = 201,
              double L     = 1.0,
              double c     = 1.0,
              double dt    = 0.001,
              int    steps = 1000)
{
    Wave1DSolver solver(Nx, L, c, dt);
    solver.reset_initial();
    solver.run(steps);
    return {solver.get_x(), solver.get_u()};
}

PYBIND11_MODULE(wave1d_cpp, m) {
    m.doc() = "1D wave equation solver in C++ (Wave1DSolver, pybind11 binding)";

    py::class_<Wave1DSolver>(m, "Wave1DSolver")
        .def(py::init<int, double, double, double>(),
             py::arg("Nx") = 201,
             py::arg("L")  = 1.0,
             py::arg("c")  = 1.0,
             py::arg("dt") = 0.001)
        .def("reset_initial", &Wave1DSolver::reset_initial)
        .def("run",           &Wave1DSolver::run,  py::arg("steps"))
        .def("step",          &Wave1DSolver::step)
        .def("get_x",         &Wave1DSolver::get_x,
             py::return_value_policy::reference_internal)
        .def("get_u",         &Wave1DSolver::get_u,
             py::return_value_policy::reference_internal);

    // 関数版（既存 Python コードのために残す）
    m.def("solve_wave_1d",
          &solve_wave_1d,
          py::arg("Nx")    = 201,
          py::arg("L")     = 1.0,
          py::arg("c")     = 1.0,
          py::arg("dt")    = 0.001,
          py::arg("steps") = 1000,
          "Solve 1D wave equation and return (x, u_final).");
}
