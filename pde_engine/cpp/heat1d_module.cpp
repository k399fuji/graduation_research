#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "heat1d_solver.hpp"

namespace py = pybind11;

std::pair<std::vector<double>, std::vector<double>>
solve_heat_1d(int Nx, double L, double alpha, double dt,
              int steps, ICType ic_type, double gaussian_k)
{
    Heat1DSolver solver(Nx, L, alpha, dt);
    solver.set_initial_condition(ic_type, gaussian_k);
    solver.run(steps);
    return {solver.get_x(), solver.get_u()};
}

PYBIND11_MODULE(heat1d_cpp, m) {
    m.doc() = "1D heat equation solver (C++ + pybind11)";

    // enum
    py::enum_<ICType>(m, "ICType")
        .value("Gaussian", ICType::Gaussian)
        .value("Sine", ICType::Sine)
        .value("TwoPeaks", ICType::TwoPeaks)
        .export_values();

    // class binding
    py::class_<Heat1DSolver>(m, "Heat1DSolver")
        .def(py::init<int,double,double,double>(),
             py::arg("Nx")=101, py::arg("L")=1.0,
             py::arg("alpha")=0.01, py::arg("dt")=0.0005)
        .def("set_initial_condition", &Heat1DSolver::set_initial_condition,
             py::arg("ic_type"), py::arg("gaussian_k")=100.0)
        .def("reset_initial", &Heat1DSolver::reset_initial)
        .def("step", &Heat1DSolver::step)
        .def("run", &Heat1DSolver::run)
        .def("get_x", &Heat1DSolver::get_x,
             py::return_value_policy::reference_internal)
        .def("get_u", &Heat1DSolver::get_u,
             py::return_value_policy::reference_internal);

    // function binding
    m.def("solve_heat_1d", &solve_heat_1d,
          py::arg("Nx")=101, py::arg("L")=1.0,
          py::arg("alpha")=0.01, py::arg("dt")=0.0005,
          py::arg("steps")=800, py::arg("ic_type")=ICType::Gaussian,
          py::arg("gaussian_k")=100.0);
}
