// heat2d_module.cpp
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "heat2d_solver.hpp"

namespace py = pybind11;

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
solve_heat_2d(int    Nx         = 101,
              int    Ny         = 101,
              double Lx         = 1.0,
              double Ly         = 1.0,
              double alpha      = 0.01,
              double dt         = 1.0e-4,
              int    steps      = 1000,
              ICType2D ic_type  = ICType2D::Gaussian,
              double gaussian_kx = 100.0,
              double gaussian_ky = 100.0)
{
    Heat2DSolver solver(Nx, Ny, Lx, Ly, alpha, dt);
    solver.set_initial_condition(ic_type, gaussian_kx, gaussian_ky);
    solver.run(steps);

    return std::make_tuple(solver.get_x(), solver.get_y(), solver.get_u());
}

PYBIND11_MODULE(heat2d_cpp, m) {
    m.doc() = "2D heat equation solver in C++ (pybind11 binding)";

    py::enum_<ICType2D>(m, "ICType2D")
        .value("Gaussian", ICType2D::Gaussian)
        .value("SineXY",   ICType2D::SineXY)
        .export_values();

    py::class_<Heat2DSolver>(m, "Heat2DSolver")
        .def(py::init<int,int,double,double,double,double>(),
             py::arg("Nx")    = 101,
             py::arg("Ny")    = 101,
             py::arg("Lx")    = 1.0,
             py::arg("Ly")    = 1.0,
             py::arg("alpha") = 0.01,
             py::arg("dt")    = 1.0e-4)
        .def("set_initial_condition", &Heat2DSolver::set_initial_condition,
             py::arg("ic_type"),
             py::arg("gaussian_kx") = 100.0,
             py::arg("gaussian_ky") = 100.0)
        .def("reset_initial", &Heat2DSolver::reset_initial)
        .def("step", &Heat2DSolver::step)
        .def("run", &Heat2DSolver::run)
        .def("get_x", &Heat2DSolver::get_x,
             py::return_value_policy::reference_internal)
        .def("get_y", &Heat2DSolver::get_y,
             py::return_value_policy::reference_internal)
        .def("get_u", &Heat2DSolver::get_u,
             py::return_value_policy::reference_internal);

    m.def("solve_heat_2d",
          &solve_heat_2d,
          py::arg("Nx")          = 101,
          py::arg("Ny")          = 101,
          py::arg("Lx")          = 1.0,
          py::arg("Ly")          = 1.0,
          py::arg("alpha")       = 0.01,
          py::arg("dt")          = 1.0e-4,
          py::arg("steps")       = 1000,
          py::arg("ic_type")     = ICType2D::Gaussian,
          py::arg("gaussian_kx") = 100.0,
          py::arg("gaussian_ky") = 100.0,
          "Solve 2D heat equation and return (x, y, u_flat).");
}
