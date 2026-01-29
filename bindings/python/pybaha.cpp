#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "../../include/baha/baha.hpp"

namespace py = pybind11;
using namespace navokoj;

// Use py::object as the state type to allow Python-defined states
using PyState = py::object;

PYBIND11_MODULE(pybaha, m) {
    m.doc() = "Python bindings for BAHA (Branch-Aware Holonomy Annealing)";

    py::enum_<BranchAwareOptimizer<PyState>::ScheduleType>(m, "ScheduleType")
        .value("LINEAR", BranchAwareOptimizer<PyState>::ScheduleType::LINEAR)
        .value("GEOMETRIC", BranchAwareOptimizer<PyState>::ScheduleType::GEOMETRIC)
        .export_values();

    py::class_<BranchAwareOptimizer<PyState>::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("beta_start", &BranchAwareOptimizer<PyState>::Config::beta_start)
        .def_readwrite("beta_end", &BranchAwareOptimizer<PyState>::Config::beta_end)
        .def_readwrite("beta_steps", &BranchAwareOptimizer<PyState>::Config::beta_steps)
        .def_readwrite("fracture_threshold", &BranchAwareOptimizer<PyState>::Config::fracture_threshold)
        .def_readwrite("beta_critical", &BranchAwareOptimizer<PyState>::Config::beta_critical)
        .def_readwrite("max_branches", &BranchAwareOptimizer<PyState>::Config::max_branches)
        .def_readwrite("samples_per_beta", &BranchAwareOptimizer<PyState>::Config::samples_per_beta)
        .def_readwrite("verbose", &BranchAwareOptimizer<PyState>::Config::verbose)
        .def_readwrite("schedule_type", &BranchAwareOptimizer<PyState>::Config::schedule_type)
        .def_readwrite("timeout_ms", &BranchAwareOptimizer<PyState>::Config::timeout_ms)
        .def_readwrite("validator", &BranchAwareOptimizer<PyState>::Config::validator);

    py::class_<BranchAwareOptimizer<PyState>::Result>(m, "Result")
        .def_readonly("best_state", &BranchAwareOptimizer<PyState>::Result::best_state)
        .def_readonly("best_energy", &BranchAwareOptimizer<PyState>::Result::best_energy)
        .def_readonly("fractures_detected", &BranchAwareOptimizer<PyState>::Result::fractures_detected)
        .def_readonly("branch_jumps", &BranchAwareOptimizer<PyState>::Result::branch_jumps)
        .def_readonly("beta_at_solution", &BranchAwareOptimizer<PyState>::Result::beta_at_solution)
        .def_readonly("steps_taken", &BranchAwareOptimizer<PyState>::Result::steps_taken)
        .def_readonly("time_ms", &BranchAwareOptimizer<PyState>::Result::time_ms)
        .def_readonly("timeout_reached", &BranchAwareOptimizer<PyState>::Result::timeout_reached)
        .def_readonly("energy_history", &BranchAwareOptimizer<PyState>::Result::energy_history)
        .def_readonly("validation_metric", &BranchAwareOptimizer<PyState>::Result::validation_metric);

    py::class_<BranchAwareOptimizer<PyState>>(m, "Optimizer")
        .def(py::init<
            BranchAwareOptimizer<PyState>::EnergyFn,
            BranchAwareOptimizer<PyState>::SamplerFn,
            BranchAwareOptimizer<PyState>::NeighborFn
        >(), py::arg("energy"), py::arg("sampler"), py::arg("neighbors") = nullptr)
        .def("optimize", &BranchAwareOptimizer<PyState>::optimize, 
             py::arg("config") = BranchAwareOptimizer<PyState>::Config());
}
