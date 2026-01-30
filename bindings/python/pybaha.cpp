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
    
    // =========================================================================
    // ZETA BREATHER OPTIMIZER BINDINGS (High-Performance Hybrid Engine)
    // =========================================================================
    // Uses std::vector<double> for continuous states
    using ContinuousState = std::vector<double>;
    
    py::class_<ZetaBreatherOptimizer<PyState>::Config>(m, "ZetaConfig")
        .def(py::init<>())
        .def_readwrite("beta_min", &ZetaBreatherOptimizer<PyState>::Config::beta_min)
        .def_readwrite("beta_max", &ZetaBreatherOptimizer<PyState>::Config::beta_max)
        .def_readwrite("period", &ZetaBreatherOptimizer<PyState>::Config::period)
        .def_readwrite("total_steps", &ZetaBreatherOptimizer<PyState>::Config::total_steps)
        .def_readwrite("chunk_size", &ZetaBreatherOptimizer<PyState>::Config::chunk_size)
        .def_readwrite("polish_steps", &ZetaBreatherOptimizer<PyState>::Config::polish_steps)
        .def_readwrite("polish_samples", &ZetaBreatherOptimizer<PyState>::Config::polish_samples)
        .def_readwrite("learning_rate", &ZetaBreatherOptimizer<PyState>::Config::learning_rate)
        .def_readwrite("timeout_ms", &ZetaBreatherOptimizer<PyState>::Config::timeout_ms)
        .def_readwrite("verbose", &ZetaBreatherOptimizer<PyState>::Config::verbose);
    
    py::class_<ZetaBreatherOptimizer<PyState>::Result>(m, "ZetaResult")
        .def_readonly("best_state", &ZetaBreatherOptimizer<PyState>::Result::best_state)
        .def_readonly("best_energy", &ZetaBreatherOptimizer<PyState>::Result::best_energy)
        .def_readonly("time_ms", &ZetaBreatherOptimizer<PyState>::Result::time_ms)
        .def_readonly("steps_taken", &ZetaBreatherOptimizer<PyState>::Result::steps_taken)
        .def_readonly("peaks_harvested", &ZetaBreatherOptimizer<PyState>::Result::peaks_harvested)
        .def_readonly("timeout_reached", &ZetaBreatherOptimizer<PyState>::Result::timeout_reached);
    
    py::class_<ZetaBreatherOptimizer<PyState>>(m, "ZetaOptimizer")
        .def(py::init<
            std::function<double(const PyState&)>,
            std::function<PyState()>,
            std::function<std::vector<PyState>(const PyState&)>,
            std::function<ContinuousState(const PyState&)>,
            std::function<PyState(const ContinuousState&)>,
            std::function<double(const ContinuousState&, double)>,
            std::function<ContinuousState(const ContinuousState&, double)>
        >(), py::arg("discrete_energy"), py::arg("sampler"), py::arg("neighbors"),
            py::arg("encode"), py::arg("decode"), 
            py::arg("continuous_energy"), py::arg("continuous_gradient"))
        .def("optimize", &ZetaBreatherOptimizer<PyState>::optimize,
             py::arg("config") = ZetaBreatherOptimizer<PyState>::Config());

    // =========================================================================
    // ADAPTIVE OPTIMIZER BINDINGS (Auto-Switching Engine)
    // =========================================================================
    // Probes fracture density, then switches: density > 0.3 → BranchAware, else Zeta
    
    py::class_<AdaptiveOptimizer<PyState>::Config>(m, "AdaptiveConfig")
        .def(py::init<>())
        .def_readwrite("fracture_threshold", &AdaptiveOptimizer<PyState>::Config::fracture_threshold)
        .def_readwrite("probe_steps", &AdaptiveOptimizer<PyState>::Config::probe_steps)
        .def_readwrite("probe_samples", &AdaptiveOptimizer<PyState>::Config::probe_samples)
        .def_readwrite("ba_beta_start", &AdaptiveOptimizer<PyState>::Config::ba_beta_start)
        .def_readwrite("ba_beta_end", &AdaptiveOptimizer<PyState>::Config::ba_beta_end)
        .def_readwrite("ba_beta_steps", &AdaptiveOptimizer<PyState>::Config::ba_beta_steps)
        .def_readwrite("ba_samples_per_beta", &AdaptiveOptimizer<PyState>::Config::ba_samples_per_beta)
        .def_readwrite("ba_max_branches", &AdaptiveOptimizer<PyState>::Config::ba_max_branches)
        .def_readwrite("zeta_beta_min", &AdaptiveOptimizer<PyState>::Config::zeta_beta_min)
        .def_readwrite("zeta_beta_max", &AdaptiveOptimizer<PyState>::Config::zeta_beta_max)
        .def_readwrite("zeta_period", &AdaptiveOptimizer<PyState>::Config::zeta_period)
        .def_readwrite("zeta_total_steps", &AdaptiveOptimizer<PyState>::Config::zeta_total_steps)
        .def_readwrite("zeta_polish_steps", &AdaptiveOptimizer<PyState>::Config::zeta_polish_steps)
        .def_readwrite("zeta_polish_samples", &AdaptiveOptimizer<PyState>::Config::zeta_polish_samples)
        .def_readwrite("zeta_learning_rate", &AdaptiveOptimizer<PyState>::Config::zeta_learning_rate)
        .def_readwrite("timeout_ms", &AdaptiveOptimizer<PyState>::Config::timeout_ms)
        .def_readwrite("verbose", &AdaptiveOptimizer<PyState>::Config::verbose);
    
    py::class_<AdaptiveOptimizer<PyState>::Result>(m, "AdaptiveResult")
        .def_readonly("best_state", &AdaptiveOptimizer<PyState>::Result::best_state)
        .def_readonly("best_energy", &AdaptiveOptimizer<PyState>::Result::best_energy)
        .def_readonly("time_ms", &AdaptiveOptimizer<PyState>::Result::time_ms)
        .def_readonly("steps_taken", &AdaptiveOptimizer<PyState>::Result::steps_taken)
        .def_readonly("fractures_detected", &AdaptiveOptimizer<PyState>::Result::fractures_detected)
        .def_readonly("fracture_density", &AdaptiveOptimizer<PyState>::Result::fracture_density)
        .def_readonly("used_branch_aware", &AdaptiveOptimizer<PyState>::Result::used_branch_aware)
        .def_readonly("timeout_reached", &AdaptiveOptimizer<PyState>::Result::timeout_reached);
    
    py::class_<AdaptiveOptimizer<PyState>>(m, "AdaptiveOptimizer")
        .def(py::init<
            std::function<double(const PyState&)>,
            std::function<PyState()>,
            std::function<std::vector<PyState>(const PyState&)>,
            std::function<ContinuousState(const PyState&)>,
            std::function<PyState(const ContinuousState&)>,
            std::function<double(const ContinuousState&, double)>,
            std::function<ContinuousState(const ContinuousState&, double)>
        >(), py::arg("discrete_energy"), py::arg("sampler"), py::arg("neighbors"),
            py::arg("encode") = nullptr, py::arg("decode") = nullptr, 
            py::arg("continuous_energy") = nullptr, py::arg("continuous_gradient") = nullptr)
        .def("optimize", &AdaptiveOptimizer<PyState>::optimize,
             py::arg("config") = AdaptiveOptimizer<PyState>::Config());
    
    // =========================================================================
    // AUTO ZETA OPTIMIZER BINDINGS (Physics-Based Auto-Relaxation)
    // =========================================================================
    // Uses std::vector<int> as discrete state type (works with Python lists)
    using IntState = std::vector<int>;
    
    py::class_<AutoZetaOptimizer<IntState>::Config>(m, "AutoZetaConfig")
        .def(py::init<>())
        .def_readwrite("beta_min", &AutoZetaOptimizer<IntState>::Config::beta_min)
        .def_readwrite("beta_max", &AutoZetaOptimizer<IntState>::Config::beta_max)
        .def_readwrite("period", &AutoZetaOptimizer<IntState>::Config::period)
        .def_readwrite("total_steps", &AutoZetaOptimizer<IntState>::Config::total_steps)
        .def_readwrite("polish_steps", &AutoZetaOptimizer<IntState>::Config::polish_steps)
        .def_readwrite("polish_samples", &AutoZetaOptimizer<IntState>::Config::polish_samples)
        .def_readwrite("learning_rate", &AutoZetaOptimizer<IntState>::Config::learning_rate)
        .def_readwrite("grad_eps", &AutoZetaOptimizer<IntState>::Config::grad_eps)
        .def_readwrite("timeout_ms", &AutoZetaOptimizer<IntState>::Config::timeout_ms)
        .def_readwrite("verbose", &AutoZetaOptimizer<IntState>::Config::verbose);
    
    py::class_<AutoZetaOptimizer<IntState>::Result>(m, "AutoZetaResult")
        .def_readonly("best_state", &AutoZetaOptimizer<IntState>::Result::best_state)
        .def_readonly("best_energy", &AutoZetaOptimizer<IntState>::Result::best_energy)
        .def_readonly("time_ms", &AutoZetaOptimizer<IntState>::Result::time_ms)
        .def_readonly("steps_taken", &AutoZetaOptimizer<IntState>::Result::steps_taken)
        .def_readonly("peaks_harvested", &AutoZetaOptimizer<IntState>::Result::peaks_harvested)
        .def_readonly("timeout_reached", &AutoZetaOptimizer<IntState>::Result::timeout_reached);
    
    py::class_<AutoZetaOptimizer<IntState>>(m, "AutoZetaOptimizer")
        .def(py::init<
            std::function<double(const IntState&)>,
            std::function<IntState()>,
            std::function<std::vector<IntState>(const IntState&)>,
            int
        >(), py::arg("discrete_energy"), py::arg("sampler"), py::arg("neighbors"),
            py::arg("domain_size"))
        .def("optimize", &AutoZetaOptimizer<IntState>::optimize,
             py::arg("config") = AutoZetaOptimizer<IntState>::Config());
    
    // =========================================================================
    // DEFAULT INTERFACE: optimize() uses AdaptiveOptimizer
    // =========================================================================
    // This is the recommended entry point for most users.
    // Simply call: pybaha.optimize(energy, sampler, neighbors)
    
    m.def("optimize", [](
        std::function<double(const PyState&)> energy,
        std::function<PyState()> sampler,
        std::function<std::vector<PyState>(const PyState&)> neighbors,
        double timeout_ms
    ) {
        AdaptiveOptimizer<PyState> opt(energy, sampler, neighbors);
        typename AdaptiveOptimizer<PyState>::Config config;
        config.timeout_ms = timeout_ms;
        return opt.optimize(config);
    },
    py::arg("energy"),
    py::arg("sampler"), 
    py::arg("neighbors"),
    py::arg("timeout_ms") = 5000.0,
    R"doc(
        Optimize using AdaptiveOptimizer (default, recommended).
        
        This is the simplest interface to BAHA. Automatically selects
        between BranchAware and ZetaBreather engines based on fracture density.
        
        Args:
            energy: Function(state) -> float, energy to minimize
            sampler: Function() -> state, generates random initial states  
            neighbors: Function(state) -> list[state], generates neighbor states
            timeout_ms: Maximum time in milliseconds (default: 5000)
        
        Returns:
            AdaptiveResult with best_state, best_energy, fracture_density, etc.
        
        Example:
            result = pybaha.optimize(energy, sampler, neighbors)
            print(f"Best energy: {result.best_energy}")
    )doc");
}
