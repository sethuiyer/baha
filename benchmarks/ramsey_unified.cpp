#include "baha.hpp"
#include "ramsey_backend.h"
#include "ramsey_common.h"

#include <chrono>
#include <climits>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string backend = "cpu";
    int n = 52;
    int k = 5;
    int colors = 3;
};

bool starts_with(const char* arg, const char* prefix) {
    return std::strncmp(arg, prefix, std::strlen(prefix)) == 0;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (starts_with(argv[i], "--backend=")) {
            args.backend = std::string(argv[i] + std::strlen("--backend="));
        } else if (starts_with(argv[i], "--n=")) {
            args.n = std::stoi(argv[i] + std::strlen("--n="));
        } else if (starts_with(argv[i], "--k=")) {
            args.k = std::stoi(argv[i] + std::strlen("--k="));
        } else if (starts_with(argv[i], "--colors=")) {
            args.colors = std::stoi(argv[i] + std::strlen("--colors="));
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ramsey_unified [--backend=cpu|cuda|mps] [--n=52] [--k=5] [--colors=3]\n";
            std::exit(0);
        }
    }
    return args;
}

std::vector<int> random_state(int num_edges, int colors) {
    std::vector<int> s(num_edges);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, colors - 1);
    for (int i = 0; i < num_edges; ++i) s[i] = dist(rng);
    return s;
}

std::vector<std::vector<int>> neighbors(const std::vector<int>& s, int num_edges, int colors) {
    std::vector<std::vector<int>> nbrs;
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> edge_dist(0, num_edges - 1);

    for (int i = 0; i < 100; ++i) {
        std::vector<int> n = s;
        int idx = edge_dist(rng);
        n[idx] = (n[idx] + 1) % colors;
        nbrs.push_back(std::move(n));
    }

    return nbrs;
}

std::unique_ptr<RamseyBackend> make_backend(
    const std::string& backend,
    const RamseyCliqueData& cliques) {
    if (backend == "cpu") {
        return create_cpu_backend(cliques.clique_edges, cliques.num_cliques, cliques.k_edges, cliques.num_edges);
    }

#ifdef BAHA_HAVE_CUDA_BACKEND
    if (backend == "cuda") {
        return create_cuda_backend(cliques.clique_edges, cliques.num_cliques, cliques.k_edges, cliques.num_edges);
    }
#endif

#ifdef BAHA_HAVE_MPS_BACKEND
    if (backend == "mps") {
        return create_mps_backend(cliques.clique_edges, cliques.num_cliques, cliques.k_edges, cliques.num_edges);
    }
#endif

    return nullptr;
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    std::cout << "============================================================\n";
    std::cout << "UNIFIED BAHA RAMSEY SOLVER\n";
    std::cout << "Backend: " << args.backend << " | N=" << args.n << " | K=" << args.k
              << " | Colors=" << args.colors << "\n";
    std::cout << "============================================================\n";

    RamseyCliqueData cliques = ramsey_build_cliques(args.n, args.k);
    std::cout << "Cliques: " << cliques.num_cliques << " | Edges: " << cliques.num_edges << "\n";

    auto backend = make_backend(args.backend, cliques);
    if (!backend) {
        // Fallback to CPU when the requested backend is not available.
        std::cerr << "Backend '" << args.backend << "' unavailable; falling back to CPU.\n";
        backend = make_backend("cpu", cliques);
    }

    auto energy = [&backend](const std::vector<int>& s) { return backend->energy(s); };
    auto sampler = [&]() { return random_state(cliques.num_edges, args.colors); };
    auto neighbor_fn = [&](const std::vector<int>& s) { return neighbors(s, cliques.num_edges, args.colors); };

    navokoj::BranchAwareOptimizer<std::vector<int>> baha(energy, sampler, neighbor_fn);
    navokoj::BranchAwareOptimizer<std::vector<int>>::Config config;

    config.beta_steps = 500;
    config.beta_end = 15.0;
    config.samples_per_beta = 40;
    config.fracture_threshold = 1.8;
    config.max_branches = 8;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<std::vector<int>>::ScheduleType::GEOMETRIC;

    struct ProgressState {
        int last_energy = INT_MAX;
        double best_seen = 1e9;
        std::chrono::high_resolution_clock::time_point start;
    };
    auto progress = std::make_shared<ProgressState>();
    progress->start = std::chrono::high_resolution_clock::now();

    config.logger = [progress, &config](int step, double beta, double current_energy, double rho, const char* event) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - progress->start).count();

        if (current_energy < progress->best_seen) {
            progress->best_seen = current_energy;
        }

        int e = static_cast<int>(progress->best_seen);
        bool improved = (e < progress->last_energy);

        if (improved || step % 50 == 0) {
            std::cout << "\r[" << std::fixed << std::setprecision(1) << elapsed << "s] "
                      << "Step " << step << "/" << config.beta_steps
                      << " | b=" << std::setprecision(2) << beta
                      << " | E=" << e;

            if (std::strcmp(event, "branch_jump") == 0) {
                std::cout << " [JUMP]";
            }
            if (rho > 10.0) {
                std::cout << " | rho=" << std::setprecision(0) << rho;
            }

            if (improved) {
                std::cout << " <-- NEW BEST\n";
                progress->last_energy = e;
            } else {
                std::cout << std::flush;
            }
        }
    };

    std::cout << "\nStarting BAHA with backend: " << backend->name() << "...\n";
    auto start = std::chrono::high_resolution_clock::now();
    auto result = baha.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "\n============================================================\n";
    std::cout << "RESULT\n";
    std::cout << "============================================================\n";
    std::cout << "Final Energy: " << result.best_energy << "\n";
    std::cout << "Time: " << elapsed << " seconds\n";
    std::cout << "Fractures: " << result.fractures_detected << "\n";
    std::cout << "Branch Jumps: " << result.branch_jumps << "\n";

    return 0;
}
