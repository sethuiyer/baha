#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "baha/baha.hpp"

using json = nlohmann::json;
using namespace navokoj;

// Problem: Number Partitioning
struct PartitionState {
    std::vector<int> s;
};

class PartitionProblem {
    std::vector<long long> numbers;
public:
    PartitionProblem(const std::vector<long long>& data) : numbers(data) {}
    
    double energy(const PartitionState& state) {
        long long sum = 0;
        for (size_t i = 0; i < numbers.size(); ++i) {
            sum += state.s[i] * numbers[i];
        }
        return static_cast<double>(std::abs(sum));
    }
    
    PartitionState random_state() {
        PartitionState state;
        state.s.resize(numbers.size());
        auto& rng = get_rng();
        std::uniform_int_distribution<> dist(0, 1);
        for (size_t i = 0; i < numbers.size(); ++i) state.s[i] = dist(rng) ? 1 : -1;
        return state;
    }
    
    std::vector<PartitionState> neighbors(const PartitionState& state) {
        std::vector<PartitionState> nbrs;
        for (size_t i = 0; i < numbers.size(); ++i) {
            PartitionState nbr = state;
            nbr.s[i] *= -1;
            nbrs.push_back(nbr);
        }
        return nbrs;
    }

private:
    std::mt19937& get_rng() {
        static std::mt19937 rng(std::random_device{}());
        return rng;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: baha-cli <problem.json>" << std::endl;
        return 1;
    }

    std::ifstream f(argv[1]);
    if (!f.is_open()) {
        std::cerr << "Could not open file: " << argv[1] << std::endl;
        return 1;
    }

    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        std::cerr << "JSON Parse Error: " << e.what() << std::endl;
        return 1;
    }

    std::string problem_type = j.value("problem", "number_partitioning");
    
    if (problem_type == "number_partitioning") {
        std::vector<long long> data = j["data"].get<std::vector<long long>>();
        PartitionProblem prob(data);
        
        auto energy = [&](const PartitionState& s) { return prob.energy(s); };
        auto sampler = [&]() { return prob.random_state(); };
        auto neighbors = [&](const PartitionState& s) { return prob.neighbors(s); };
        
        BranchAwareOptimizer<PartitionState> opt(energy, sampler, neighbors);
        BranchAwareOptimizer<PartitionState>::Config config;
        
        if (j.contains("config")) {
            auto& c = j["config"];
            config.beta_steps = c.value("beta_steps", 1000);
            config.beta_end = c.value("beta_end", 20.0);
            config.verbose = c.value("verbose", true);
            config.timeout_ms = c.value("timeout_ms", -1.0);
        }
        
        auto result = opt.optimize(config);
        
        json res_json;
        res_json["status"] = "success";
        res_json["best_energy"] = result.best_energy;
        res_json["best_state"] = result.best_state.s;
        res_json["time_ms"] = result.time_ms;
        res_json["fractures"] = result.fractures_detected;
        res_json["jumps"] = result.branch_jumps;
        res_json["timeout_reached"] = result.timeout_reached;
        res_json["energy_history"] = result.energy_history;
        res_json["validation_metric"] = result.validation_metric;
        
        std::cout << res_json.dump(2) << std::endl;
    } else {
        std::cerr << "Unknown problem type: " << problem_type << std::endl;
        return 1;
    }

    return 0;
}
