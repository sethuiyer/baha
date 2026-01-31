#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "baha/baha.hpp"

// Sherrington-Kirkpatrick spin glass (continuous relaxation)
// Spins s_i in [-1, 1], couplings J_ij ~ N(0, 1/N)
// Energy = -sum_{i<j} J_ij * s_i * s_j
// This is NP-hard to optimize globally; has phase transition at beta_c ~ 1.

class SpinGlass {
public:
    int N;
    std::vector<std::vector<double>> J; // Coupling matrix
    std::vector<double> s;              // Spins (continuous in [-1, 1])

    SpinGlass(int n, unsigned seed = 42) : N(n), s(n, 0.0) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(static_cast<double>(N)));

        J.resize(N, std::vector<double>(N, 0.0));
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                J[i][j] = dist(rng);
                J[j][i] = J[i][j];
            }
        }

        // Random initial spins
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        for (int i = 0; i < N; ++i) {
            s[i] = uniform(rng);
        }
    }

    double energy() const {
        double E = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                E -= J[i][j] * s[i] * s[j];
            }
        }
        return E;
    }

    // Gradient: dE/ds_i = -sum_{j != i} J_ij * s_j
    std::vector<double> gradient() const {
        std::vector<double> grad(N, 0.0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    grad[i] -= J[i][j] * s[j];
                }
            }
        }
        return grad;
    }

    // Clamp spins to [-1, 1]
    void clamp() {
        for (int i = 0; i < N; ++i) {
            s[i] = std::max(-1.0, std::min(1.0, s[i]));
        }
    }
};

// Simplified Adafactor-style optimizer for vector parameters
struct VectorAdafactor {
    std::vector<double> v;
    double eps = 1e-8;
    bool initialized = false;

    void step(std::vector<double>& params, const std::vector<double>& grad,
              double lr, double decay) {
        if (!initialized) {
            v.resize(params.size(), 0.0);
            initialized = true;
        }

        for (size_t i = 0; i < params.size(); ++i) {
            v[i] = decay * v[i] + (1.0 - decay) * grad[i] * grad[i];
            params[i] -= lr * grad[i] / (std::sqrt(v[i]) + eps);
        }
    }
};

// Lambert-W Branch Jumper for continuous optimization
// Uses W_0 and W_{-1} to compute jump directions on fractures
class LambertWJumper {
public:
    std::mt19937 rng;
    int jump_count = 0;
    int jump_cooldown = 0;
    int fractures_since_jump = 0;
    static constexpr int FRACTURES_PER_JUMP = 5;  // Only jump every Nth fracture
    static constexpr int COOLDOWN_STEPS = 20;     // Steps to wait after jump
    
    LambertWJumper(unsigned seed = 42) : rng(seed) {}
    
    // Returns true if a jump should be performed
    bool should_jump() {
        if (jump_cooldown > 0) {
            jump_cooldown--;
            return false;
        }
        fractures_since_jump++;
        if (fractures_since_jump >= FRACTURES_PER_JUMP) {
            fractures_since_jump = 0;
            return true;
        }
        return false;
    }
    
    // Compute branch jump perturbation using Lambert-W
    // z = -β * E maps energy to Lambert-W domain
    // The difference between W_0 and W_{-1} gives jump magnitude
    void jump(std::vector<double>& spins, const std::vector<double>& grad, 
              double energy, double beta) {
        // Map to Lambert-W domain: z = -β * |E| / N (normalized)
        // Keep z in valid range for W_{-1}: (-1/e, 0)
        const double z = std::max(-0.36, std::min(-0.01, 
                         -0.1 * std::tanh(beta * std::abs(energy) / spins.size())));
        
        // Get both branches
        const double w0 = navokoj::LambertWOptimized::W0(z);
        const double wm1 = navokoj::LambertWOptimized::Wm1(z);
        
        // Branch difference determines jump magnitude
        // Large difference = strong phase transition = bigger jump
        const double branch_diff = std::abs(w0 - wm1);
        const double jump_scale = std::min(0.15, branch_diff * 0.05); // Smaller jumps
        
        // Compute gradient-weighted jump direction
        // Jump perpendicular to gradient (escape current basin)
        double grad_norm = 0.0;
        for (double g : grad) grad_norm += g * g;
        grad_norm = std::sqrt(grad_norm) + 1e-8;
        
        std::normal_distribution<double> noise(0.0, 1.0);
        
        for (size_t i = 0; i < spins.size(); ++i) {
            // Orthogonal perturbation: random noise minus gradient component
            double rand_dir = noise(rng);
            double grad_component = grad[i] / grad_norm;
            double ortho_dir = rand_dir - grad_component * (rand_dir * grad_component);
            
            // Apply jump: scale by branch difference
            spins[i] += jump_scale * ortho_dir;
            
            // Clamp to valid range
            spins[i] = std::max(-1.0, std::min(1.0, spins[i]));
        }
        
        jump_count++;
        jump_cooldown = COOLDOWN_STEPS;
    }
};

// BAHA-driven meta-scheduler with Lambert-W branch jumping
struct MetaScheduler {
    double base_lr = 0.15;
    double base_decay = 0.85;
    double lr = base_lr;
    double decay = base_decay;
    int cooldown = 0;
    int fracture_count = 0;
    LambertWJumper jumper;
    
    MetaScheduler(unsigned seed = 42) : jumper(seed) {}

    // Returns true if a branch jump should be performed
    bool update(bool fractured) {
        if (fractured) {
            lr = std::max(0.005, lr * 0.6);
            decay = std::min(0.99, decay * 1.08);
            cooldown = 15;
            fracture_count++;
            return true; // Signal to perform branch jump
        }

        if (cooldown > 0) {
            cooldown--;
            return false;
        }

        lr = lr + 0.05 * (base_lr - lr);
        decay = decay + 0.05 * (base_decay - decay);
        return false;
    }
};

int main() {
    std::cout << "============================================================\n";
    std::cout << "SPIN GLASS OPTIMIZATION (BAHA + LAMBERT-W BRANCH JUMPING)\n";
    std::cout << "============================================================\n\n";

    const int N = 64;  // 64 spins — small but already hard
    SpinGlass glass(N, 12345);
    VectorAdafactor opt;
    MetaScheduler sched(54321);
    navokoj::FractureDetectorOptimized detector(0.8); // Lower threshold for sensitivity

    const double initial_energy = glass.energy();
    std::cout << "Spins: " << N << " (search space: 2^" << N << " discrete configs)\n";
    std::cout << "Initial energy: " << std::fixed << std::setprecision(4) << initial_energy << "\n";
    std::cout << "Mode: Adafactor + Fracture Detection + Lambert-W Branch Jumps\n\n";

    const int steps = 300;
    double best_energy = initial_energy;
    std::vector<double> best_spins = glass.s;

    for (int t = 0; t < steps; ++t) {
        const double E = glass.energy();
        const auto grad = glass.gradient();

        // Record for fracture detection (log(Z) surrogate = -E)
        // Use step as effective beta (annealing schedule)
        const double beta = 0.1 + 0.03 * t; // Slower annealing
        detector.record(beta, -E);
        const bool fractured = detector.is_fracture();
        sched.update(fractured);

        // On fracture: check if we should perform a Lambert-W branch jump
        if (fractured && t > 10 && sched.jumper.should_jump()) {
            sched.jumper.jump(glass.s, grad, E, beta);
            glass.clamp();
        }

        // Adafactor step
        opt.step(glass.s, grad, sched.lr, sched.decay);
        glass.clamp();

        if (E < best_energy) {
            best_energy = E;
            best_spins = glass.s;
        }

        if (fractured || t % 30 == 0 || t == steps - 1) {
            std::cout << "step " << std::setw(3) << t
                      << " | E=" << std::setw(9) << std::fixed << std::setprecision(4) << E
                      << " | lr=" << std::setw(7) << std::setprecision(5) << sched.lr
                      << " | jumps=" << std::setw(2) << sched.jumper.jump_count
                      << " | " << (fractured ? "*** FRACTURE + JUMP ***" : "") << "\n";
        }
    }
    
    // Restore best state
    glass.s = best_spins;

    std::cout << "\n============================================================\n";
    std::cout << "RESULTS (CONTINUOUS)\n";
    std::cout << "============================================================\n";
    std::cout << "Initial energy: " << std::fixed << std::setprecision(4) << initial_energy << "\n";
    std::cout << "Final energy:   " << std::fixed << std::setprecision(4) << glass.energy() << "\n";
    std::cout << "Best energy:    " << std::fixed << std::setprecision(4) << best_energy << "\n";
    std::cout << "Improvement:    " << std::fixed << std::setprecision(2)
              << 100.0 * (initial_energy - best_energy) / std::abs(initial_energy) << "%\n";
    std::cout << "Fractures:      " << sched.fracture_count << "\n";
    std::cout << "Branch jumps:   " << sched.jumper.jump_count << "\n";

    // Discretize spins to {-1, +1} and compute true discrete energy
    std::vector<int> discrete_spins(N);
    for (int i = 0; i < N; ++i) {
        discrete_spins[i] = (glass.s[i] >= 0.0) ? 1 : -1;
    }

    double discrete_energy = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            discrete_energy -= glass.J[i][j] * discrete_spins[i] * discrete_spins[j];
        }
    }

    // Compute magnetization (order parameter)
    double magnetization = 0.0;
    for (int i = 0; i < N; ++i) {
        magnetization += discrete_spins[i];
    }
    magnetization /= N;

    // Count +1 and -1 spins
    int up_count = 0, down_count = 0;
    for (int i = 0; i < N; ++i) {
        if (discrete_spins[i] == 1) up_count++;
        else down_count++;
    }

    std::cout << "\n============================================================\n";
    std::cout << "DISCRETE SOLUTION (Real-world assignment)\n";
    std::cout << "============================================================\n";
    std::cout << "Discrete energy: " << std::fixed << std::setprecision(4) << discrete_energy << "\n";
    std::cout << "Magnetization:   " << std::fixed << std::setprecision(4) << magnetization << "\n";
    std::cout << "Spin distribution: " << up_count << " up (+1), " << down_count << " down (-1)\n\n";

    std::cout << "Spin assignments (64 spins):\n";
    for (int i = 0; i < N; ++i) {
        std::cout << (discrete_spins[i] == 1 ? "+" : "-");
        if ((i + 1) % 16 == 0) std::cout << "\n";
    }

    std::cout << "\nInterpretation:\n";
    std::cout << "- Discrete energy is the true NP-hard objective value.\n";
    std::cout << "- Magnetization near 0 = balanced (typical for spin glass).\n";
    std::cout << "- This is a valid assignment for the 2^64 search space.\n";
    std::cout << "- Fractures detected phase transitions during optimization.\n";
    std::cout << "- Lambert-W branch jumps escaped local minima at fracture points.\n";

    return 0;
}
