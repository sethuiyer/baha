#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "baha/baha.hpp"

struct Params {
    double x = 2.5;
    double y = -1.8;
};

struct Grad {
    double dx = 0.0;
    double dy = 0.0;
};

// Toy objective with ripples + ridge (non-convex)
static double loss(const Params& p) {
    const double bowl = 0.5 * (p.x * p.x + p.y * p.y);
    const double ripples = 0.3 * std::sin(3.0 * p.x) * std::sin(3.0 * p.y);
    const double ridge = 0.15 * std::exp(-0.5 * (p.x - 1.2) * (p.x - 1.2)) * (p.y * p.y);
    return bowl + ripples + ridge;
}

static Grad grad(const Params& p) {
    Grad g;
    const double exp_term = std::exp(-0.5 * (p.x - 1.2) * (p.x - 1.2));

    g.dx = p.x
           + 0.9 * std::cos(3.0 * p.x) * std::sin(3.0 * p.y)
           + 0.15 * (p.y * p.y) * exp_term * (-(p.x - 1.2));
    g.dy = p.y
           + 0.9 * std::sin(3.0 * p.x) * std::cos(3.0 * p.y)
           + 0.30 * p.y * exp_term;
    return g;
}

// Simplified Adafactor-style update for vector parameters.
// Assumption: factored second moment reduces to RMS for a 2D vector.
struct ToyAdafactor {
    double v_x = 0.0;
    double v_y = 0.0;
    double eps = 1e-8;

    void step(Params& p, const Grad& g, double lr, double decay) {
        v_x = decay * v_x + (1.0 - decay) * g.dx * g.dx;
        v_y = decay * v_y + (1.0 - decay) * g.dy * g.dy;

        p.x -= lr * g.dx / (std::sqrt(v_x) + eps);
        p.y -= lr * g.dy / (std::sqrt(v_y) + eps);
    }
};

// BAHA-driven meta-scheduler: fracture events adjust optimizer hyperparams.
struct MetaScheduler {
    double base_lr = 0.08;
    double base_decay = 0.90;
    double lr = base_lr;
    double decay = base_decay;
    int cooldown = 0;

    void update(bool fractured) {
        if (fractured) {
            lr = std::max(0.01, lr * 0.7);
            decay = std::min(0.99, decay * 1.05);
            cooldown = 10;
            return;
        }

        // Fallback behavior: if stable for a while, relax back to base settings.
        if (cooldown > 0) {
            cooldown--;
            return;
        }

        lr = lr + 0.1 * (base_lr - lr);
        decay = decay + 0.1 * (base_decay - decay);
    }
};

int main() {
    std::cout << "============================================================\n";
    std::cout << "ADAFECTOR META-SCHEDULER (BAHA FRACTURE-AWARE)\n";
    std::cout << "============================================================\n\n";

    Params p;
    ToyAdafactor opt;
    MetaScheduler sched;
    navokoj::FractureDetectorOptimized detector(1.2);

    const int steps = 180;
    for (int t = 0; t < steps; ++t) {
        const double l = loss(p);
        const Grad g = grad(p);

        // Assumption: log(Z) surrogate is -loss for detection; fractures indicate sharp regime shifts.
        detector.record(static_cast<double>(t), -l);
        const bool fractured = detector.is_fracture();
        sched.update(fractured);

        opt.step(p, g, sched.lr, sched.decay);

        if (fractured || t % 20 == 0) {
            std::cout << "step " << std::setw(3) << t
                      << " | loss=" << std::setw(9) << std::fixed << std::setprecision(5) << l
                      << " | lr=" << std::setw(6) << std::setprecision(4) << sched.lr
                      << " | decay=" << std::setw(6) << std::setprecision(4) << sched.decay
                      << " | fracture=" << (fractured ? "yes" : "no") << "\n";
        }
    }

    std::cout << "\nFinal params: x=" << p.x << ", y=" << p.y << "\n";
    std::cout << "Final loss: " << loss(p) << "\n";
    std::cout << "\nNotes:\n";
    std::cout << "- Fractures trigger conservative scheduling (lower lr, higher decay).\n";
    std::cout << "- Stable regions relax back toward base hyperparameters.\n";
    std::cout << "- This is a proof-of-concept for inference-time schedulers.\n";
    return 0;
}
