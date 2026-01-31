#include <cassert>
#include <cmath>
#include <iostream>

#include "baha/baha.hpp"

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

        if (cooldown > 0) {
            cooldown--;
            return;
        }

        lr = lr + 0.1 * (base_lr - lr);
        decay = decay + 0.1 * (base_decay - decay);
    }
};

int main() {
    navokoj::FractureDetectorOptimized detector(0.5);
    MetaScheduler sched;

    // Assumption: log(Z) surrogate is -loss; large jumps should trigger fractures.
    detector.record(0.0, 0.0);
    detector.record(1.0, -10.0);

    const bool fractured = detector.is_fracture();
    sched.update(fractured);

    assert(fractured);
    assert(sched.lr < sched.base_lr);
    assert(sched.decay > sched.base_decay);

    std::cout << "Meta-scheduler fracture test passed.\n";
    return 0;
}
