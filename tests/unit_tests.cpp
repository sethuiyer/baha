#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <functional>
#include <iomanip>
#include <limits>

#include "baha/baha.hpp"

// =============================================================================
// MINIMAL TEST HARNESS
// =============================================================================

int g_tests_passed = 0;
int g_tests_failed = 0;

#define TEST_CASE(name) \
    void test_##name(); \
    int main_##name = (test_##name(), 0); \
    void test_##name() { \
        std::cout << "[TEST] " << #name << "..." << std::endl; \

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            std::cout << "  FAILED: " << #condition << " at line " << __LINE__ << std::endl; \
            g_tests_failed++; \
        } else { \
            g_tests_passed++; \
        } \
    } while(0)

#define CHECK_CLOSE(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            std::cout << "  FAILED: " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ") at line " << __LINE__ << std::endl; \
            g_tests_failed++; \
        } else { \
            g_tests_passed++; \
        } \
    } while(0)

#define REQUIRE(condition) \
    do { \
        if (!(condition)) { \
            std::cout << "  FATAL: " << #condition << " at line " << __LINE__ << std::endl; \
            g_tests_failed++; \
            return; \
        } else { \
            g_tests_passed++; \
        } \
    } while(0)

// =============================================================================
// LAMBERT-W TESTS
// =============================================================================

void test_lambert_w0_identity() {
    std::cout << "[TEST] LambertW W0 Identity check..." << std::endl;
    // Check W(z) * exp(W(z)) = z
    std::vector<double> test_values = {0.1, 0.5, 1.0, 2.0, 10.0, -0.1, -0.2, -0.3};
    for (double z : test_values) {
        double w = navokoj::LambertW::W0(z);
        double check = w * std::exp(w);
        CHECK_CLOSE(check, z, 1e-8);
    }
}

void test_lambert_w0_special_values() {
    std::cout << "[TEST] LambertW W0 Special Values..." << std::endl;
    // W0(0) = 0
    CHECK_CLOSE(navokoj::LambertW::W0(0.0), 0.0, 1e-10);
    // W0(e) = 1
    CHECK_CLOSE(navokoj::LambertW::W0(std::exp(1.0)), 1.0, 1e-10);
    // W0(-1/e) = -1
    CHECK_CLOSE(navokoj::LambertW::W0(-navokoj::LambertW::E_INV), -1.0, 1e-8);
}

void test_lambert_wm1_identity() {
    std::cout << "[TEST] LambertW Wm1 Identity check..." << std::endl;
    // Domain [-1/e, 0). W is <= -1.
    std::vector<double> test_values = {-0.01, -0.1, -0.2, -0.36}; /* -1/e ~= -0.3678 */
    for (double z : test_values) {
        double w = navokoj::LambertW::Wm1(z);
        double check = w * std::exp(w);
        CHECK_CLOSE(check, z, 1e-8);
        CHECK(w <= -1.0); 
    }
}

// =============================================================================
// MATH UTIL TESTS
// =============================================================================

void test_log_sum_exp() {
    std::cout << "[TEST] LogSumExp Stability..." << std::endl;
    std::vector<double> vals = {1000.0, 1000.0, 1000.0}; 
    // real sum = 3 * e^1000. log sum = log(3) + 1000.
    double expected = std::log(3.0) + 1000.0;
    CHECK_CLOSE(navokoj::log_sum_exp(vals), expected, 1e-10);

    // Mixed scales
    vals = {0.0, -100.0}; // -100 is negligible
    CHECK_CLOSE(navokoj::log_sum_exp(vals), 0.0, 1e-10); // log(e^0 + e^-100) ~= log(1) = 0
}

// =============================================================================
// FRACTURE DETECTOR TESTS
// =============================================================================

void test_fracture_detection() {
    std::cout << "[TEST] FractureDetector Logic..." << std::endl;
    navokoj::FractureDetector detector(1.0); // Threshold 1.0
    
    // Constant slope (no fracture)
    // Beta: 0 -> 1 -> 2
    // LogZ: 0 -> 0.5 -> 1.0 (Slope 0.5)
    detector.record(0.0, 0.0);
    detector.record(1.0, 0.5);
    CHECK(detector.fracture_rate() == 0.5);
    CHECK(detector.is_fracture() == false);
    
    // Sudden jump
    // Beta: 1.0 -> 1.1 (Delta 0.1)
    // LogZ: 0.5 -> 1.5 (Delta 1.0) => Rate = 10.0
    detector.record(1.1, 1.5);
    CHECK_CLOSE(detector.fracture_rate(), 10.0, 1e-10);
    CHECK(detector.is_fracture() == true);
}

// =============================================================================
// OPTIMIZER INTEGRATION TEST
// =============================================================================

struct ScalarState {
    double x;
};

void test_optimizer_double_well() {
    std::cout << "[TEST] Optimization: Double Well Potential..." << std::endl;
    
    // E(x) = (x^2 - 1)^2
    // Minima at x = -1 and x = +1, E = 0.
    // Barrier at x = 0, E = 1.
    auto energy = [](const ScalarState& s) {
        double t = s.x * s.x - 1.0;
        return t * t;
    };
    
    auto sampler = []() {
        return ScalarState{0.0}; // Start at barrier peak
    };
    
    // Simple Gaussian mutation
    auto neighbors = [](const ScalarState& s) {
        return std::vector<ScalarState>{ 
            {s.x - 0.1}, {s.x + 0.1}
        };
    };
    
    navokoj::BranchAwareOptimizer<ScalarState> opt(energy, sampler, neighbors);
    
    // We expect it to slide down to either -1 or 1 quickly.
    // This tests the core loop, not necessarily fracture jumping (which needs complex phase space).
    navokoj::BranchAwareOptimizer<ScalarState>::Config config;
    config.beta_steps = 50;
    config.beta_end = 5.0;
    
    auto result = opt.optimize(config);
    
    CHECK(result.best_energy < 0.01);
    CHECK(std::abs(std::abs(result.best_state.x) - 1.0) < 0.1);
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   RUNNING BAHA UNIT TESTS" << std::endl;
    std::cout << "========================================" << std::endl;

    test_lambert_w0_special_values();
    test_lambert_w0_identity();
    test_lambert_wm1_identity();
    test_log_sum_exp();
    test_fracture_detection();
    test_optimizer_double_well();

    std::cout << "========================================" << std::endl;
    std::cout << "Passed: " << g_tests_passed << std::endl;
    std::cout << "Failed: " << g_tests_failed << std::endl;
    
    if (g_tests_failed > 0) return 1;
    return 0;
}
