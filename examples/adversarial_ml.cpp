/*
 * Adversarial ML Example Generation using BAHA
 * Find minimal perturbations to fool a neural network classifier
 * 
 * This demonstrates BAHA's ability to detect phase transitions at decision boundaries
 * and exploit rare high-signal events (adversarial examples are rare but critical).
 */

#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

// Simple neural network classifier (linear + ReLU for demonstration)
class SimpleClassifier {
    std::vector<std::vector<double>> weights;  // weights[i][j] = weight from input j to class i
    std::vector<double> biases;
    int n_classes;
    int input_dim;
    
public:
    SimpleClassifier(int dim, int classes, int seed = 42) 
        : input_dim(dim), n_classes(classes) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, 0.1);
        
        weights.assign(n_classes, std::vector<double>(input_dim));
        biases.assign(n_classes, 0.0);
        
        for (int i = 0; i < n_classes; ++i) {
            for (int j = 0; j < input_dim; ++j) {
                weights[i][j] = dist(rng);
            }
            biases[i] = dist(rng);
        }
    }
    
    // Predict class probabilities
    std::vector<double> predict(const std::vector<double>& x) const {
        std::vector<double> logits(n_classes);
        
        for (int i = 0; i < n_classes; ++i) {
            double sum = biases[i];
            for (int j = 0; j < input_dim; ++j) {
                sum += weights[i][j] * x[j];
            }
            logits[i] = sum;
        }
        
        // Softmax
        double max_logit = *std::max_element(logits.begin(), logits.end());
        std::vector<double> exp_logits(n_classes);
        double sum_exp = 0.0;
        for (int i = 0; i < n_classes; ++i) {
            exp_logits[i] = std::exp(logits[i] - max_logit);
            sum_exp += exp_logits[i];
        }
        
        std::vector<double> probs(n_classes);
        for (int i = 0; i < n_classes; ++i) {
            probs[i] = exp_logits[i] / sum_exp;
        }
        
        return probs;
    }
    
    // Get predicted class
    int predict_class(const std::vector<double>& x) const {
        auto probs = predict(x);
        return std::max_element(probs.begin(), probs.end()) - probs.begin();
    }
    
    // Get confidence (max probability)
    double confidence(const std::vector<double>& x) const {
        auto probs = predict(x);
        return *std::max_element(probs.begin(), probs.end());
    }
    
    // Get margin (difference between top 2 classes)
    double margin(const std::vector<double>& x) const {
        auto probs = predict(x);
        std::sort(probs.begin(), probs.end(), std::greater<double>());
        return probs[0] - probs[1];
    }
};

struct AdversarialState {
    std::vector<double> perturbation;  // delta to add to original image
    int input_dim;
    
    AdversarialState() : input_dim(0) {}
    AdversarialState(int dim) : input_dim(dim), perturbation(dim, 0.0) {}
};

struct AdversarialProblem {
    SimpleClassifier classifier;
    std::vector<double> original_image;
    int true_label;
    int target_label;  // -1 for untargeted attack
    double epsilon_max;  // Maximum allowed perturbation (L_inf norm)
    double lambda;  // Trade-off between fooling and perturbation size
    
    AdversarialProblem(int dim, int n_classes, int true_class, 
                      int target = -1, double eps = 0.1, double lam = 10.0)
        : classifier(dim, n_classes), true_label(true_class), 
          target_label(target), epsilon_max(eps), lambda(lam) {
        
        // Generate random original image (normalized to [0, 1])
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        original_image.assign(dim, 0.0);
        for (int i = 0; i < dim; ++i) {
            original_image[i] = dist(rng);
        }
        
        // Verify original classification
        int pred = classifier.predict_class(original_image);
        std::cout << "Original image: predicted class " << pred 
                  << " (true: " << true_label << ")" << std::endl;
        std::cout << "Original confidence: " 
                  << std::fixed << std::setprecision(4)
                  << classifier.confidence(original_image) << std::endl;
    }
    
    // Get perturbed image
    std::vector<double> perturbed_image(const AdversarialState& state) const {
        std::vector<double> x = original_image;
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = std::max(0.0, std::min(1.0, x[i] + state.perturbation[i]));
        }
        return x;
    }
    
    // Energy function: minimize to find adversarial example
    // Lower energy = better adversarial example
    double energy(const AdversarialState& state) const {
        auto x_pert = perturbed_image(state);
        int pred = classifier.predict_class(x_pert);
        
        // Check L_inf constraint
        double l_inf = 0.0;
        for (double p : state.perturbation) {
            l_inf = std::max(l_inf, std::abs(p));
        }
        if (l_inf > epsilon_max) {
            return 1e6;  // Penalty for violating constraint
        }
        
        // Check if we've fooled the classifier
        bool is_adversarial = (target_label == -1) 
            ? (pred != true_label)  // Untargeted: any misclassification
            : (pred == target_label);  // Targeted: specific class
        
        if (is_adversarial) {
            // Success! Minimize perturbation size
            double perturbation_norm = 0.0;
            for (double p : state.perturbation) {
                perturbation_norm += p * p;
            }
            return perturbation_norm;  // Minimize L2 norm of perturbation
        } else {
            // Not adversarial yet - minimize confidence in true class
            double conf = classifier.confidence(x_pert);
            double margin_val = classifier.margin(x_pert);
            
            // Energy = high confidence penalty + margin penalty
            // Want to push decision boundary
            return lambda * (conf + margin_val) + l_inf * 100.0;
        }
    }
    
    // Check if state is adversarial
    bool is_adversarial(const AdversarialState& state) const {
        auto x_pert = perturbed_image(state);
        int pred = classifier.predict_class(x_pert);
        
        if (target_label == -1) {
            return pred != true_label;
        } else {
            return pred == target_label;
        }
    }
    
    // Get perturbation statistics
    void print_stats(const AdversarialState& state) const {
        auto x_pert = perturbed_image(state);
        int pred = classifier.predict_class(x_pert);
        double conf = classifier.confidence(x_pert);
        
        double l_inf = 0.0;
        double l2 = 0.0;
        for (double p : state.perturbation) {
            l_inf = std::max(l_inf, std::abs(p));
            l2 += p * p;
        }
        l2 = std::sqrt(l2);
        
        std::cout << "  Perturbed prediction: " << pred << std::endl;
        std::cout << "  Confidence: " << std::fixed << std::setprecision(4) << conf << std::endl;
        std::cout << "  L_inf norm: " << std::setprecision(6) << l_inf << std::endl;
        std::cout << "  L2 norm: " << std::setprecision(6) << l2 << std::endl;
        std::cout << "  Is adversarial: " << (is_adversarial(state) ? "YES" : "NO") << std::endl;
    }
};

int main() {
    std::cout << "🎯 Adversarial ML Example Generation with BAHA\n";
    std::cout << "================================================\n\n";
    
    const int input_dim = 100;  // Image dimension (e.g., 10x10 grayscale)
    const int n_classes = 10;
    const int true_label = 3;
    const int target_label = -1;  // -1 for untargeted attack
    const double epsilon_max = 0.1;  // Max perturbation per pixel
    const double lambda = 10.0;  // Trade-off parameter
    
    AdversarialProblem problem(input_dim, n_classes, true_label, 
                              target_label, epsilon_max, lambda);
    
    // Energy function
    auto energy = [&](const AdversarialState& s) {
        return problem.energy(s);
    };
    
    // Sampler: start with zero perturbation
    auto sampler = [&]() {
        AdversarialState state(input_dim);
        return state;
    };
    
    // Neighbors: small random changes to perturbation
    auto neighbors = [&](const AdversarialState& s) {
        std::vector<AdversarialState> nbrs;
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<double> dist(0.0, 0.01);
        
        // Generate 10 neighbors
        for (int i = 0; i < 10; ++i) {
            AdversarialState nbr = s;
            // Randomly modify a few perturbation components
            for (int j = 0; j < input_dim; ++j) {
                if (rng() % 10 == 0) {  // Modify ~10% of components
                    nbr.perturbation[j] += dist(rng);
                    // Clamp to [-epsilon_max, epsilon_max]
                    nbr.perturbation[j] = std::max(-epsilon_max, 
                                                   std::min(epsilon_max, nbr.perturbation[j]));
                }
            }
            nbrs.push_back(nbr);
        }
        
        return nbrs;
    };
    
    // Create BAHA optimizer
    navokoj::BranchAwareOptimizer<AdversarialState> optimizer(energy, sampler, neighbors);
    
    // Configure BAHA
    navokoj::BranchAwareOptimizer<AdversarialState>::Config config;
    config.beta_start = 0.01;
    config.beta_end = 50.0;
    config.beta_steps = 2000;
    config.fracture_threshold = 2.0;
    config.samples_per_beta = 50;
    config.schedule_type = navokoj::BranchAwareOptimizer<AdversarialState>::ScheduleType::GEOMETRIC;
    
    std::cout << "Running BAHA optimization...\n";
    std::cout << "Target: " << (target_label == -1 ? "Untargeted attack" : 
                                "Target class " + std::to_string(target_label)) << "\n";
    std::cout << "Max perturbation (L_inf): " << epsilon_max << "\n\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = optimizer.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n=== RESULTS ===\n";
    std::cout << "Best energy: " << std::fixed << std::setprecision(6) 
              << result.best_energy << "\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Solve time: " << duration.count() << " ms\n\n";
    
    std::cout << "Best adversarial example:\n";
    problem.print_stats(result.best_state);
    
    if (problem.is_adversarial(result.best_state)) {
        std::cout << "\n✅ SUCCESS: Found adversarial example!\n";
        std::cout << "BAHA successfully detected phase transitions at the decision boundary\n";
        std::cout << "and exploited rare high-signal events (adversarial examples).\n";
    } else {
        std::cout << "\n⚠️  Did not find adversarial example (may need more iterations or different parameters)\n";
    }
    
    return 0;
}
