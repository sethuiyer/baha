/*
 * Course Scheduling Problem using BAHA
 * Assign courses to time slots avoiding conflicts (students, rooms, instructors)
 */
#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <set>
#include <numeric>

struct ScheduleState {
    std::vector<int> assignments;  // assignments[i] = time slot for course i
    int n_courses;
    int n_slots;
    
    ScheduleState() : n_courses(0), n_slots(0) {}
    ScheduleState(int courses, int slots) : n_courses(courses), n_slots(slots), assignments(courses, -1) {}
};

struct Course {
    std::vector<int> students;  // Students enrolled
    int instructor;
    int room_requirement;  // Room type needed
};

int main(int argc, char** argv) {
    int n_courses = (argc > 1) ? std::stoi(argv[1]) : 20;
    int n_slots = (argc > 2) ? std::stoi(argv[2]) : 8;
    int n_students = (argc > 3) ? std::stoi(argv[3]) : 50;
    
    // Generate courses with random enrollments
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> enrollment_dist(10, 30);
    std::uniform_int_distribution<int> instructor_dist(0, 9);
    std::uniform_int_distribution<int> room_dist(0, 2);
    
    std::vector<Course> courses(n_courses);
    for (int i = 0; i < n_courses; ++i) {
        int enrollment = enrollment_dist(rng);
        courses[i].students.resize(enrollment);
        for (int j = 0; j < enrollment; ++j) {
            courses[i].students[j] = rng() % n_students;
        }
        courses[i].instructor = instructor_dist(rng);
        courses[i].room_requirement = room_dist(rng);
    }
    
    std::cout << "============================================================\n";
    std::cout << "COURSE SCHEDULING PROBLEM: " << n_courses << " courses, " 
              << n_slots << " time slots, " << n_students << " students\n";
    std::cout << "============================================================\n";
    
    // Energy: count conflicts (student overlaps, instructor overlaps, room conflicts)
    auto energy = [&courses, n_slots](const ScheduleState& s) -> double {
        int conflicts = 0;
        
        // Student conflicts: same student in multiple courses at same time
        for (int slot = 0; slot < n_slots; ++slot) {
            std::set<int> students_in_slot;
            for (int c = 0; c < s.n_courses; ++c) {
                if (s.assignments[c] == slot) {
                    for (int student : courses[c].students) {
                        if (students_in_slot.count(student)) {
                            conflicts++;
                        }
                        students_in_slot.insert(student);
                    }
                }
            }
        }
        
        // Instructor conflicts: same instructor teaching multiple courses at same time
        for (int slot = 0; slot < n_slots; ++slot) {
            std::set<int> instructors_in_slot;
            for (int c = 0; c < s.n_courses; ++c) {
                if (s.assignments[c] == slot) {
                    if (instructors_in_slot.count(courses[c].instructor)) {
                        conflicts++;
                    }
                    instructors_in_slot.insert(courses[c].instructor);
                }
            }
        }
        
        // Room conflicts: too many courses needing same room type at same time
        for (int slot = 0; slot < n_slots; ++slot) {
            std::vector<int> room_usage(3, 0);
            for (int c = 0; c < s.n_courses; ++c) {
                if (s.assignments[c] == slot) {
                    room_usage[courses[c].room_requirement]++;
                }
            }
            // Assume 2 rooms of each type available
            for (int r = 0; r < 3; ++r) {
                if (room_usage[r] > 2) {
                    conflicts += (room_usage[r] - 2);
                }
            }
        }
        
        // Penalty for unassigned courses
        int unassigned = std::count(s.assignments.begin(), s.assignments.end(), -1);
        
        return static_cast<double>(conflicts) + unassigned * 100.0;
    };
    
    // Random initial assignment
    auto sampler = [n_courses, n_slots]() -> ScheduleState {
        ScheduleState s(n_courses, n_slots);
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> slot_dist(0, n_slots - 1);
        
        for (int i = 0; i < n_courses; ++i) {
            s.assignments[i] = slot_dist(rng);
        }
        
        return s;
    };
    
    // Neighbors: change course time slot, or swap two courses
    auto neighbors = [n_courses, n_slots](const ScheduleState& s) -> std::vector<ScheduleState> {
        std::vector<ScheduleState> nbrs;
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> course_dist(0, n_courses - 1);
        std::uniform_int_distribution<int> slot_dist(0, n_slots - 1);
        
        for (int k = 0; k < 30; ++k) {
            ScheduleState nbr = s;
            int op = rng() % 2;
            
            if (op == 0) {
                // Change one course's time slot
                int course = course_dist(rng);
                nbr.assignments[course] = slot_dist(rng);
            } else {
                // Swap two courses' time slots
                int c1 = course_dist(rng);
                int c2 = course_dist(rng);
                std::swap(nbr.assignments[c1], nbr.assignments[c2]);
            }
            
            nbrs.push_back(nbr);
        }
        
        return nbrs;
    };
    
    navokoj::BranchAwareOptimizer<ScheduleState> opt(energy, sampler, neighbors);
    
    typename navokoj::BranchAwareOptimizer<ScheduleState>::Config config;
    config.beta_steps = 500;
    config.beta_end = 12.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.7;
    config.max_branches = 5;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<ScheduleState>::ScheduleType::GEOMETRIC;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = opt.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    // Count final conflicts
    int final_conflicts = static_cast<int>(result.best_energy);
    int unassigned = std::count(result.best_state.assignments.begin(), 
                                result.best_state.assignments.end(), -1);
    final_conflicts -= unassigned * 100;
    
    std::cout << "\nResult:\n";
    std::cout << "Conflicts: " << final_conflicts << "\n";
    std::cout << "Unassigned courses: " << unassigned << "\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed << "s\n";
    
    if (final_conflicts == 0 && unassigned == 0) {
        std::cout << "\n✅ VALID SCHEDULE FOUND!\n\n";
        for (int slot = 0; slot < n_slots; ++slot) {
            std::cout << "Time Slot " << slot << ": ";
            bool first = true;
            for (int c = 0; c < n_courses; ++c) {
                if (result.best_state.assignments[c] == slot) {
                    if (!first) std::cout << ", ";
                    std::cout << "C" << c;
                    first = false;
                }
            }
            std::cout << "\n";
        }
    }
    
    return 0;
}
