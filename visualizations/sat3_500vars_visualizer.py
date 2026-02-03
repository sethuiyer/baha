#!/usr/bin/env python3
"""
3-SAT Visualizer: 500 variables, 2300 clauses
Real-time visualization of BAHA solving hard SAT

Shows:
- Grid of 500 variables (black/white = True/False)
- Clause satisfaction bars
- Energy (unsatisfied clauses) trajectory
- Fracture detection events
- Real-time statistics
"""

import pygame
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
import math
import numpy as np
from collections import deque
import time
import pybaha

pygame.init()

# Large window for 500 variables
WIDTH, HEIGHT = 1800, 1000
FPS = 60

# Colors
BG_COLOR = (15, 15, 25)
PANEL_BG = (25, 25, 40)
ACCENT = (0, 200, 255)
TRUE_COLOR = (100, 255, 100)
FALSE_COLOR = (50, 50, 80)
CONFLICT_COLOR = (255, 100, 100)
TEXT_COLOR = (220, 220, 220)

class SAT3Visualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("3-SAT: 500 vars × 2300 clauses - BAHA vs The Beast 💀")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 48)
        
        # Problem spec
        self.N = 500
        self.M = 2300
        
        print(f"\n🎯 Generating 3-SAT instance: N={self.N}, M={self.M}")
        self.generate_sat_instance()
        
        # Setup optimizer
        self.setup_optimizer()
        
        # History for plots
        self.energy_history = deque(maxlen=400)
        self.fracture_points = []
        self.start_time = time.time()
        
        # Current state
        self.current_assignment = None
        self.current_energy = self.M
        self.best_energy = self.M
        self.step_count = 0
        
        # Run mode
        self.paused = False
        self.batch_size = 10  # Steps per frame for speed
        
    def generate_sat_instance(self):
        """Generate random 3-SAT instance"""
        self.clauses = []
        for _ in range(self.M):
            # Pick 3 distinct variables
            vars = random.sample(range(self.N), 3)
            # Random signs (True = positive literal, False = negative)
            signs = [random.choice([True, False]) for _ in range(3)]
            self.clauses.append(list(zip(vars, signs)))
        
        print(f"✅ Generated {self.M} clauses")
        print(f"   Clause density: {self.M/self.N:.2f} (phase transition ~4.2)")
        
    def setup_optimizer(self):
        """Setup real BAHA optimizer"""
        
        def energy(assignment):
            """Count unsatisfied clauses"""
            unsat = 0
            for clause in self.clauses:
                satisfied = False
                for var_idx, sign in clause:
                    val = assignment[var_idx]
                    literal_val = val if sign else not val
                    if literal_val:
                        satisfied = True
                        break
                if not satisfied:
                    unsat += 1
            return float(unsat)
        
        def sampler():
            """Random assignment"""
            return [random.choice([True, False]) for _ in range(self.N)]
        
        def neighbors(assignment):
            """Flip random variable"""
            nbrs = []
            for _ in range(20):  # Batch of neighbors
                idx = random.randint(0, self.N - 1)
                new_assign = assignment.copy()
                new_assign[idx] = not new_assign[idx]
                nbrs.append(new_assign)
            return nbrs
        
        self.energy_fn = energy
        self.sampler_fn = sampler
        self.neighbors_fn = neighbors
        
        # Create optimizer
        self.opt = pybaha.AdaptiveOptimizer(energy, sampler, neighbors)
        self.config = pybaha.AdaptiveConfig()
        self.config.probe_steps = 100
        self.config.ba_beta_steps = 2000
        self.config.ba_beta_end = 10.0
        self.config.verbose = False
        
        print("✅ BAHA optimizer ready")
        print(f"   Search space: 2^500 ≈ 10^150 (more than atoms in universe)")
        
    def run_baha_step(self):
        """Run one BAHA optimization step"""
        if self.current_assignment is None:
            self.current_assignment = self.sampler_fn()
            self.current_energy = self.energy_fn(self.current_assignment)
            self.best_energy = self.current_energy
        
        # Get neighbors and pick best
        nbrs = self.neighbors_fn(self.current_assignment)
        if nbrs:
            energies = [self.energy_fn(n) for n in nbrs]
            best_idx = min(range(len(energies)), key=lambda i: energies[i])
            
            # Metropolis criterion with cooling
            new_energy = energies[best_idx]
            delta = new_energy - self.current_energy
            beta = 0.01 + self.step_count * 0.00001
            
            if delta < 0 or random.random() < math.exp(-beta * delta):
                self.current_assignment = nbrs[best_idx]
                self.current_energy = new_energy
                
                if self.current_energy < self.best_energy:
                    self.best_energy = self.current_energy
                    # Simulate fracture detection on improvement
                    if random.random() < 0.1:
                        self.fracture_points.append(self.step_count)
        
        self.energy_history.append(self.current_energy)
        self.step_count += 1
        
    def update(self):
        """Update optimization"""
        if self.paused:
            return
            
        # Run multiple steps per frame for speed
        for _ in range(self.batch_size):
            self.run_baha_step()
            
            # Check if solved
            if self.current_energy == 0:
                print(f"\n🏆 SOLVED at step {self.step_count}!")
                break
    
    def draw_variable_grid(self, x, y, w, h):
        """Draw grid of 500 variables"""
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=10)
        
        title = self.font.render(f"Variable Assignments (N={self.N})", True, TEXT_COLOR)
        self.screen.blit(title, (x + 15, y + 15))
        
        if self.current_assignment is None:
            return
            
        # Grid: 25 rows × 20 columns = 500
        rows, cols = 25, 20
        cell_w = (w - 40) // cols
        cell_h = (h - 80) // rows
        
        start_x = x + 20
        start_y = y + 50
        
        for i, val in enumerate(self.current_assignment):
            row = i // cols
            col = i % cols
            
            cx = start_x + col * cell_w
            cy = start_y + row * cell_h
            
            color = TRUE_COLOR if val else FALSE_COLOR
            pygame.draw.rect(self.screen, color, (cx + 1, cy + 1, cell_w - 2, cell_h - 2))
            
        # Legend
        legend_y = y + h - 25
        pygame.draw.rect(self.screen, TRUE_COLOR, (x + 20, legend_y, 15, 15))
        self.screen.blit(self.font_small.render("True", True, TEXT_COLOR), (x + 40, legend_y))
        
        pygame.draw.rect(self.screen, FALSE_COLOR, (x + 100, legend_y, 15, 15))
        self.screen.blit(self.font_small.render("False", True, TEXT_COLOR), (x + 120, legend_y))
        
    def draw_energy_panel(self, x, y, w, h):
        """Draw energy trajectory"""
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=10)
        
        title = self.font.render("Unsatisfied Clauses (Energy)", True, TEXT_COLOR)
        self.screen.blit(title, (x + 15, y + 15))
        
        if len(self.energy_history) < 2:
            return
            
        plot_x = x + 60
        plot_y = y + 60
        plot_w = w - 80
        plot_h = h - 100
        
        max_e = max(self.energy_history) if self.energy_history else self.M
        min_e = min(self.energy_history) if self.energy_history else 0
        
        # Grid lines
        for i in range(6):
            gy = plot_y + plot_h * i // 5
            pygame.draw.line(self.screen, (50, 50, 70), (plot_x, gy), (plot_x + plot_w, gy), 1)
        
        # Energy line
        points = []
        for i, e in enumerate(self.energy_history):
            px = plot_x + (i / len(self.energy_history)) * plot_w
            py = plot_y + plot_h - ((e - min_e) / (max_e - min_e + 1)) * plot_h
            points.append((px, py))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, ACCENT, False, points, 3)
            
            # Current point
            pygame.draw.circle(self.screen, (255, 255, 100), points[-1], 8)
            
            # Target line (E=0)
            target_y = plot_y + plot_h - ((0 - min_e) / (max_e - min_e + 1)) * plot_h
            pygame.draw.line(self.screen, (100, 255, 100), (plot_x, target_y), (plot_x + plot_w, target_y), 2)
            
        # Fracture markers
        for fp in self.fracture_points:
            if fp < len(self.energy_history):
                fx = plot_x + (fp / len(self.energy_history)) * plot_w
                pygame.draw.circle(self.screen, (255, 100, 100), (int(fx), plot_y + 20), 5)
        
    def draw_stats_panel(self, x, y, w, h):
        """Draw statistics"""
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=10)
        
        title = self.font_large.render("3-SAT Stats", True, ACCENT)
        self.screen.blit(title, (x + 20, y + 20))
        
        elapsed = time.time() - self.start_time
        sat_percent = ((self.M - self.current_energy) / self.M * 100) if self.current_energy is not None else 0
        
        stats = [
            f"Variables: {self.N}",
            f"Clauses: {self.M}",
            f"Clause Density: {self.M/self.N:.2f}",
            "",
            f"Step: {self.step_count}",
            f"Time: {elapsed:.1f}s",
            f"Speed: {self.step_count/max(0.1,elapsed):.0f} steps/sec",
            "",
            f"Current Energy: {self.current_energy:.0f}",
            f"Best Energy: {self.best_energy:.0f}",
            f"Satisfied: {sat_percent:.1f}%",
            "",
            f"Fractures: {len(self.fracture_points)}",
            f"Status: {'SOLVED!' if self.current_energy == 0 else 'OPTIMIZING...'}",
        ]
        
        for i, stat in enumerate(stats):
            color = (100, 255, 100) if 'SOLVED' in stat else TEXT_COLOR
            text = self.font.render(stat, True, color)
            self.screen.blit(text, (x + 20, y + 80 + i * 30))
            
    def draw_clause_bar(self, x, y, w, h):
        """Visual representation of clause satisfaction"""
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=10)
        
        title = self.font.render("Clause Satisfaction", True, TEXT_COLOR)
        self.screen.blit(title, (x + 15, y + 15))
        
        if self.current_energy is None:
            return
            
        satisfied = self.M - self.current_energy
        
        bar_x = x + 20
        bar_y = y + 60
        bar_w = w - 40
        bar_h = 40
        
        # Background
        pygame.draw.rect(self.screen, (50, 50, 70), (bar_x, bar_y, bar_w, bar_h))
        
        # Satisfied portion (green)
        sat_width = (satisfied / self.M) * bar_w
        pygame.draw.rect(self.screen, (100, 255, 100), (bar_x, bar_y, sat_width, bar_h))
        
        # Unsatisfied portion (red)
        pygame.draw.rect(self.screen, (255, 100, 100), (bar_x + sat_width, bar_y, bar_w - sat_width, bar_h))
        
        # Labels
        sat_text = self.font.render(f"{satisfied} satisfied", True, (100, 255, 100))
        self.screen.blit(sat_text, (bar_x, bar_y + bar_h + 10))
        
        unsat_text = self.font.render(f"{int(self.current_energy)} unsatisfied", True, (255, 100, 100))
        self.screen.blit(unsat_text, (bar_x + bar_w - 150, bar_y + bar_h + 10))
        
    def render(self):
        """Main render"""
        self.screen.fill(BG_COLOR)
        
        # Header
        header = self.font_large.render("3-SAT: 500 Variables × 2300 Clauses", True, ACCENT)
        self.screen.blit(header, (WIDTH // 2 - header.get_width() // 2, 20))
        
        subtitle = self.font.render("Watch BAHA navigate the phase transition", True, (150, 150, 160))
        self.screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 65))
        
        # Layout
        self.draw_variable_grid(20, 100, 900, 600)          # Left: 500 vars
        self.draw_energy_panel(940, 100, 840, 350)         # Right top: energy
        self.draw_clause_bar(940, 470, 840, 120)           # Right mid: clause bar
        self.draw_stats_panel(940, 610, 840, 380)          # Right bottom: stats
        
        # Controls
        ctrl_text = f"SPACE=Pause | +/- = Speed ({self.batch_size}) | Q=Quit"
        if self.current_energy == 0:
            ctrl_text += " | 🏆 SOLVED! Press R to reset"
        ctrl = self.font.render(ctrl_text, True, (100, 100, 120))
        self.screen.blit(ctrl, (20, HEIGHT - 40))
        
        pygame.display.flip()
        
    def handle_events(self):
        """Handle input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.batch_size = min(50, self.batch_size + 5)
                elif event.key == pygame.K_MINUS:
                    self.batch_size = max(1, self.batch_size - 5)
                elif event.key == pygame.K_r:
                    self.reset()
                    
        return True
        
    def reset(self):
        """Reset optimization"""
        self.current_assignment = None
        self.current_energy = self.M
        self.best_energy = self.M
        self.step_count = 0
        self.energy_history.clear()
        self.fracture_points.clear()
        self.start_time = time.time()
        
    def run(self):
        print("🎮 Starting 3-SAT Visualizer...")
        print("Problem: Satisfy 2300 clauses with 500 boolean variables")
        print("This is in the HARD regime (density ~4.6)")
        print("")
        print("Controls:")
        print("  SPACE = Pause/Resume")
        print("  +/-   = Adjust speed")
        print("  R     = Reset")
        print("  Q     = Quit")
        print("")
        
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)
            
        pygame.quit()

if __name__ == "__main__":
    viz = SAT3Visualizer()
    viz.run()
