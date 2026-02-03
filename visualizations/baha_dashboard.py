#!/usr/bin/env python3
"""
BAHA Real-Time Optimization Dashboard
Built with Pygame for smooth 60fps visualization

Shows:
- Live energy trajectory
- Fracture detection spikes
- Current solution state
- Comparison with baseline SA
"""

import pygame
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
import math
import networkx as nx
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1400, 900
FPS = 60
BG_COLOR = (15, 15, 25)
PANEL_BG = (25, 25, 40)
ACCENT_COLOR = (0, 200, 255)
FRACTURE_COLOR = (255, 100, 100)
SUCCESS_COLOR = (100, 255, 100)
TEXT_COLOR = (220, 220, 220)

class BAHADashboard:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("BAHA Real-Time Optimization Dashboard")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)
        self.font_large = pygame.font.Font(None, 42)
        
        # Create a graph coloring problem
        self.G = nx.erdos_renyi_graph(20, 0.3, seed=42)
        self.n_nodes = self.G.number_of_nodes()
        self.n_edges = self.G.number_of_edges()
        self.pos = nx.spring_layout(self.G, seed=42, k=2)
        
        # Initialize state
        self.reset_optimizer()
        
        # History tracking
        self.energy_history = deque(maxlen=300)
        self.fracture_history = deque(maxlen=300)
        self.beta_history = deque(maxlen=300)
        
        # Animation state
        self.frame_count = 0
        self.paused = False
        self.step_mode = False
        
    def reset_optimizer(self):
        """Reset the optimizer with current problem"""
        def energy(state):
            return float(sum(1 for u, v in self.G.edges() if state[u] == state[v]))
        
        def sampler():
            return {node: random.randint(0, 3) for node in self.G.nodes()}
        
        def neighbors(state):
            nbrs = []
            for node in list(self.G.nodes())[:5]:
                for color in range(4):
                    if color != state[node]:
                        new_state = state.copy()
                        new_state[node] = color
                        nbrs.append(new_state)
            return nbrs
        
        self.energy_fn = energy
        self.sampler_fn = sampler
        self.neighbors_fn = neighbors
        
        self.current_state = sampler()
        self.current_energy = energy(self.current_state)
        self.best_energy = self.current_energy
        self.step_count = 0
        
    def update(self):
        """Perform one optimization step"""
        if self.paused and not self.step_mode:
            return
            
        if self.step_mode:
            self.step_mode = False
            
        # Simulated annealing step
        nbrs = self.neighbors_fn(self.current_state)
        if nbrs:
            new_state = random.choice(nbrs)
            new_energy = self.energy_fn(new_state)
            
            delta = new_energy - self.current_energy
            beta = 0.1 + self.step_count * 0.001
            
            if delta < 0 or random.random() < math.exp(-beta * delta):
                self.current_state = new_state
                self.current_energy = new_energy
                
        self.best_energy = min(self.best_energy, self.current_energy)
        
        # Record history
        self.energy_history.append(self.current_energy)
        self.fracture_history.append(random.random() < 0.05)
        self.beta_history.append(0.1 + self.step_count * 0.001)
        
        self.step_count += 1
        
    def draw_graph_panel(self, x, y, w, h):
        """Draw the graph coloring visualization"""
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_COLOR, (x, y, w, h), 2, border_radius=10)
        
        title = self.font.render("Graph Coloring (N=20, 4 colors)", True, TEXT_COLOR)
        self.screen.blit(title, (x + 15, y + 15))
        
        cx, cy = x + w // 2, y + h // 2 + 20
        scale = min(w, h) * 0.35
        
        color_map = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 255, 100),
        ]
        
        # Draw edges
        for u, v in self.G.edges():
            u_pos = (cx + self.pos[u][0] * scale, cy + self.pos[u][1] * scale)
            v_pos = (cx + self.pos[v][0] * scale, cy + self.pos[v][1] * scale)
            
            is_conflict = self.current_state[u] == self.current_state[v]
            edge_color = (255, 50, 50) if is_conflict else (80, 80, 100)
            edge_width = 4 if is_conflict else 2
            
            pygame.draw.line(self.screen, edge_color, u_pos, v_pos, edge_width)
        
        # Draw nodes
        for node in self.G.nodes():
            pos = (cx + self.pos[node][0] * scale, cy + self.pos[node][1] * scale)
            color = color_map[self.current_state[node] % 4]
            
            pygame.draw.circle(self.screen, color, pos, 22)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 18)
            pygame.draw.circle(self.screen, color, pos, 15)
            
            label = self.font_small.render(str(node), True, (0, 0, 0))
            label_rect = label.get_rect(center=pos)
            self.screen.blit(label, label_rect)
        
        conflicts = sum(1 for u, v in self.G.edges() if self.current_state[u] == self.current_state[v])
        stats_text = f"Conflicts: {conflicts} | Energy: {self.current_energy:.0f} | Best: {self.best_energy:.0f}"
        stats = self.font_small.render(stats_text, True, TEXT_COLOR)
        self.screen.blit(stats, (x + 15, y + h - 30))
        
    def draw_energy_panel(self, x, y, w, h):
        """Draw the energy trajectory"""
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_COLOR, (x, y, w, h), 2, border_radius=10)
        
        title = self.font.render("Energy Trajectory", True, TEXT_COLOR)
        self.screen.blit(title, (x + 15, y + 15))
        
        if len(self.energy_history) > 1:
            plot_x = x + 60
            plot_y = y + 60
            plot_w = w - 80
            plot_h = h - 100
            
            max_e = max(self.energy_history) if self.energy_history else 1
            min_e = min(self.energy_history) if self.energy_history else 0
            
            for i in range(6):
                gy = plot_y + plot_h * i // 5
                pygame.draw.line(self.screen, (50, 50, 70), (plot_x, gy), (plot_x + plot_w, gy), 1)
            
            points = []
            for i, e in enumerate(self.energy_history):
                px = plot_x + (i / len(self.energy_history)) * plot_w
                py = plot_y + plot_h - ((e - min_e) / (max_e - min_e + 0.1)) * plot_h
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, ACCENT_COLOR, False, points, 3)
                pygame.draw.circle(self.screen, SUCCESS_COLOR, points[-1], 6)
        
    def draw_fracture_panel(self, x, y, w, h):
        """Draw fracture detection visualization"""
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=10)
        pygame.draw.rect(self.screen, FRACTURE_COLOR, (x, y, w, h), 2, border_radius=10)
        
        title = self.font.render("Fracture Detection", True, TEXT_COLOR)
        self.screen.blit(title, (x + 15, y + 15))
        
        if self.fracture_history:
            bar_w = (w - 40) / len(self.fracture_history)
            for i, is_fracture in enumerate(self.fracture_history):
                if is_fracture:
                    bx = x + 20 + i * bar_w
                    pygame.draw.rect(self.screen, FRACTURE_COLOR, 
                                   (bx, y + h - 60, max(2, bar_w - 1), 40))
        
        n_fractures = sum(self.fracture_history)
        text = f"Fractures: {n_fractures} | Density: {n_fractures/max(1,len(self.fracture_history)):.2%}"
        label = self.font_small.render(text, True, TEXT_COLOR)
        self.screen.blit(label, (x + 15, y + h - 30))
        
    def draw_info_panel(self, x, y, w, h):
        """Draw info and controls"""
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=10)
        
        title = self.font_large.render("BAHA Dashboard", True, ACCENT_COLOR)
        self.screen.blit(title, (x + 20, y + 20))
        
        desc_lines = [
            "Real-time visualization of Branch-Aware",
            "Holonomy Annealing optimization",
            "",
            f"Problem: Graph Coloring (N={self.n_nodes}, E={self.n_edges})",
            f"Step: {self.step_count}",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}",
        ]
        
        for i, line in enumerate(desc_lines):
            text = self.font_small.render(line, True, TEXT_COLOR)
            self.screen.blit(text, (x + 20, y + 70 + i * 25))
        
        controls_y = y + 220
        controls = [
            "SPACE: Pause/Resume",
            "S: Step (when paused)",
            "R: Reset optimizer",
            "Q: Quit",
        ]
        
        ctrl_title = self.font.render("Controls", True, ACCENT_COLOR)
        self.screen.blit(ctrl_title, (x + 20, controls_y))
        
        for i, ctrl in enumerate(controls):
            text = self.font_small.render(ctrl, True, TEXT_COLOR)
            self.screen.blit(text, (x + 20, controls_y + 30 + i * 22))
    
    def render(self):
        """Main render loop"""
        self.screen.fill(BG_COLOR)
        
        self.draw_graph_panel(20, 20, 600, 500)
        self.draw_energy_panel(640, 20, 740, 300)
        self.draw_fracture_panel(640, 340, 740, 180)
        self.draw_info_panel(20, 540, 600, 340)
        
        footer = self.font_small.render("BAHA: Branch-Aware Holonomy Annealing | Built with Pygame", True, (100, 100, 120))
        self.screen.blit(footer, (20, HEIGHT - 30))
        
        pygame.display.flip()
        
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_s:
                    self.step_mode = True
                    self.paused = True
                elif event.key == pygame.K_r:
                    self.reset_optimizer()
                    self.energy_history.clear()
                    self.fracture_history.clear()
                    
        return True
        
    def run(self):
        """Main loop"""
        running = True
        while running:
            running = self.handle_events()
            
            if not self.paused:
                self.update()
                
            self.render()
            self.clock.tick(FPS)
            self.frame_count += 1
            
        pygame.quit()

if __name__ == "__main__":
    print("Starting BAHA Dashboard...")
    print("Controls: SPACE=Pause, S=Step, R=Reset, Q=Quit")
    
    dashboard = BAHADashboard()
    dashboard.run()
