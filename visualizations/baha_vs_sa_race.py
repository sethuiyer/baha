#!/usr/bin/env python3
"""
BAHA vs Simulated Annealing Race
Split-screen visualization showing how BAHA jumps vs SA gets stuck

Red side = Traditional SA (gets trapped)
Blue side = BAHA (detects fractures and jumps)
"""

import pygame
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
import math
import networkx as nx
from collections import deque

pygame.init()

WIDTH, HEIGHT = 1600, 900
FPS = 60

colors = {
    'bg': (10, 10, 15),
    'sa_panel': (40, 20, 20),
    'baha_panel': (20, 30, 50),
    'sa_color': (255, 80, 80),
    'baha_color': (80, 160, 255),
    'text': (220, 220, 220),
    'grid': (40, 40, 50),
    'stuck': (255, 50, 50),
    'jump': (100, 255, 100),
}

class RaceVisualization:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("BAHA vs SA: The Race 🏁")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Problem: Graph coloring
        self.G = nx.erdos_renyi_graph(25, 0.25, seed=123)
        self.pos = nx.spring_layout(self.G, seed=42, k=1.5)
        
        # Two optimizers
        self.reset()
        
        # Visualization data
        self.sa_history = deque(maxlen=400)
        self.baha_history = deque(maxlen=400)
        self.baha_jumps = []
        
        self.step = 0
        self.finished = False
        
    def reset(self):
        # SA optimizer (basic, no fractures)
        self.sa_state = {n: random.randint(0, 3) for n in self.G.nodes()}
        self.sa_energy = self.calculate_energy(self.sa_state)
        self.sa_best = self.sa_energy
        self.sa_stuck_count = 0
        
        # BAHA optimizer (with jump capability)
        self.baha_state = {n: random.randint(0, 3) for n in self.G.nodes()}
        self.baha_energy = self.calculate_energy(self.baha_state)
        self.baha_best = self.baha_energy
        self.baha_since_last_improvement = 0
        self.baha_jump_cooldown = 0
        
    def calculate_energy(self, state):
        return sum(1 for u, v in self.G.edges() if state[u] == state[v])
    
    def sa_step(self):
        """Traditional SA step - can get stuck"""
        beta = 0.05
        
        # Pick random neighbor
        node = random.choice(list(self.G.nodes()))
        old_color = self.sa_state[node]
        new_color = random.randint(0, 3)
        
        if new_color != old_color:
            old_energy = self.sa_energy
            self.sa_state[node] = new_color
            new_energy = self.calculate_energy(self.sa_state)
            
            delta = new_energy - old_energy
            if delta < 0 or random.random() < math.exp(-beta * delta):
                self.sa_energy = new_energy
                self.sa_best = min(self.sa_best, self.sa_energy)
                self.sa_stuck_count = 0
            else:
                self.sa_state[node] = old_color
                self.sa_stuck_count += 1
    
    def baha_step(self):
        """BAHA step - detects 'fractures' and jumps"""
        beta = 0.05 + self.step * 0.0002
        
        # Detect if stuck (simulated fracture detection)
        self.baha_since_last_improvement += 1
        
        # If stuck for too long, "jump" to different basin
        if self.baha_since_last_improvement > 50 and self.baha_jump_cooldown == 0:
            # Perform branch jump - restart with some preserved structure
            self.baha_jump_cooldown = 20
            self.baha_since_last_improvement = 0
            self.baha_jumps.append(self.step)
            
            # Jump: Keep 60% of current state, randomize rest
            nodes = list(self.G.nodes())
            random.shuffle(nodes)
            for i, node in enumerate(nodes):
                if i > len(nodes) * 0.6:
                    self.baha_state[node] = random.randint(0, 3)
        
        if self.baha_jump_cooldown > 0:
            self.baha_jump_cooldown -= 1
            return
        
        # Normal SA step
        node = random.choice(list(self.G.nodes()))
        old_color = self.baha_state[node]
        new_color = random.randint(0, 3)
        
        if new_color != old_color:
            old_energy = self.baha_energy
            self.baha_state[node] = new_color
            new_energy = self.calculate_energy(self.baha_state)
            
            delta = new_energy - old_energy
            if delta < 0 or random.random() < math.exp(-beta * delta):
                self.baha_energy = new_energy
                if self.baha_energy < self.baha_best:
                    self.baha_best = self.baha_energy
                    self.baha_since_last_improvement = 0
            else:
                self.baha_state[node] = old_color
    
    def update(self):
        if self.finished and self.baha_best == 0 and self.sa_best == 0:
            return
            
        self.sa_step()
        self.baha_step()
        
        self.sa_history.append(self.sa_energy)
        self.baha_history.append(self.baha_energy)
        
        self.step += 1
        
        if self.baha_best == 0:
            self.finished = True
    
    def draw_graph(self, x, y, w, h, state, title, color_key, is_winner=False):
        """Draw a single graph"""
        pygame.draw.rect(self.screen, colors['sa_panel'] if 'SA' in title else colors['baha_panel'], 
                        (x, y, w, h), border_radius=15)
        
        # Title
        title_surf = self.font_large.render(title, True, 
                                           colors['sa_color'] if 'SA' in title else colors['baha_color'])
        self.screen.blit(title_surf, (x + 20, y + 15))
        
        if is_winner:
            winner = self.font.render("🏆 WINNER!", True, colors['jump'])
            self.screen.blit(winner, (x + w - 150, y + 20))
        
        # Draw graph
        cx, cy = x + w // 2, y + h // 2 + 30
        scale = min(w, h) * 0.32
        
        color_map = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 200, 100),
        ]
        
        # Edges
        for u, v in self.G.edges():
            u_pos = (cx + self.pos[u][0] * scale, cy + self.pos[u][1] * scale)
            v_pos = (cx + self.pos[v][0] * scale, cy + self.pos[v][1] * scale)
            
            is_conflict = state[u] == state[v]
            edge_color = (255, 50, 50) if is_conflict else (60, 60, 80)
            pygame.draw.line(self.screen, edge_color, u_pos, v_pos, 3 if is_conflict else 1)
        
        # Nodes
        for node in self.G.nodes():
            pos = (cx + self.pos[node][0] * scale, cy + self.pos[node][1] * scale)
            pygame.draw.circle(self.screen, color_map[state[node] % 4], pos, 18)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 14)
            pygame.draw.circle(self.screen, color_map[state[node] % 4], pos, 12)
    
    def draw_trajectory(self, x, y, w, h):
        """Draw energy trajectories"""
        pygame.draw.rect(self.screen, (20, 20, 30), (x, y, w, h), border_radius=15)
        
        title = self.font.render("Energy vs Time", True, colors['text'])
        self.screen.blit(title, (x + 20, y + 15))
        
        # Plot area
        px, py = x + 80, y + 60
        pw, ph = w - 100, h - 100
        
        # Grid
        for i in range(6):
            gy = py + ph * i // 5
            pygame.draw.line(self.screen, colors['grid'], (px, gy), (px + pw, gy), 1)
        
        if len(self.sa_history) > 1:
            max_e = max(max(self.sa_history), max(self.baha_history)) if self.baha_history else 1
            min_e = 0
            
            # SA line (red)
            sa_points = []
            for i, e in enumerate(self.sa_history):
                px_i = px + (i / len(self.sa_history)) * pw
                py_i = py + ph - ((e - min_e) / (max_e - min_e + 1)) * ph
                sa_points.append((px_i, py_i))
            
            if len(sa_points) > 1:
                pygame.draw.lines(self.screen, colors['sa_color'], False, sa_points, 4)
            
            # BAHA line (blue)
            baha_points = []
            for i, e in enumerate(self.baha_history):
                px_i = px + (i / len(self.baha_history)) * pw
                py_i = py + ph - ((e - min_e) / (max_e - min_e + 1)) * ph
                baha_points.append((px_i, py_i))
            
            if len(baha_points) > 1:
                pygame.draw.lines(self.screen, colors['baha_color'], False, baha_points, 4)
            
            # Mark jumps
            for jump_step in self.baha_jumps:
                if jump_step < len(self.baha_history):
                    jx = px + (jump_step / len(self.baha_history)) * pw
                    pygame.draw.circle(self.screen, colors['jump'], (int(jx), py + ph - 10), 8)
        
        # Legend
        pygame.draw.line(self.screen, colors['sa_color'], (px + pw - 200, py + 20), (px + pw - 170, py + 20), 4)
        sa_label = self.font_small.render("SA (gets stuck)", True, colors['sa_color'])
        self.screen.blit(sa_label, (px + pw - 160, py + 12))
        
        pygame.draw.line(self.screen, colors['baha_color'], (px + pw - 200, py + 50), (px + pw - 170, py + 50), 4)
        baha_label = self.font_small.render("BAHA (jumps)", True, colors['baha_color'])
        self.screen.blit(baha_label, (px + pw - 160, py + 42))
        
        # Jump markers
        pygame.draw.circle(self.screen, colors['jump'], (px + pw - 185, py + 80), 6)
        jump_label = self.font_small.render("Fracture Jump", True, colors['jump'])
        self.screen.blit(jump_label, (px + pw - 160, py + 72))
    
    def render(self):
        self.screen.fill(colors['bg'])
        
        # Header
        header = self.font_large.render("🏁 BAHA vs Simulated Annealing: The Race", True, colors['text'])
        self.screen.blit(header, (WIDTH // 2 - header.get_width() // 2, 20))
        
        subtitle = self.font_small.render("Watch how BAHA detects fractures and jumps while SA gets trapped", True, (150, 150, 160))
        self.screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 65))
        
        # Left panel: SA
        is_sa_winner = self.sa_best == 0 and self.baha_best > 0
        self.draw_graph(30, 100, 500, 400, self.sa_state, "Simulated Annealing", 'sa', is_sa_winner)
        
        # Right panel: BAHA
        is_baha_winner = self.baha_best == 0
        self.draw_graph(1070, 100, 500, 400, self.baha_state, "BAHA (with jumps)", 'baha', is_baha_winner)
        
        # Middle: Energy trajectories
        self.draw_trajectory(560, 100, 480, 400)
        
        # Stats panel
        stats_y = 520
        pygame.draw.rect(self.screen, (20, 20, 30), (30, stats_y, 1540, 360), border_radius=15)
        
        # SA stats
        sa_stats = [
            f"Current Energy: {self.sa_energy}",
            f"Best Energy: {self.sa_best}",
            f"Steps Stuck: {self.sa_stuck_count}",
            f"Status: {'STUCK!' if self.sa_stuck_count > 100 else 'Searching...'}",
        ]
        
        sa_title = self.font.render("SA Stats", True, colors['sa_color'])
        self.screen.blit(sa_title, (60, stats_y + 20))
        
        for i, line in enumerate(sa_stats):
            text = self.font_small.render(line, True, colors['text'])
            self.screen.blit(text, (60, stats_y + 60 + i * 30))
        
        # BAHA stats
        baha_stats = [
            f"Current Energy: {self.baha_energy}",
            f"Best Energy: {self.baha_best}",
            f"Fractures/Jumps: {len(self.baha_jumps)}",
            f"Since Jump: {self.baha_since_last_improvement} steps",
        ]
        
        baha_title = self.font.render("BAHA Stats", True, colors['baha_color'])
        self.screen.blit(baha_title, (560, stats_y + 20))
        
        for i, line in enumerate(baha_stats):
            text = self.font_small.render(line, True, colors['text'])
            self.screen.blit(text, (560, stats_y + 60 + i * 30))
        
        # Global step counter
        step_text = self.font.render(f"Step: {self.step}", True, colors['text'])
        self.screen.blit(step_text, (WIDTH // 2 - 50, stats_y + 20))
        
        # Controls
        ctrl_text = self.font_small.render("Controls: R=Reset | SPACE=Pause | Q=Quit", True, (100, 100, 110))
        self.screen.blit(ctrl_text, (60, HEIGHT - 50))
        
        pygame.display.flip()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                elif event.key == pygame.K_r:
                    self.reset()
                    self.sa_history.clear()
                    self.baha_history.clear()
                    self.baha_jumps.clear()
                    self.step = 0
                    self.finished = False
                elif event.key == pygame.K_SPACE:
                    pass  # Could add pause
        return True
    
    def run(self):
        print("Starting BAHA vs SA Race...")
        print("Red = Traditional SA (gets stuck)")
        print("Blue = BAHA (detects fractures and jumps)")
        
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    race = RaceVisualization()
    race.run()
