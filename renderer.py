"""Pygame rendering for the battlefield — block-based robots."""

from __future__ import annotations

import math
import pygame
import numpy as np

from entities import Robot, Bullet, Base, Turret
from modules import BlockType, BLOCK_PIXEL_SIZE, Direction
import settings

# Colors for block types
BLOCK_COLORS = {
    BlockType.PLAIN:   (120, 120, 120),  # gray
    BlockType.ENGINE:  (60, 180, 60),    # green
    BlockType.WEAPON:  (220, 80, 40),    # red-orange
    BlockType.SENSOR:  (60, 160, 220),   # blue
    BlockType.SCANNER: (200, 180, 40),   # yellow
}


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont("monospace", 14)
        self.big_font = pygame.font.SysFont("monospace", 28, bold=True)

    def clear(self):
        self.screen.fill(settings.DARK_GRAY)

    def draw_arena_border(self):
        pygame.draw.rect(self.screen, settings.MID_GRAY,
                         (0, 0, settings.ARENA_WIDTH, settings.ARENA_HEIGHT), 2)

    def draw_base(self, base: Base):
        cx, cy = int(base.center[0]), int(base.center[1])
        color = settings.TEAM_COLORS[base.team]

        if base.wall_alive:
            wall_alpha = base.wall_hp / settings.BASE_WALL_HP
            r, g, b = color
            wall_color = (
                int(r * 0.4 + 60 * wall_alpha),
                int(g * 0.4 + 60 * wall_alpha),
                int(b * 0.4 + 60 * wall_alpha),
            )
            pygame.draw.circle(self.screen, wall_color, (cx, cy),
                               int(base.radius) + settings.BASE_WALL_THICKNESS,
                               settings.BASE_WALL_THICKNESS * 2)
            hp_text = self.font.render(f"{int(base.wall_hp)}", True, settings.LIGHT_GRAY)
            self.screen.blit(hp_text, (cx - hp_text.get_width() // 2,
                                       cy + int(base.radius) + 12))
        else:
            pygame.draw.circle(self.screen, settings.MID_GRAY, (cx, cy),
                               int(base.radius), 1)

        if base.commander_alive:
            pygame.draw.circle(self.screen, settings.RED, (cx, cy),
                               settings.COMMANDER_RADIUS)
            pygame.draw.circle(self.screen, settings.WHITE, (cx, cy),
                               settings.COMMANDER_RADIUS, 1)

        for turret in base.turrets:
            self.draw_turret(turret)

    def draw_turret(self, turret: Turret):
        pos = turret.pos
        tx, ty = int(pos[0]), int(pos[1])
        color = settings.TEAM_COLORS[turret.team]

        pygame.draw.circle(self.screen, color, (tx, ty), 6)
        pygame.draw.circle(self.screen, settings.WHITE, (tx, ty), 6, 1)

        aim = turret.target_angle
        end_x = tx + int(math.cos(aim) * 12)
        end_y = ty + int(math.sin(aim) * 12)
        pygame.draw.line(self.screen, settings.WHITE, (tx, ty), (end_x, end_y), 2)

    def draw_robot(self, robot: Robot):
        if not robot.alive:
            return

        team_color = settings.TEAM_COLORS[robot.team]
        cos_a = math.cos(robot.angle)
        sin_a = math.sin(robot.angle)
        half = BLOCK_PIXEL_SIZE / 2

        for block in robot.blocks:
            if not block.alive:
                continue

            # Block center in world space
            local_x = block.grid_x * BLOCK_PIXEL_SIZE
            local_y = block.grid_y * BLOCK_PIXEL_SIZE
            wx = robot.pos[0] + cos_a * local_x - sin_a * local_y
            wy = robot.pos[1] + sin_a * local_x + cos_a * local_y

            # Rotated rectangle corners
            corners_local = [
                (-half, -half), (half, -half),
                (half, half), (-half, half),
            ]
            corners_world = []
            for lx, ly in corners_local:
                rx = wx + cos_a * lx - sin_a * ly
                ry = wy + sin_a * lx + cos_a * ly
                corners_world.append((rx, ry))

            # Block color: tinted by team
            base_color = BLOCK_COLORS[block.block_type]
            tr, tg, tb = team_color
            br, bg, bb = base_color
            color = (
                (br + tr) // 2,
                (bg + tg) // 2,
                (bb + tb) // 2,
            )

            # Dim if damaged
            hp_frac = block.hp / block.max_hp
            color = tuple(int(c * (0.3 + 0.7 * hp_frac)) for c in color)

            pygame.draw.polygon(self.screen, color, corners_world)
            pygame.draw.polygon(self.screen, settings.WHITE, corners_world, 1)

            # Draw direction indicator for directional blocks
            if block.block_type in (BlockType.ENGINE, BlockType.WEAPON, BlockType.SENSOR):
                dir_angle = robot.angle + block.direction.angle
                ix = wx + math.cos(dir_angle) * (half * 0.8)
                iy = wy + math.sin(dir_angle) * (half * 0.8)
                indicator_color = settings.WHITE
                if block.block_type == BlockType.WEAPON:
                    indicator_color = settings.YELLOW
                elif block.block_type == BlockType.SENSOR:
                    indicator_color = settings.CYAN
                pygame.draw.circle(self.screen, indicator_color, (int(ix), int(iy)), 2)

    def draw_bullet(self, bullet: Bullet):
        if not bullet.alive:
            return
        x, y = int(bullet.pos[0]), int(bullet.pos[1])
        color = settings.TEAM_COLORS[bullet.team]
        bright = tuple(min(255, c + 100) for c in color)
        pygame.draw.circle(self.screen, bright, (x, y), bullet.radius)

    def draw_hud(self, bases: list[Base], robots: list[Robot], tick: int):
        for i, base in enumerate(bases):
            side = "P1" if base.team == 0 else "P2"
            color = settings.TEAM_COLORS[base.team]
            wall_str = f"Wall: {int(base.wall_hp)}" if base.wall_alive else "BREACHED"
            cmd_str = "CMD: OK" if base.commander_alive else "CMD: DEAD"
            n_bots = sum(1 for r in robots if r.team == base.team and r.alive)
            text = f"{side}  {wall_str}  {cmd_str}  Bots:{n_bots}"
            surf = self.font.render(text, True, color)
            x = 10 if i == 0 else settings.SCREEN_WIDTH - surf.get_width() - 10
            self.screen.blit(surf, (x, 8))

        tick_surf = self.font.render(f"T:{tick}", True, settings.LIGHT_GRAY)
        self.screen.blit(tick_surf, (settings.SCREEN_WIDTH // 2 - tick_surf.get_width() // 2, 8))

    def draw_game_over(self, winner: int):
        text = f"Player {winner + 1} Wins!"
        color = settings.TEAM_COLORS[winner]
        surf = self.big_font.render(text, True, color)
        x = settings.SCREEN_WIDTH // 2 - surf.get_width() // 2
        y = settings.SCREEN_HEIGHT // 2 - surf.get_height() // 2
        padding = 20
        pygame.draw.rect(self.screen, settings.BLACK,
                         (x - padding, y - padding,
                          surf.get_width() + padding * 2,
                          surf.get_height() + padding * 2))
        pygame.draw.rect(self.screen, color,
                         (x - padding, y - padding,
                          surf.get_width() + padding * 2,
                          surf.get_height() + padding * 2), 2)
        self.screen.blit(surf, (x, y))
