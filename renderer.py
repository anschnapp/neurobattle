"""Pygame rendering for the battlefield and training arenas."""

from __future__ import annotations

import math
import pygame
import numpy as np

from entities import Robot, Bullet, Base, Turret
from modules import BlockType, BLOCK_PIXEL_SIZE, Direction
import settings

# Colors for block types
BLOCK_COLORS = {
    BlockType.PLAIN:    (120, 120, 120),  # gray
    BlockType.ENGINE:   (60, 180, 60),    # green
    BlockType.WEAPON:   (220, 80, 40),    # red-orange
    BlockType.SENSOR:   (60, 160, 220),   # blue
    BlockType.SCANNER:  (200, 180, 40),   # yellow
    BlockType.GATHERER: (180, 60, 200),   # purple
}


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont("monospace", 14)
        self.big_font = pygame.font.SysFont("monospace", 28, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 11)

    def clear(self):
        self.screen.fill(settings.DARK_GRAY)

    # --- Battlefield rendering (offset into center strip) --------------------

    def draw_arena_border(self):
        oy = settings.BATTLEFIELD_Y
        pygame.draw.rect(self.screen, settings.MID_GRAY,
                         (0, oy, settings.ARENA_WIDTH, settings.ARENA_HEIGHT), 2)

    def draw_base(self, base: Base):
        oy = settings.BATTLEFIELD_Y
        cx, cy = int(base.center[0]), int(base.center[1]) + oy
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
            self.draw_turret(turret, oy)

    def draw_turret(self, turret: Turret, offset_y: int = 0):
        pos = turret.pos
        tx, ty = int(pos[0]), int(pos[1]) + offset_y
        color = settings.TEAM_COLORS[turret.team]

        pygame.draw.circle(self.screen, color, (tx, ty), 6)
        pygame.draw.circle(self.screen, settings.WHITE, (tx, ty), 6, 1)

        aim = turret.target_angle
        end_x = tx + int(math.cos(aim) * 12)
        end_y = ty + int(math.sin(aim) * 12)
        pygame.draw.line(self.screen, settings.WHITE, (tx, ty), (end_x, end_y), 2)

    def draw_robot(self, robot: Robot, offset_x: int = 0, offset_y: int = 0):
        if not robot.alive:
            return

        oy = offset_y if offset_y else settings.BATTLEFIELD_Y
        ox = offset_x

        team_color = settings.TEAM_COLORS[robot.team]
        cos_a = math.cos(robot.angle)
        sin_a = math.sin(robot.angle)
        half = BLOCK_PIXEL_SIZE / 2

        for block in robot.blocks:
            if not block.alive:
                continue

            local_x = block.grid_x * BLOCK_PIXEL_SIZE
            local_y = block.grid_y * BLOCK_PIXEL_SIZE
            wx = robot.pos[0] + cos_a * local_x - sin_a * local_y + ox
            wy = robot.pos[1] + sin_a * local_x + cos_a * local_y + oy

            corners_local = [
                (-half, -half), (half, -half),
                (half, half), (-half, half),
            ]
            corners_world = []
            for lx, ly in corners_local:
                rx = wx + cos_a * lx - sin_a * ly
                ry = wy + sin_a * lx + cos_a * ly
                corners_world.append((int(rx), int(ry)))

            base_color = BLOCK_COLORS[block.block_type]
            tr, tg, tb = team_color
            br, bg, bb = base_color
            color = (
                (br + tr) // 2,
                (bg + tg) // 2,
                (bb + tb) // 2,
            )

            hp_frac = block.hp / block.max_hp
            color = tuple(int(c * (0.3 + 0.7 * hp_frac)) for c in color)

            pygame.draw.polygon(self.screen, color, corners_world)
            pygame.draw.polygon(self.screen, settings.WHITE, corners_world, 1)

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

    def draw_bullet(self, bullet: Bullet, offset_x: int = 0, offset_y: int = 0):
        if not bullet.alive:
            return
        oy = offset_y if offset_y else settings.BATTLEFIELD_Y
        ox = offset_x
        x, y = int(bullet.pos[0]) + ox, int(bullet.pos[1]) + oy
        color = settings.TEAM_COLORS[bullet.team]
        bright = tuple(min(255, c + 100) for c in color)
        pygame.draw.circle(self.screen, bright, (x, y), bullet.radius)

    def draw_hud(self, bases: list[Base], robots: list[Robot], tick: int):
        oy = settings.BATTLEFIELD_Y
        for i, base in enumerate(bases):
            side = "P1" if base.team == 0 else "P2"
            color = settings.TEAM_COLORS[base.team]
            wall_str = f"Wall: {int(base.wall_hp)}" if base.wall_alive else "BREACHED"
            cmd_str = "CMD: OK" if base.commander_alive else "CMD: DEAD"
            n_bots = sum(1 for r in robots if r.team == base.team and r.alive)
            text = f"{side}  {wall_str}  {cmd_str}  Bots:{n_bots}"
            surf = self.font.render(text, True, color)
            x = 10 if i == 0 else settings.SCREEN_WIDTH - surf.get_width() - 10
            self.screen.blit(surf, (x, oy + 8))

        tick_surf = self.font.render(f"T:{tick}", True, settings.LIGHT_GRAY)
        self.screen.blit(tick_surf, (settings.SCREEN_WIDTH // 2 - tick_surf.get_width() // 2, oy + 8))

    def draw_game_over(self, winner: int):
        oy = settings.BATTLEFIELD_Y
        text = f"Player {winner + 1} Wins!"
        color = settings.TEAM_COLORS[winner]
        surf = self.big_font.render(text, True, color)
        x = settings.SCREEN_WIDTH // 2 - surf.get_width() // 2
        y = oy + settings.ARENA_HEIGHT // 2 - surf.get_height() // 2
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

    # --- Training arena rendering -------------------------------------------

    def draw_training_strip(self, player_id: int, arenas, override_y: int | None = None):
        """Draw 3 training arenas in a horizontal strip.

        player_id 0 = top strip, player_id 1 = bottom strip.
        arenas: list of 3 TrainingArena or None.
        override_y: if set, draw at this y instead of the default position.
        """
        if override_y is not None:
            strip_y = override_y
        else:
            strip_y = 0 if player_id == 0 else (settings.SCREEN_HEIGHT - settings.TRAINING_STRIP_HEIGHT)
        strip_h = settings.TRAINING_STRIP_HEIGHT

        # Divider line between training strip and battlefield
        div_y = strip_y + strip_h if player_id == 0 else strip_y
        team_color = settings.TEAM_COLORS[player_id]
        dim_color = tuple(c // 3 for c in team_color)
        pygame.draw.line(self.screen, dim_color, (0, div_y), (settings.SCREEN_WIDTH, div_y), 2)

        # Player label
        label = self.font.render(f"P{player_id + 1} TRAINING", True, team_color)
        self.screen.blit(label, (6, strip_y + 3))

        # Layout: 3 arenas across
        arena_gap = 8
        total_gap = arena_gap * 4  # gaps on edges + between
        arena_draw_w = (settings.SCREEN_WIDTH - total_gap) // 3
        arena_draw_h = strip_h - 24  # leave room for label at top

        for slot in range(3):
            arena = arenas[slot] if arenas else None
            ax = arena_gap + slot * (arena_draw_w + arena_gap)
            ay = strip_y + 20

            # Arena background
            pygame.draw.rect(self.screen, (20, 20, 28), (ax, ay, arena_draw_w, arena_draw_h))
            pygame.draw.rect(self.screen, (50, 50, 60), (ax, ay, arena_draw_w, arena_draw_h), 1)

            if arena is None:
                txt = self.small_font.render("(no design)", True, settings.MID_GRAY)
                self.screen.blit(txt, (ax + arena_draw_w // 2 - txt.get_width() // 2,
                                       ay + arena_draw_h // 2 - 6))
                continue

            # Scale factor from arena sim coords to draw coords
            scale_x = arena_draw_w / arena.width
            scale_y = arena_draw_h / arena.height
            scale = min(scale_x, scale_y)
            # Center the arena within the draw rect
            scaled_w = arena.width * scale
            scaled_h = arena.height * scale
            ox = ax + (arena_draw_w - scaled_w) / 2
            oy = ay + (arena_draw_h - scaled_h) / 2

            # Draw robots (students + sparring) — use raw drawing for speed
            self._draw_arena_robots(arena, ox, oy, scale)

            # Draw bullets
            for bullet in arena.bullets:
                if not bullet.alive:
                    continue
                bx = int(bullet.pos[0] * scale + ox)
                by = int(bullet.pos[1] * scale + oy)
                color = settings.TEAM_COLORS[bullet.team]
                bright = tuple(min(255, c + 80) for c in color)
                pygame.draw.circle(self.screen, bright, (bx, by), max(1, int(2 * scale)))

            # Stats overlay
            stats = arena.get_stats()
            gen_txt = f"G:{stats['generation']}"
            fit_txt = f"Best:{stats['best_fitness']:.0f}"
            alive_txt = f"{stats['alive_students']}/{stats['total_students']}"
            tick_txt = f"{stats['gen_tick']}/{settings.TRAINING_TICKS_PER_GENERATION}"

            self.screen.blit(self.small_font.render(gen_txt, True, settings.WHITE),
                             (ax + 3, ay + 2))
            self.screen.blit(self.small_font.render(fit_txt, True, settings.GREEN),
                             (ax + 3, ay + 14))
            self.screen.blit(self.small_font.render(alive_txt, True, settings.LIGHT_GRAY),
                             (ax + arena_draw_w - 40, ay + 2))
            self.screen.blit(self.small_font.render(tick_txt, True, settings.MID_GRAY),
                             (ax + arena_draw_w - 55, ay + 14))

            # Slot label
            slot_lbl = self.small_font.render(f"Bot {slot + 1}", True, team_color)
            self.screen.blit(slot_lbl, (ax + arena_draw_w // 2 - slot_lbl.get_width() // 2,
                                         ay + arena_draw_h - 14))

    def _draw_arena_robots(self, arena, ox: float, oy: float, scale: float):
        """Draw robots inside a training arena viewport (scaled)."""
        for robot in arena.all_robots:
            if not robot.alive:
                continue

            is_student = robot.team == 0
            team_color = settings.TEAM_COLORS[arena.player_id] if is_student else (150, 80, 80)
            cos_a = math.cos(robot.angle)
            sin_a = math.sin(robot.angle)
            half = BLOCK_PIXEL_SIZE / 2 * scale

            for block in robot.blocks:
                if not block.alive:
                    continue

                local_x = block.grid_x * BLOCK_PIXEL_SIZE * scale
                local_y = block.grid_y * BLOCK_PIXEL_SIZE * scale
                wx = robot.pos[0] * scale + ox + cos_a * local_x - sin_a * local_y
                wy = robot.pos[1] * scale + oy + sin_a * local_x + cos_a * local_y

                corners_local = [
                    (-half, -half), (half, -half),
                    (half, half), (-half, half),
                ]
                corners_world = []
                for lx, ly in corners_local:
                    rx = wx + cos_a * lx - sin_a * ly
                    ry = wy + sin_a * lx + cos_a * ly
                    corners_world.append((int(rx), int(ry)))

                base_color = BLOCK_COLORS[block.block_type]
                br, bg, bb = base_color
                tr, tg, tb = team_color
                color = ((br + tr) // 2, (bg + tg) // 2, (bb + tb) // 2)

                hp_frac = block.hp / block.max_hp
                color = tuple(int(c * (0.3 + 0.7 * hp_frac)) for c in color)

                pygame.draw.polygon(self.screen, color, corners_world)
