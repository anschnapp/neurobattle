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

    # --- Training zone rendering ----------------------------------------------

    def draw_training_zone(self, player_id: int, zone, ui, setup_remaining: int = 0,
                           override_y: int | None = None):
        """Draw a single training zone: config panel on left, arena viewport on right.

        zone: TrainingZoneProxy
        ui: TrainingZoneUI
        setup_remaining: ticks left in 15s setup period (0 = training active)
        """
        from training import TrainingZoneUI, FITNESS_PARAMS

        if override_y is not None:
            strip_y = override_y
        else:
            strip_y = 0 if player_id == 0 else (settings.SCREEN_HEIGHT - settings.TRAINING_STRIP_HEIGHT)
        strip_h = settings.TRAINING_STRIP_HEIGHT
        team_color = settings.TEAM_COLORS[player_id]

        # Divider line
        div_y = strip_y + strip_h if player_id == 0 else strip_y
        dim_color = tuple(c // 3 for c in team_color)
        pygame.draw.line(self.screen, dim_color, (0, div_y), (settings.SCREEN_WIDTH, div_y), 2)

        # --- Config panel (left side) ---
        panel_w = settings.TRAINING_CONFIG_PANEL_WIDTH
        panel_x = 4
        panel_y = strip_y + 2

        # Header
        stats = zone.get_stats()
        gen = stats.get('generation', 0)
        slot = ui.config.active_slot
        header = f"P{player_id + 1} TRAINING - Bot {slot + 1} (Gen {gen})"
        self.screen.blit(self.font.render(header, True, team_color), (panel_x, panel_y))

        if setup_remaining > 0:
            secs = setup_remaining // settings.FPS + 1
            setup_txt = self.font.render(f"SETUP {secs}s", True, settings.YELLOW)
            self.screen.blit(setup_txt, (panel_x + 260, panel_y))

        # Config rows
        row_h = 16
        row_y_start = panel_y + 20

        for row in range(ui.NUM_ROWS):
            ry = row_y_start + row * row_h
            if ry + row_h > strip_y + strip_h:
                break

            is_selected = (row == ui.cursor_row)
            label = ui.ROW_LABELS[row]
            value = ui.get_value_str(row)

            # Highlight selected row
            if is_selected:
                pygame.draw.rect(self.screen, (40, 40, 60),
                                 (panel_x, ry, panel_w - 8, row_h))
                indicator = ">"
            else:
                indicator = " "

            label_color = settings.WHITE if is_selected else settings.LIGHT_GRAY
            value_color = settings.YELLOW if is_selected else settings.WHITE

            # Separator before fitness section
            if row == ui.ROW_FITNESS_START:
                sep_y = ry - 2
                pygame.draw.line(self.screen, settings.MID_GRAY,
                                 (panel_x, sep_y), (panel_x + panel_w - 12, sep_y), 1)

            txt = f"{indicator} {label}:"
            self.screen.blit(self.small_font.render(txt, True, label_color), (panel_x, ry + 1))
            self.screen.blit(self.small_font.render(value, True, value_color),
                             (panel_x + 140, ry + 1))

            # Show slot generation counts on the design row
            if row == ui.ROW_DESIGN:
                slot_gens = stats.get('slot_generations', {})
                for s in range(3):
                    sx = panel_x + 200 + s * 50
                    sg = slot_gens.get(s, 0)
                    sc = team_color if s == slot else settings.MID_GRAY
                    slot_txt = f"B{s+1}:G{sg}"
                    self.screen.blit(self.small_font.render(slot_txt, True, sc), (sx, ry + 1))

        # --- Arena viewport (right side) ---
        vp_x = panel_w + 4
        vp_y = strip_y + 4
        vp_w = settings.SCREEN_WIDTH - panel_w - 8
        vp_h = strip_h - 8

        # Viewport background
        pygame.draw.rect(self.screen, (20, 20, 28), (vp_x, vp_y, vp_w, vp_h))
        pygame.draw.rect(self.screen, (50, 50, 60), (vp_x, vp_y, vp_w, vp_h), 1)

        # Scale from sim coords to viewport
        scale_x = vp_w / zone.width
        scale_y = vp_h / zone.height
        scale = min(scale_x, scale_y)
        scaled_w = zone.width * scale
        scaled_h = zone.height * scale
        ox = vp_x + (vp_w - scaled_w) / 2
        oy = vp_y + (vp_h - scaled_h) / 2

        # Draw robots
        self._draw_zone_robots(zone, player_id, ox, oy, scale)

        # Draw bullets
        for bullet in zone.bullets:
            if not bullet.alive:
                continue
            bx = int(bullet.pos[0] * scale + ox)
            by = int(bullet.pos[1] * scale + oy)
            color = settings.TEAM_COLORS[bullet.team]
            bright = tuple(min(255, c + 80) for c in color)
            pygame.draw.circle(self.screen, bright, (bx, by), max(1, int(2 * scale)))

        # Draw resource drops
        for res_pos in zone.resources:
            rx = int(res_pos[0] * scale + ox)
            ry = int(res_pos[1] * scale + oy)
            pygame.draw.circle(self.screen, settings.YELLOW, (rx, ry), max(2, int(3 * scale)))

        # Stats overlay in viewport
        fit_txt = f"Best:{stats['best_fitness']:.0f} Avg:{stats['avg_fitness']:.0f}"
        alive_txt = f"{stats['alive_students']}/{stats['total_students']} alive"
        tick_txt = f"{stats['gen_tick']}/{settings.TRAINING_TICKS_PER_GENERATION}"

        self.screen.blit(self.small_font.render(fit_txt, True, settings.GREEN),
                         (vp_x + 3, vp_y + 2))
        self.screen.blit(self.small_font.render(alive_txt, True, settings.LIGHT_GRAY),
                         (vp_x + vp_w - 70, vp_y + 2))
        self.screen.blit(self.small_font.render(tick_txt, True, settings.MID_GRAY),
                         (vp_x + vp_w - 55, vp_y + 14))

    def _draw_zone_robots(self, zone, player_id: int, ox: float, oy: float, scale: float):
        """Draw robots inside a training zone viewport (scaled)."""
        for robot in zone.all_robots:
            if not robot.alive:
                continue

            is_student = robot.team == 0
            team_color = settings.TEAM_COLORS[player_id] if is_student else (150, 80, 80)
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
