"""Pygame rendering for the battlefield and training arenas."""

from __future__ import annotations

import math
import pygame
import numpy as np

from entities import Robot, Bullet, Base
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
    BlockType.RADAR:    (40, 220, 200),    # cyan
    BlockType.BEACON:   (255, 160, 40),    # orange
}


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont("monospace", 14)
        self.big_font = pygame.font.SysFont("monospace", 28, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 11)
        self.gen_font = pygame.font.SysFont("monospace", 9)

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

    def _draw_robot_perception(self, robot: Robot, ox: int, oy: int, scale: float = 1.0):
        """Draw sensor cones, health bar, and speed arrow for a robot."""
        rx = robot.pos[0] * scale + ox
        ry = robot.pos[1] * scale + oy
        radius = robot.radius * scale

        # --- Sensor cones (only for full Robot objects with block methods) ---
        if hasattr(robot, 'get_block_world_pos'):
            for block in robot.blocks:
                if block.block_type != BlockType.SENSOR:
                    continue
                s_pos = robot.get_block_world_pos(block)
                sx = s_pos[0] * scale + ox
                sy = s_pos[1] * scale + oy
                s_angle = robot.get_block_world_angle(block)
                s_range = block.sensor_range * scale
                half_fov = block.sensor_fov / 2

                # Cone edges
                a1 = s_angle - half_fov
                a2 = s_angle + half_fov
                tip = (int(sx), int(sy))
                p1 = (int(sx + math.cos(a1) * s_range), int(sy + math.sin(a1) * s_range))
                p2 = (int(sx + math.cos(a2) * s_range), int(sy + math.sin(a2) * s_range))
                # Draw cone as lines
                pygame.draw.line(self.screen, (60, 160, 220), tip, p1, 1)
                pygame.draw.line(self.screen, (60, 160, 220), tip, p2, 1)
                # Arc along the outer edge
                n_segs = 6
                prev = p1
                for i in range(1, n_segs + 1):
                    t = i / n_segs
                    a = a1 + t * (a2 - a1)
                    p = (int(sx + math.cos(a) * s_range), int(sy + math.sin(a) * s_range))
                    pygame.draw.line(self.screen, (60, 160, 220), prev, p, 1)
                    prev = p

        # --- Health bar (only show after first hit) ---
        hp_frac = robot.hp / robot.max_hp if robot.max_hp > 0 else 0.0
        if hp_frac < 1.0:
            bar_w = max(radius * 2, 12)
            bar_h = max(2 * scale, 2)
            bar_x = rx - bar_w / 2
            bar_y = ry - radius - 4 * scale
            # Background (dark red)
            pygame.draw.rect(self.screen, (80, 20, 20),
                             (int(bar_x), int(bar_y), int(bar_w), int(bar_h)))
            # Fill (green -> yellow -> red)
            if hp_frac > 0.5:
                g = 200
                r = int((1.0 - hp_frac) * 2 * 200)
            else:
                r = 200
                g = int(hp_frac * 2 * 200)
            fill_w = int(bar_w * hp_frac)
            if fill_w > 0:
                pygame.draw.rect(self.screen, (r, g, 40),
                                 (int(bar_x), int(bar_y), fill_w, int(bar_h)))

        # --- Speed arrow ---
        speed = float(np.linalg.norm(robot.velocity))
        if speed > 0.15:
            speed_frac = min(speed / settings.ROBOT_DEFAULT_SPEED, 1.0)
            arrow_len = (radius + 8) * speed_frac * scale
            vx = robot.velocity[0] / speed
            vy = robot.velocity[1] / speed
            ax = rx + vx * (radius * scale + 2)
            ay = ry + vy * (radius * scale + 2)
            ex = ax + vx * arrow_len
            ey = ay + vy * arrow_len
            pygame.draw.line(self.screen, (180, 180, 180), (int(ax), int(ay)), (int(ex), int(ey)), 1)

    def draw_robot(self, robot: Robot, offset_x: int = 0, offset_y: int = 0):
        if not robot.alive:
            return

        oy = offset_y if offset_y else settings.BATTLEFIELD_Y
        ox = offset_x

        # Draw perception overlays behind the robot
        self._draw_robot_perception(robot, ox, oy)

        team_color = settings.TEAM_COLORS[robot.team]
        cos_a = math.cos(robot.angle)
        sin_a = math.sin(robot.angle)
        half = BLOCK_PIXEL_SIZE / 2

        for block in robot.blocks:
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

        # Generation number below the robot
        gen_label = str(robot.generation)
        gen_color = settings.TEAM_COLORS[robot.team]
        gen_surf = self.gen_font.render(gen_label, True, gen_color)
        gx = int(robot.pos[0]) + ox - gen_surf.get_width() // 2
        gy = int(robot.pos[1]) + oy + int(robot.radius) + 2
        self.screen.blit(gen_surf, (gx, gy))

    def draw_bullet(self, bullet: Bullet, offset_x: int = 0, offset_y: int = 0):
        if not bullet.alive:
            return
        oy = offset_y if offset_y else settings.BATTLEFIELD_Y
        ox = offset_x
        x, y = int(bullet.pos[0]) + ox, int(bullet.pos[1]) + oy
        color = settings.TEAM_COLORS[bullet.team]
        bright = tuple(min(255, c + 100) for c in color)
        pygame.draw.circle(self.screen, bright, (x, y), bullet.radius)

    def draw_hud(self, bases: list[Base], robots: list[Robot], tick: int,
                 resources: list[float] | None = None):
        oy = settings.BATTLEFIELD_Y
        for i, base in enumerate(bases):
            side = "P1" if base.team == 0 else "P2"
            color = settings.TEAM_COLORS[base.team]
            wall_str = f"Wall:{int(base.wall_hp)}" if base.wall_alive else "BREACHED"
            cmd_str = "CMD:OK" if base.commander_alive else "CMD:DEAD"
            n_bots = sum(1 for r in robots if r.team == base.team and r.alive)
            res_str = ""
            if resources is not None:
                res_str = f"  ${int(resources[base.team])}"
            text = f"{side}  {wall_str}  {cmd_str}  Bots:{n_bots}{res_str}"
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

        # Three-column layout: spawn (left), config (center), fitness (right)
        row_h = 16
        row_y_start = panel_y + 20
        # Per-column widths and value offsets (proportional to content)
        spawn_col_w = 130
        config_col_w = 170
        fitness_col_w = panel_w - spawn_col_w - config_col_w
        spawn_val_off = 88
        config_val_off = 98
        fitness_val_off = 108

        # Resource display in header area
        res_txt = f"${int(ui.resources)}"
        res_surf = self.font.render(res_txt, True, settings.YELLOW)
        self.screen.blit(res_surf, (panel_x + panel_w - res_surf.get_width() - 4, panel_y))

        # --- Spawn column (left, col 0) ---
        spawn_x = panel_x
        for row in range(ui.NUM_SPAWN_ROWS):
            ry = row_y_start + row * row_h
            if ry + row_h > strip_y + strip_h:
                break

            is_selected = (ui.cursor_col == 0 and row == ui.cursor_row)
            label = ui.SPAWN_LABELS[row]
            value = ui.get_spawn_value_str(row)

            is_spawn_row = row in (ui.ROW_SPAWN_0, ui.ROW_SPAWN_1, ui.ROW_SPAWN_2)
            is_destroy_row = (row == ui.ROW_DESTROY)

            if is_selected:
                pygame.draw.rect(self.screen, (40, 40, 60),
                                 (spawn_x, ry, spawn_col_w - 4, row_h))
                indicator = ">"
            else:
                indicator = " "

            if is_spawn_row:
                slot_idx = row - ui.ROW_SPAWN_0
                bp = ui.blueprints[slot_idx]
                if not bp.blocks:
                    label_color = settings.MID_GRAY
                    value_color = settings.MID_GRAY
                else:
                    cost = len(bp.blocks) * settings.SPAWN_COST_PER_BLOCK
                    affordable = ui.resources >= cost
                    label_color = settings.GREEN if affordable else settings.RED
                    value_color = settings.GREEN if affordable else settings.RED
                    if is_selected:
                        label_color = settings.WHITE
                        value_color = settings.YELLOW
            elif is_destroy_row:
                label_color = settings.ORANGE if is_selected else (180, 100, 40)
                value_color = label_color
            else:
                label_color = settings.WHITE if is_selected else settings.LIGHT_GRAY
                value_color = settings.YELLOW if is_selected else settings.WHITE

            txt = f"{indicator} {label}:"
            self.screen.blit(self.small_font.render(txt, True, label_color), (spawn_x, ry + 1))
            self.screen.blit(self.small_font.render(value, True, value_color),
                             (spawn_x + spawn_val_off, ry + 1))

        # --- Config column (center, col 1) ---
        cfg_x = panel_x + spawn_col_w
        for row in range(ui.NUM_CONFIG_ROWS):
            ry = row_y_start + row * row_h
            if ry + row_h > strip_y + strip_h:
                break

            is_selected = (ui.cursor_col == 1 and row == ui.cursor_row)
            label = ui.CONFIG_LABELS[row]
            value = ui.get_config_value_str(row)

            if is_selected:
                pygame.draw.rect(self.screen, (40, 40, 60),
                                 (cfg_x, ry, config_col_w - 4, row_h))
                indicator = ">"
            else:
                indicator = " "

            label_color = settings.WHITE if is_selected else settings.LIGHT_GRAY
            value_color = settings.YELLOW if is_selected else settings.WHITE

            txt = f"{indicator} {label}:"
            self.screen.blit(self.small_font.render(txt, True, label_color), (cfg_x, ry + 1))
            self.screen.blit(self.small_font.render(value, True, value_color),
                             (cfg_x + config_val_off, ry + 1))

            # Show slot generation counts on the design row
            if row == ui.ROW_DESIGN:
                slot_gens = stats.get('slot_generations', {})
                for s in range(3):
                    sx = cfg_x + config_val_off + 48 + s * 35
                    sg = slot_gens.get(s, 0)
                    sc = team_color if s == slot else settings.MID_GRAY
                    slot_txt = f"B{s+1}:G{sg}"
                    self.screen.blit(self.small_font.render(slot_txt, True, sc), (sx, ry + 1))

        # --- Fitness column (right, col 2) ---
        fit_x = panel_x + spawn_col_w + config_col_w
        for row in range(ui.NUM_FITNESS_ROWS):
            ry = row_y_start + row * row_h
            if ry + row_h > strip_y + strip_h:
                break

            is_selected = (ui.cursor_col == 2 and row == ui.cursor_row)
            label = ui.FITNESS_LABELS[row]
            value = ui.get_fitness_value_str(row)

            if is_selected:
                pygame.draw.rect(self.screen, (40, 40, 60),
                                 (fit_x, ry, fitness_col_w - 4, row_h))
                indicator = ">"
            else:
                indicator = " "

            label_color = settings.WHITE if is_selected else settings.LIGHT_GRAY

            # Color-code fitness values
            key = FITNESS_PARAMS[row][0]
            fval = ui.config.fitness_weights.get(key, FITNESS_PARAMS[row][2])
            if is_selected:
                value_color = settings.YELLOW
            elif fval > 0:
                value_color = settings.GREEN
            elif fval < 0:
                value_color = settings.RED
            else:
                value_color = settings.MID_GRAY

            txt = f"{indicator} {label}:"
            self.screen.blit(self.small_font.render(txt, True, label_color), (fit_x, ry + 1))
            self.screen.blit(self.small_font.render(value, True, value_color),
                             (fit_x + fitness_val_off, ry + 1))

        # --- Arena viewport (right side) ---
        vp_x = panel_w + 4
        vp_y = strip_y + 4
        vp_w = settings.SCREEN_WIDTH - panel_w - 8
        vp_h = strip_h - 8

        # Scale from sim coords to viewport
        scale_x = vp_w / zone.width
        scale_y = vp_h / zone.height
        scale = min(scale_x, scale_y)
        scaled_w = zone.width * scale
        scaled_h = zone.height * scale
        ox = vp_x + (vp_w - scaled_w) / 2
        oy = vp_y + (vp_h - scaled_h) / 2

        # Viewport background — only the actual rendered area
        pygame.draw.rect(self.screen, (20, 20, 28),
                         (int(ox), int(oy), int(scaled_w), int(scaled_h)))
        pygame.draw.rect(self.screen, (50, 50, 60),
                         (int(ox), int(oy), int(scaled_w), int(scaled_h)), 1)

        # Draw bases
        base_r = int(settings.BASE_RADIUS * scale)
        for base_pos, is_friendly in ((zone.friendly_base_pos, True),
                                       (zone.enemy_base_pos, False)):
            bx = int(base_pos[0] * scale + ox)
            by = int(base_pos[1] * scale + oy)
            wall_hp = zone.friendly_base_wall_hp if is_friendly else zone.enemy_base_wall_hp
            wall_alive = wall_hp > 0
            team_color = settings.TEAM_COLORS[player_id] if is_friendly else settings.TEAM_COLORS[1 - player_id]
            if wall_alive:
                # Brightness scales with remaining HP
                wall_alpha = max(0.2, wall_hp / settings.BASE_WALL_HP)
                base_color = tuple(int(c * wall_alpha * 0.5) for c in team_color)
                pygame.draw.circle(self.screen, base_color, (bx, by),
                                   base_r + 2, max(1, int(3 * scale)))
            else:
                # Wall destroyed — dim outline
                pygame.draw.circle(self.screen, (40, 40, 40), (bx, by),
                                   base_r, max(1, int(1 * scale)))
            # Commander dot
            pygame.draw.circle(self.screen, team_color, (bx, by),
                               max(2, int(settings.COMMANDER_RADIUS * scale)))

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

            # Draw perception overlays (scaled)
            self._draw_robot_perception(robot, int(ox), int(oy), scale)

            is_student = robot.team == 0
            team_color = settings.TEAM_COLORS[player_id] if is_student else (150, 80, 80)
            cos_a = math.cos(robot.angle)
            sin_a = math.sin(robot.angle)
            half = BLOCK_PIXEL_SIZE / 2 * scale

            for block in robot.blocks:
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

                pygame.draw.polygon(self.screen, color, corners_world)
