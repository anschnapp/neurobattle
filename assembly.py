"""Pre-game robot assembly screen.

Both players design 3 robot blueprints on a grid.
Controls: move cursor, place/cycle block types, set direction, switch slots, ready up.
"""

from __future__ import annotations

import json
import math
import os
import time

import pygame

import settings
from modules import RobotBlueprint, Block, BlockType, Direction

SAVE_FILE = os.path.join(os.path.dirname(__file__), "last_designs.json")
from renderer import BLOCK_COLORS

# --- Grid config ---
GRID_HALF = 4                       # grid coords: -4 to 4
GRID_SIZE = GRID_HALF * 2 + 1       # 9x9
CELL_SIZE = 44                       # pixels per cell

# Special cursor rows (outside the grid)
TAB_ROW = -GRID_HALF - 1            # -5: slot selector
READY_ROW = GRID_HALF + 1           #  5: ready button

# Block type cycle order (primary action cycles through these)
BLOCK_CYCLE = [
    BlockType.PLAIN,
    BlockType.ENGINE,
    BlockType.WEAPON,
    BlockType.SENSOR,
    BlockType.SCANNER,
    BlockType.GATHERER,
    None,  # remove
]

DIRECTION_CYCLE = [Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN]

# Directional block types (secondary cycles their direction)
DIRECTIONAL_TYPES = {
    BlockType.ENGINE, BlockType.WEAPON, BlockType.SENSOR,
    BlockType.SCANNER, BlockType.GATHERER,
}

# Short labels for block types
BLOCK_LABELS = {
    BlockType.PLAIN:    "A",
    BlockType.ENGINE:   "E",
    BlockType.WEAPON:   "W",
    BlockType.SENSOR:   "S",
    BlockType.SCANNER:  "Sc",
    BlockType.GATHERER: "G",
}

# Key repeat timing (frames)
REPEAT_DELAY = 12   # frames before repeat kicks in
REPEAT_RATE = 4      # frames between repeats once active


# ---------------------------------------------------------------------------
# Per-player assembly state
# ---------------------------------------------------------------------------

class PlayerAssembly:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.cursor_x = 0
        self.cursor_y = 0
        self.current_slot = 0
        self.blueprints: list[RobotBlueprint] = [RobotBlueprint() for _ in range(3)]
        self.ready = False

        # Key state tracking (for edge detection and repeat)
        self._prev = {}            # action -> bool
        self._hold_frames = {}     # action -> int

    @property
    def current_blueprint(self) -> RobotBlueprint:
        return self.blueprints[self.current_slot]

    def get_block_at(self, gx: int, gy: int) -> Block | None:
        for b in self.current_blueprint.blocks:
            if b.grid_x == gx and b.grid_y == gy:
                return b
        return None

    # --- input handling ---------------------------------------------------

    def handle_input(self, keys_pressed):
        binds = settings.PLAYER_KEYS[self.player_id]
        just = {}
        for action, key in binds.items():
            cur = keys_pressed[key]
            prev = self._prev.get(action, False)
            just[action] = cur and not prev

            # Track hold duration for directional repeat
            if action in ('up', 'down', 'left', 'right'):
                if cur and not prev:
                    self._hold_frames[action] = 0
                elif cur:
                    self._hold_frames[action] = self._hold_frames.get(action, 0) + 1
                else:
                    self._hold_frames[action] = 0

            self._prev[action] = cur

        # When ready, only allow un-readying
        if self.ready:
            if just.get('primary') and self.cursor_y == READY_ROW:
                self.ready = False
            return

        # Movement (with repeat)
        def should_move(a):
            if just.get(a):
                return True
            t = self._hold_frames.get(a, 0)
            return t > REPEAT_DELAY and (t - REPEAT_DELAY) % REPEAT_RATE == 0

        dx = (1 if should_move('right') else 0) - (1 if should_move('left') else 0)
        dy = (1 if should_move('down') else 0) - (1 if should_move('up') else 0)
        if dx or dy:
            self._move_cursor(dx, dy)

        if just.get('primary'):
            self._primary()
        if just.get('secondary'):
            self._secondary()

    def _move_cursor(self, dx: int, dy: int):
        nx = self.cursor_x + dx
        ny = self.cursor_y + dy

        # Vertical clamp
        ny = max(TAB_ROW, min(READY_ROW, ny))

        if ny == TAB_ROW:
            # Entering tab row: snap x to current slot
            if self.cursor_y != TAB_ROW:
                nx = self.current_slot
            nx = max(0, min(2, nx))
        elif ny == READY_ROW:
            nx = 0
        else:
            nx = max(-GRID_HALF, min(GRID_HALF, nx))

        self.cursor_x = nx
        self.cursor_y = ny

    def _primary(self):
        if self.cursor_y == TAB_ROW:
            self.current_slot = self.cursor_x
        elif self.cursor_y == READY_ROW:
            if self._can_ready():
                self.ready = True
        else:
            self._cycle_block_type()

    def _secondary(self):
        if self.cursor_y == TAB_ROW or self.cursor_y == READY_ROW:
            return
        block = self.get_block_at(self.cursor_x, self.cursor_y)
        if block and block.block_type in DIRECTIONAL_TYPES:
            idx = DIRECTION_CYCLE.index(block.direction)
            block.direction = DIRECTION_CYCLE[(idx + 1) % len(DIRECTION_CYCLE)]

    def _cycle_block_type(self):
        block = self.get_block_at(self.cursor_x, self.cursor_y)
        if block is None:
            self.current_blueprint.add_block(
                self.cursor_x, self.cursor_y, BlockType.PLAIN, Direction.RIGHT,
            )
        else:
            idx = BLOCK_CYCLE.index(block.block_type)
            nxt = BLOCK_CYCLE[(idx + 1) % len(BLOCK_CYCLE)]
            if nxt is None:
                self.current_blueprint.blocks.remove(block)
            else:
                block.block_type = nxt
                if nxt == BlockType.PLAIN:
                    block.direction = Direction.RIGHT

    def _can_ready(self) -> bool:
        return all(len(bp.blocks) > 0 for bp in self.blueprints)


# ---------------------------------------------------------------------------
# Assembly screen (draws both players side by side)
# ---------------------------------------------------------------------------

class AssemblyScreen:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.players = [PlayerAssembly(0), PlayerAssembly(1)]
        self._load_designs()

        self.font = pygame.font.SysFont("monospace", 14)
        self.big_font = pygame.font.SysFont("monospace", 22, bold=True)
        self.title_font = pygame.font.SysFont("monospace", 32, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 12)

        # Layout
        self.panel_w = settings.SCREEN_WIDTH // 2 - 20
        self.panel_x = [10, settings.SCREEN_WIDTH // 2 + 10]
        grid_px = GRID_SIZE * CELL_SIZE
        self.grid_ox = (self.panel_w - grid_px) // 2   # x offset within panel
        self.grid_oy = 120                               # y offset from top

    @property
    def both_ready(self) -> bool:
        return all(p.ready for p in self.players)

    def get_blueprints(self) -> list[list[RobotBlueprint]]:
        """Return [p1_blueprints, p2_blueprints]. Also saves designs to disk."""
        self._save_designs()
        return [p.blueprints for p in self.players]

    # --- save / load -------------------------------------------------------

    def _save_designs(self):
        data = {
            f"player_{i}": [bp.to_dict() for bp in p.blueprints]
            for i, p in enumerate(self.players)
        }
        try:
            with open(SAVE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass  # non-critical

    def _load_designs(self):
        try:
            with open(SAVE_FILE) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        for i, p in enumerate(self.players):
            key = f"player_{i}"
            if key not in data:
                continue
            blueprints = []
            for bp_data in data[key]:
                try:
                    blueprints.append(RobotBlueprint.from_dict(bp_data))
                except (KeyError, ValueError):
                    blueprints.append(RobotBlueprint())
            # Pad to 3 if needed
            while len(blueprints) < 3:
                blueprints.append(RobotBlueprint())
            p.blueprints = blueprints[:3]

    # --- tick / draw -------------------------------------------------------

    def update(self):
        keys = pygame.key.get_pressed()
        for p in self.players:
            p.handle_input(keys)

    def draw(self):
        self.screen.fill(settings.DARK_GRAY)

        # Title
        title = self.title_font.render("ROBOT ASSEMBLY", True, settings.WHITE)
        self.screen.blit(title, (settings.SCREEN_WIDTH // 2 - title.get_width() // 2, 10))

        # Center divider
        mx = settings.SCREEN_WIDTH // 2
        pygame.draw.line(self.screen, settings.MID_GRAY, (mx, 50), (mx, settings.SCREEN_HEIGHT), 2)

        for i, player in enumerate(self.players):
            self._draw_panel(player, self.panel_x[i])

        if self.both_ready:
            surf = self.big_font.render("BOTH READY - STARTING...", True, settings.GREEN)
            self.screen.blit(surf, (settings.SCREEN_WIDTH // 2 - surf.get_width() // 2,
                                    settings.SCREEN_HEIGHT - 40))

    # --- per-player panel --------------------------------------------------

    def _draw_panel(self, p: PlayerAssembly, px: int):
        color = settings.TEAM_COLORS[p.player_id]

        # Player label
        label = self.big_font.render(f"PLAYER {p.player_id + 1}", True, color)
        self.screen.blit(label, (px + 10, 50))

        self._draw_tabs(p, px)
        self._draw_grid(p, px)
        self._draw_info(p, px)
        self._draw_ready(p, px)
        self._draw_legend(p, px)
        self._draw_hint(p, px)

    # --- slot tabs ---------------------------------------------------------

    def _draw_tabs(self, p: PlayerAssembly, px: int):
        tw, th, gap = 100, 30, 20
        total = 3 * tw + 2 * gap
        sx = px + (self.panel_w - total) // 2
        ty = 80

        for i in range(3):
            tx = sx + i * (tw + gap)
            n = len(p.blueprints[i].blocks)
            selected = (i == p.current_slot)
            hovered = (p.cursor_y == TAB_ROW and p.cursor_x == i)

            fill = (50, 50, 70) if selected else (30, 30, 40)
            border = settings.YELLOW if hovered else (settings.WHITE if selected else settings.MID_GRAY)

            pygame.draw.rect(self.screen, fill, (tx, ty, tw, th))
            pygame.draw.rect(self.screen, border, (tx, ty, tw, th), 2)

            txt = self.font.render(f"Bot {i + 1} [{n}]", True,
                                   settings.WHITE if selected else settings.LIGHT_GRAY)
            self.screen.blit(txt, (tx + tw // 2 - txt.get_width() // 2, ty + 8))

    # --- grid --------------------------------------------------------------

    def _draw_grid(self, p: PlayerAssembly, px: int):
        gx0 = px + self.grid_ox
        gy0 = self.grid_oy

        # Grid lines
        for r in range(GRID_SIZE + 1):
            y = gy0 + r * CELL_SIZE
            pygame.draw.line(self.screen, (40, 40, 50),
                             (gx0, y), (gx0 + GRID_SIZE * CELL_SIZE, y))
        for c in range(GRID_SIZE + 1):
            x = gx0 + c * CELL_SIZE
            pygame.draw.line(self.screen, (40, 40, 50),
                             (x, gy0), (x, gy0 + GRID_SIZE * CELL_SIZE))

        # Center dot
        cx = gx0 + GRID_HALF * CELL_SIZE + CELL_SIZE // 2
        cy = gy0 + GRID_HALF * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self.screen, (60, 60, 70), (cx, cy), 3)

        # Blocks
        team_color = settings.TEAM_COLORS[p.player_id]
        for block in p.current_blueprint.blocks:
            col = block.grid_x + GRID_HALF
            row = block.grid_y + GRID_HALF
            bx = gx0 + col * CELL_SIZE + 2
            by = gy0 + row * CELL_SIZE + 2
            bw = CELL_SIZE - 4
            bh = CELL_SIZE - 4

            base_c = BLOCK_COLORS[block.block_type]
            color = tuple((bc + tc) // 2 for bc, tc in zip(base_c, team_color))

            pygame.draw.rect(self.screen, color, (bx, by, bw, bh))
            pygame.draw.rect(self.screen, settings.WHITE, (bx, by, bw, bh), 1)

            # Direction arrow for directional blocks
            if block.block_type in DIRECTIONAL_TYPES:
                mid_x = bx + bw // 2
                mid_y = by + bh // 2
                dx = math.cos(block.direction.angle) * (bw // 2 - 4)
                dy = math.sin(block.direction.angle) * (bh // 2 - 4)
                ex, ey = int(mid_x + dx), int(mid_y + dy)
                pygame.draw.line(self.screen, settings.WHITE, (mid_x, mid_y), (ex, ey), 2)
                pygame.draw.circle(self.screen, settings.WHITE, (ex, ey), 3)

            # Type label
            lbl = BLOCK_LABELS.get(block.block_type, "?")
            lbl_s = self.small_font.render(lbl, True, settings.WHITE)
            self.screen.blit(lbl_s, (bx + 3, by + 2))

        # Cursor
        if -GRID_HALF <= p.cursor_y <= GRID_HALF:
            col = p.cursor_x + GRID_HALF
            row = p.cursor_y + GRID_HALF
            rx = gx0 + col * CELL_SIZE
            ry = gy0 + row * CELL_SIZE
            pulse = int(abs(math.sin(time.time() * 4)) * 100) + 155
            pygame.draw.rect(self.screen, (pulse, pulse, 0),
                             (rx, ry, CELL_SIZE, CELL_SIZE), 3)

    # --- info below grid ---------------------------------------------------

    def _draw_info(self, p: PlayerAssembly, px: int):
        y0 = self.grid_oy + GRID_SIZE * CELL_SIZE + 10

        # Block under cursor
        if -GRID_HALF <= p.cursor_y <= GRID_HALF:
            block = p.get_block_at(p.cursor_x, p.cursor_y)
            if block:
                txt = f"[{block.block_type.name}] dir: {block.direction.name}"
            else:
                txt = "[empty] primary to place"
            self.screen.blit(self.font.render(txt, True, settings.LIGHT_GRAY), (px + 10, y0))

        # Blueprint stats
        bp = p.current_blueprint
        nb = len(bp.blocks)
        ne = sum(1 for b in bp.blocks if b.block_type == BlockType.ENGINE)
        nw = sum(1 for b in bp.blocks if b.block_type == BlockType.WEAPON)
        ns = sum(1 for b in bp.blocks if b.block_type == BlockType.SENSOR)
        stats = f"Blocks:{nb}  E:{ne} W:{nw} S:{ns}  Brain:{bp.brain_input_size}in/{bp.brain_output_size}out"
        self.screen.blit(self.font.render(stats, True, settings.LIGHT_GRAY), (px + 10, y0 + 20))

    # --- ready button ------------------------------------------------------

    def _draw_ready(self, p: PlayerAssembly, px: int):
        ry = self.grid_oy + GRID_SIZE * CELL_SIZE + 60
        bw, bh = 200, 40
        bx = px + (self.panel_w - bw) // 2
        hovered = (p.cursor_y == READY_ROW)
        can = p._can_ready()

        if p.ready:
            fill, border, txt, tc = (30, 100, 30), settings.GREEN, "READY!", settings.GREEN
        elif can:
            fill = (40, 40, 60)
            border = settings.YELLOW if hovered else settings.MID_GRAY
            txt, tc = "READY?", settings.WHITE
        else:
            fill, border = (30, 30, 30), settings.MID_GRAY
            txt, tc = "need 3 designs", settings.MID_GRAY

        pygame.draw.rect(self.screen, fill, (bx, ry, bw, bh))
        pygame.draw.rect(self.screen, border, (bx, ry, bw, bh), 3 if hovered else 2)
        surf = self.big_font.render(txt, True, tc)
        self.screen.blit(surf, (bx + bw // 2 - surf.get_width() // 2,
                                ry + bh // 2 - surf.get_height() // 2))

    # --- block type legend -------------------------------------------------

    def _draw_legend(self, p: PlayerAssembly, px: int):
        y = self.grid_oy + GRID_SIZE * CELL_SIZE + 110
        items = [
            ("A", "Armor", BLOCK_COLORS[BlockType.PLAIN]),
            ("E", "Engine", BLOCK_COLORS[BlockType.ENGINE]),
            ("W", "Weapon", BLOCK_COLORS[BlockType.WEAPON]),
            ("S", "Sensor", BLOCK_COLORS[BlockType.SENSOR]),
            ("Sc", "Scanner", BLOCK_COLORS[BlockType.SCANNER]),
            ("G", "Gatherer", BLOCK_COLORS[BlockType.GATHERER]),
        ]
        x = px + 10
        for short, name, color in items:
            pygame.draw.rect(self.screen, color, (x, y, 12, 12))
            lbl = self.small_font.render(f"{short}={name}", True, settings.LIGHT_GRAY)
            self.screen.blit(lbl, (x + 16, y))
            x += lbl.get_width() + 26

    # --- controls hint -----------------------------------------------------

    def _draw_hint(self, p: PlayerAssembly, px: int):
        y = settings.SCREEN_HEIGHT - 30
        if p.player_id == 0:
            txt = "WASD:move  E:place/cycle  Q:rotate dir"
        else:
            txt = "Arrows:move  RCtrl:place/cycle  RShift:rotate dir"
        self.screen.blit(self.font.render(txt, True, (80, 80, 80)), (px + 10, y))
