"""Block-based modular robot components.

A robot is a collection of small rectangular blocks arranged on a grid.
Each block can optionally have a special function (engine, weapon, sensor, scanner).
Directional modules face a specific direction relative to the robot body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto


class BlockType(Enum):
    PLAIN = auto()     # just structure / armor
    ENGINE = auto()    # provides thrust in its facing direction
    WEAPON = auto()    # shoots in its facing direction
    SENSOR = auto()    # detects entities in its facing direction
    SCANNER = auto()   # scans enemies in its facing direction
    GATHERER = auto()  # magnetically collects battlefield resource drops


class Direction(Enum):
    """Facing direction relative to robot body (robot faces RIGHT by default)."""
    RIGHT = 0     #  0 degrees
    UP = 1        # -90 degrees (screen coords: up = negative y)
    LEFT = 2      # 180 degrees
    DOWN = 3      #  90 degrees

    @property
    def angle(self) -> float:
        """Angle in radians."""
        return [0.0, -math.pi / 2, math.pi, math.pi / 2][self.value]

    @property
    def dx(self) -> float:
        return [1.0, 0.0, -1.0, 0.0][self.value]

    @property
    def dy(self) -> float:
        return [0.0, -1.0, 0.0, 1.0][self.value]


@dataclass
class Block:
    """A single rectangular module in the robot body."""
    grid_x: int            # position on robot grid (0,0 = center)
    grid_y: int
    block_type: BlockType = BlockType.PLAIN
    direction: Direction = Direction.RIGHT  # facing direction for directional modules
    hp: float = 20.0
    max_hp: float = 20.0
    alive: bool = True

    # Weapon state
    cooldown: int = 0
    fire_rate: int = 30       # ticks between shots (weapon only)
    bullet_speed: float = 6.0
    bullet_damage: float = 10.0

    # Engine state
    thrust: float = 0.5       # acceleration per tick (engine only)

    # Sensor state
    sensor_range: float = 150.0
    sensor_fov: float = math.pi / 3  # 60 degree cone

    # Scanner state
    scan_range: float = 100.0
    scan_cooldown: int = 0
    scan_rate: int = 180

    def tick(self):
        if self.cooldown > 0:
            self.cooldown -= 1
        if self.scan_cooldown > 0:
            self.scan_cooldown -= 1

    def take_damage(self, amount: float):
        self.hp -= amount
        if self.hp <= 0:
            self.hp = 0
            self.alive = False


# --- Block size for rendering ---
BLOCK_PIXEL_SIZE = 8  # each block is 8x8 pixels


@dataclass
class RobotBlueprint:
    """A robot design: a collection of blocks forming a body.

    The blueprint defines the shape and capabilities.
    Brain size is derived from the blocks present.
    """
    blocks: list[Block] = field(default_factory=list)
    hidden_size: int = 16

    def add_block(self, grid_x: int, grid_y: int,
                  block_type: BlockType = BlockType.PLAIN,
                  direction: Direction = Direction.RIGHT) -> Block:
        block = Block(grid_x=grid_x, grid_y=grid_y,
                      block_type=block_type, direction=direction)
        self.blocks.append(block)
        return block

    @property
    def engines(self) -> list[Block]:
        return [b for b in self.blocks if b.block_type == BlockType.ENGINE and b.alive]

    @property
    def weapons(self) -> list[Block]:
        return [b for b in self.blocks if b.block_type == BlockType.WEAPON and b.alive]

    @property
    def sensors(self) -> list[Block]:
        return [b for b in self.blocks if b.block_type == BlockType.SENSOR and b.alive]

    @property
    def scanners(self) -> list[Block]:
        return [b for b in self.blocks if b.block_type == BlockType.SCANNER and b.alive]

    @property
    def alive_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.alive]

    @property
    def brain_input_size(self) -> int:
        """Each sensor provides 2 inputs: [distance, type]."""
        n_sensors = len(self.sensors)
        if n_sensors == 0:
            return 1  # minimal dummy input (robot is blind)
        return n_sensors * 2

    @property
    def brain_output_size(self) -> int:
        """Each engine provides 1 output (thrust amount).
        Each weapon provides 1 output (shoot signal).
        """
        n_engines = sum(1 for b in self.blocks if b.block_type == BlockType.ENGINE)
        n_weapons = sum(1 for b in self.blocks if b.block_type == BlockType.WEAPON)
        total = n_engines + n_weapons
        if total == 0:
            return 1  # dummy output
        return total

    def to_dict(self) -> dict:
        """Serialize blueprint to a plain dict (for JSON save)."""
        return {
            "hidden_size": self.hidden_size,
            "blocks": [
                {
                    "grid_x": b.grid_x,
                    "grid_y": b.grid_y,
                    "type": b.block_type.name,
                    "direction": b.direction.name,
                }
                for b in self.blocks
            ],
        }

    @staticmethod
    def from_dict(data: dict) -> RobotBlueprint:
        """Deserialize blueprint from a plain dict."""
        bp = RobotBlueprint(hidden_size=data.get("hidden_size", 16))
        for bd in data.get("blocks", []):
            bp.add_block(
                grid_x=bd["grid_x"],
                grid_y=bd["grid_y"],
                block_type=BlockType[bd["type"]],
                direction=Direction[bd["direction"]],
            )
        return bp

    def copy_blocks(self) -> list[Block]:
        """Deep copy all blocks (for spawning a new robot from this blueprint)."""
        return [
            Block(
                grid_x=b.grid_x, grid_y=b.grid_y,
                block_type=b.block_type, direction=b.direction,
                hp=b.max_hp, max_hp=b.max_hp, alive=True,
                fire_rate=b.fire_rate, bullet_speed=b.bullet_speed,
                bullet_damage=b.bullet_damage, thrust=b.thrust,
                sensor_range=b.sensor_range, sensor_fov=b.sensor_fov,
                scan_range=b.scan_range, scan_rate=b.scan_rate,
            )
            for b in self.blocks
        ]

    @staticmethod
    def default_fighter() -> RobotBlueprint:
        """A simple fighter: cross shape with engine, weapon, sensors."""
        bp = RobotBlueprint(hidden_size=12)
        # Core body (plain armor)
        bp.add_block(0, 0, BlockType.PLAIN)
        # Front weapon
        bp.add_block(1, 0, BlockType.WEAPON, Direction.RIGHT)
        # Rear engine
        bp.add_block(-1, 0, BlockType.ENGINE, Direction.LEFT)
        # Side sensors
        bp.add_block(0, -1, BlockType.SENSOR, Direction.UP)
        bp.add_block(0, 1, BlockType.SENSOR, Direction.DOWN)
        # Front sensor
        bp.add_block(1, -1, BlockType.SENSOR, Direction.RIGHT)
        return bp

    @staticmethod
    def default_tank() -> RobotBlueprint:
        """A bulkier design: more armor, forward weapon, rear engines."""
        bp = RobotBlueprint(hidden_size=16)
        # 3x3 core
        for x in range(-1, 2):
            for y in range(-1, 2):
                bp.add_block(x, y, BlockType.PLAIN)
        # Override specials
        bp.blocks[1].block_type = BlockType.SENSOR    # (0,-1) top sensor
        bp.blocks[1].direction = Direction.UP
        bp.blocks[3].block_type = BlockType.ENGINE     # (-1,0) left engine
        bp.blocks[3].direction = Direction.LEFT
        bp.blocks[5].block_type = BlockType.WEAPON     # (1,0) right weapon
        bp.blocks[5].direction = Direction.RIGHT
        bp.blocks[7].block_type = BlockType.SENSOR     # (0,1) bottom sensor
        bp.blocks[7].direction = Direction.DOWN
        bp.blocks[6].block_type = BlockType.ENGINE     # (-1,1) engine
        bp.blocks[6].direction = Direction.LEFT
        # Front sensor
        bp.blocks[2].block_type = BlockType.SENSOR     # (1,-1) front-top
        bp.blocks[2].direction = Direction.RIGHT
        return bp

    @staticmethod
    def minimal_scout() -> RobotBlueprint:
        """Tiny fast robot: one sensor, one engine, no weapon."""
        bp = RobotBlueprint(hidden_size=8)
        bp.add_block(0, 0, BlockType.PLAIN)
        bp.add_block(1, 0, BlockType.SENSOR, Direction.RIGHT)
        bp.add_block(-1, 0, BlockType.ENGINE, Direction.LEFT)
        return bp
