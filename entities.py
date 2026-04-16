"""Game entities: robots (block-based), bullets, bases."""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field

from brain import Brain
from modules import RobotBlueprint, Block, BlockType, BLOCK_PIXEL_SIZE

import settings


@dataclass
class Robot:
    pos: np.ndarray              # [x, y] float — center of robot
    angle: float                 # radians, 0 = right
    team: int                    # 0 or 1
    blueprint: RobotBlueprint
    brain: Brain
    blocks: list[Block] = field(default_factory=list)
    alive: bool = True
    hp: float = 0.0
    max_hp: float = 0.0
    velocity: np.ndarray = field(default=None)
    # Generation tracking (which training generation this brain came from)
    generation: int = 0
    # Fitness tracking
    hits_dealt: int = 0
    hits_friend: int = 0
    hits_ebase: int = 0
    hits_fbase: int = 0
    hits_taken: int = 0
    ticks_alive: int = 0
    distance_traveled: float = 0.0
    resources_collected: int = 0
    scans_enemy: int = 0
    # Delta-based distance tracking: initial distance and best (minimum) distance reached
    init_dist_to_enemy: float = -1.0   # -1 = not yet recorded
    init_dist_to_friend: float = -1.0
    init_dist_to_ebase: float = -1.0
    init_dist_to_fbase: float = -1.0
    best_dist_to_enemy: float = float('inf')
    best_dist_to_friend: float = float('inf')
    best_dist_to_ebase: float = float('inf')
    best_dist_to_fbase: float = float('inf')

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(2, dtype=np.float32)
        if not self.blocks:
            self.blocks = self.blueprint.copy_blocks()
        # Initialize HP pool from blocks (armor counts double)
        if self.max_hp == 0.0:
            self.max_hp = sum(b.hp_contribution for b in self.blocks)
            self.hp = self.max_hp

    @property
    def radius(self) -> float:
        """Approximate bounding radius from blocks."""
        if not self.blocks:
            return BLOCK_PIXEL_SIZE
        max_r = 0.0
        for b in self.blocks:
            dist = math.sqrt(b.grid_x ** 2 + b.grid_y ** 2) * BLOCK_PIXEL_SIZE + BLOCK_PIXEL_SIZE
            if dist > max_r:
                max_r = dist
        return max_r

    def get_block_world_pos(self, block: Block) -> np.ndarray:
        """Get the world position of a block, accounting for robot rotation."""
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        local_x = block.grid_x * BLOCK_PIXEL_SIZE
        local_y = block.grid_y * BLOCK_PIXEL_SIZE
        world_x = self.pos[0] + cos_a * local_x - sin_a * local_y
        world_y = self.pos[1] + sin_a * local_x + cos_a * local_y
        return np.array([world_x, world_y], dtype=np.float32)

    def get_block_world_angle(self, block: Block) -> float:
        """Get the world-space angle of a directional block."""
        return self.angle + block.direction.angle

    def think(self, sensor_inputs: np.ndarray):
        """Run brain and interpret outputs for engines and weapons."""
        if not self.alive:
            return

        outputs = self.brain.forward(sensor_inputs)
        # Outputs are mapped to: [engine0, engine1, ..., weapon0, weapon1, ...]
        # All in [-1, 1] from tanh

        idx = 0

        # Apply engine outputs
        accel_x, accel_y = 0.0, 0.0
        for block in self.blocks:
            if block.block_type != BlockType.ENGINE:
                continue
            if idx < len(outputs):
                thrust_signal = outputs[idx]
                idx += 1
                world_angle = self.get_block_world_angle(block)
                push_angle = world_angle + math.pi
                accel_x += math.cos(push_angle) * thrust_signal * block.thrust
                accel_y += math.sin(push_angle) * thrust_signal * block.thrust

        self.velocity[0] += accel_x
        self.velocity[1] += accel_y

        # Clamp speed
        speed = np.linalg.norm(self.velocity)
        max_speed = settings.ROBOT_DEFAULT_SPEED
        if speed > max_speed:
            self.velocity *= max_speed / speed

        # Body does NOT rotate with velocity — assembly layout stays fixed.
        # Engines always push in their assembly directions, weapons always
        # fire in their assembly directions.

        # Weapon outputs: iterate ALL weapon blocks to keep output index stable.
        self._weapon_signals = []
        for block in self.blocks:
            if block.block_type != BlockType.WEAPON:
                continue
            if idx < len(outputs):
                self._weapon_signals.append((block, outputs[idx] > 0.0))
                idx += 1

    def update(self):
        """Move robot, apply friction, tick block cooldowns."""
        if not self.alive:
            return

        old_pos = self.pos.copy()
        self.pos += self.velocity
        self.distance_traveled += np.linalg.norm(self.pos - old_pos)
        self.ticks_alive += 1

        # Friction
        self.velocity *= 0.95

        # Tick all blocks (cooldowns)
        for block in self.blocks:
            block.tick()

    def take_damage(self, amount: float):
        """Subtract damage from the robot's HP pool."""
        self.hp -= amount
        self.hits_taken += 1
        if self.hp <= 0:
            self.hp = 0
            self.alive = False

    def try_shoot(self) -> list[Bullet]:
        """Fire from all weapons that want to shoot. Returns list of bullets."""
        if not self.alive:
            return []

        bullets = []
        for block, wants_fire in getattr(self, '_weapon_signals', []):
            if not wants_fire:
                continue
            if block.cooldown > 0:
                continue

            block.cooldown = block.fire_rate
            aim = self.get_block_world_angle(block)
            direction = np.array([math.cos(aim), math.sin(aim)], dtype=np.float32)
            spawn_pos = self.get_block_world_pos(block) + direction * (BLOCK_PIXEL_SIZE + 2)
            bullets.append(Bullet(
                pos=spawn_pos,
                velocity=direction * block.bullet_speed,
                team=self.team,
                damage=block.bullet_damage,
                owner=self,
            ))
        return bullets


@dataclass
class Bullet:
    pos: np.ndarray
    velocity: np.ndarray
    team: int
    damage: float = 10.0
    alive: bool = True
    lifetime: int = settings.BULLET_LIFETIME
    owner: Robot | None = None

    @property
    def radius(self) -> float:
        return settings.BULLET_RADIUS

    def update(self):
        self.pos += self.velocity
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.alive = False


@dataclass
class Base:
    center: np.ndarray
    radius: float
    team: int
    wall_hp: float = settings.BASE_WALL_HP
    commander_alive: bool = True

    @property
    def wall_alive(self) -> bool:
        return self.wall_hp > 0

    def take_wall_damage(self, amount: float):
        self.wall_hp -= amount
        if self.wall_hp < 0:
            self.wall_hp = 0

    def commander_hit(self):
        self.commander_alive = False

    @staticmethod
    def create(team: int) -> Base:
        center = settings.BASE_POSITIONS[team].copy()
        return Base(center=center, radius=settings.BASE_RADIUS, team=team)
