"""Game entities: robots (block-based), bullets, bases, turrets."""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field

from brain import Brain
from modules import RobotBlueprint, Block, BlockType, Direction, BLOCK_PIXEL_SIZE

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
    velocity: np.ndarray = field(default=None)
    # Fitness tracking
    hits_dealt: int = 0
    hits_friend: int = 0
    hits_taken: int = 0
    ticks_alive: int = 0
    distance_traveled: float = 0.0
    resources_collected: int = 0

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(2, dtype=np.float32)
        if not self.blocks:
            self.blocks = self.blueprint.copy_blocks()

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

    @property
    def hp(self) -> float:
        return sum(b.hp for b in self.blocks if b.alive)

    @property
    def max_hp(self) -> float:
        return sum(b.max_hp for b in self.blocks)

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

        # Apply engine outputs: iterate ALL engine blocks to keep output index stable.
        # Dead engines consume their slot but produce no thrust.
        accel_x, accel_y = 0.0, 0.0
        for block in self.blocks:
            if block.block_type != BlockType.ENGINE:
                continue
            if idx < len(outputs):
                thrust_signal = outputs[idx]
                idx += 1
                if not block.alive:
                    continue  # slot consumed, no effect
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

        # Update facing angle based on velocity
        if speed > 0.1:
            self.angle = math.atan2(self.velocity[1], self.velocity[0])

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

        # Tick all blocks
        for block in self.blocks:
            block.tick()

        # Check if robot is dead (all blocks destroyed)
        if not any(b.alive for b in self.blocks):
            self.alive = False

    def take_damage_at(self, world_pos: np.ndarray, amount: float):
        """Damage the block closest to world_pos."""
        best_block = None
        best_dist = float('inf')
        for block in self.blocks:
            if not block.alive:
                continue
            bpos = self.get_block_world_pos(block)
            dist = np.linalg.norm(bpos - world_pos)
            if dist < best_dist:
                best_dist = dist
                best_block = block

        if best_block is not None:
            best_block.take_damage(amount)
            self.hits_taken += 1
            if not any(b.alive for b in self.blocks):
                self.alive = False

    def try_shoot(self) -> list[Bullet]:
        """Fire from all weapons that want to shoot. Returns list of bullets."""
        if not self.alive:
            return []

        bullets = []
        for block, wants_fire in getattr(self, '_weapon_signals', []):
            if not wants_fire or not block.alive:
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

    @property
    def radius(self) -> float:
        return settings.BULLET_RADIUS

    def update(self):
        self.pos += self.velocity
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.alive = False


@dataclass
class Turret:
    """A turret mounted on a base wall."""
    base_center: np.ndarray
    base_radius: float
    wall_angle: float
    team: int
    brain: Brain | None = None
    fire_rate: int = settings.TURRET_COOLDOWN
    cooldown: int = 0
    bullet_speed: float = settings.BULLET_SPEED
    bullet_damage: float = settings.BULLET_DAMAGE
    target_angle: float = 0.0
    wants_to_shoot: bool = False

    @property
    def pos(self) -> np.ndarray:
        return self.base_center + np.array([
            math.cos(self.wall_angle) * self.base_radius,
            math.sin(self.wall_angle) * self.base_radius,
        ], dtype=np.float32)

    def update_simple_ai(self, enemies: list[Robot]):
        if self.cooldown > 0:
            self.cooldown -= 1
        if not enemies:
            self.wants_to_shoot = False
            return

        my_pos = self.pos
        nearest = None
        nearest_dist = float('inf')
        for e in enemies:
            if not e.alive:
                continue
            dist = np.linalg.norm(e.pos - my_pos)
            if dist < nearest_dist and dist < settings.TURRET_RANGE:
                nearest = e
                nearest_dist = dist

        if nearest is not None:
            self.target_angle = math.atan2(
                nearest.pos[1] - my_pos[1],
                nearest.pos[0] - my_pos[0],
            )
            self.wants_to_shoot = True
        else:
            self.wants_to_shoot = False

    def try_shoot(self) -> Bullet | None:
        if not self.wants_to_shoot or self.cooldown > 0:
            return None

        self.cooldown = self.fire_rate
        direction = np.array([math.cos(self.target_angle),
                              math.sin(self.target_angle)], dtype=np.float32)
        bullet_pos = self.pos + direction * (settings.BULLET_RADIUS + 2)
        return Bullet(
            pos=bullet_pos,
            velocity=direction * self.bullet_speed,
            team=self.team,
            damage=self.bullet_damage,
        )


@dataclass
class Base:
    center: np.ndarray
    radius: float
    team: int
    wall_hp: float = settings.BASE_WALL_HP
    commander_alive: bool = True
    turrets: list[Turret] = field(default_factory=list)

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
        base = Base(center=center, radius=settings.BASE_RADIUS, team=team)
        turret = Turret(
            base_center=center,
            base_radius=settings.BASE_RADIUS,
            wall_angle=math.pi if team == 0 else 0.0,
            team=team,
        )
        base.turrets.append(turret)
        return base
