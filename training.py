"""Training system — single zone per player with configurable sparring and fitness.

Each player has one training zone (full screen width strip). The zone trains
one robot design at a time — the player selects which of their 3 designs to
train and can hot-swap at any time. Populations are preserved per design.

Each zone runs in its own subprocess via multiprocessing. The main process
holds a lightweight proxy + UI state for input handling and rendering.
"""

from __future__ import annotations

import math
import time
import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np

from brain import Brain
from evolution import Population
from modules import RobotBlueprint, BlockType, BLOCK_PIXEL_SIZE
from entities import Robot, Bullet
from physics import clamp_to_arena
import settings


# --- Lightweight physics for small arenas (avoids NumPy overhead) -----------

def _simple_sensor_readings(robots: list[Robot]) -> list[np.ndarray | None]:
    """Simple O(N*S*N) sensor readings — faster than NumPy for N<30."""
    results: list[np.ndarray | None] = [None] * len(robots)
    for i, robot in enumerate(robots):
        if not robot.alive:
            continue
        sensor_blocks = [b for b in robot.blocks if b.block_type == BlockType.SENSOR]
        radar_blocks = [b for b in robot.blocks if b.block_type == BlockType.RADAR]
        if not sensor_blocks and not radar_blocks:
            results[i] = np.zeros(robot.blueprint.brain_input_size, dtype=np.float32)
            continue

        readings = []

        # --- Directional sensors ---
        for sensor in sensor_blocks:
            if not sensor.alive:
                readings.append(0.0)
                readings.append(0.0)
                continue
            s_pos = robot.get_block_world_pos(sensor)
            s_angle = robot.get_block_world_angle(sensor)
            cos_fov = math.cos(sensor.sensor_fov / 2)
            s_dx = math.cos(s_angle)
            s_dy = math.sin(s_angle)
            s_range = sensor.sensor_range

            best_dist = s_range
            best_type = 0.0

            for j, other in enumerate(robots):
                if j == i or not other.alive:
                    continue
                dx = other.pos[0] - s_pos[0]
                dy = other.pos[1] - s_pos[1]
                d = math.sqrt(dx * dx + dy * dy)
                if d >= s_range or d < 0.001:
                    continue
                dot = (dx * s_dx + dy * s_dy) / d
                if dot < cos_fov:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_type = 1.0 if other.team == robot.team else -1.0

            readings.append(best_dist / s_range)
            readings.append(best_type)

        # --- Radar blocks: Nth nearest enemy + Nth nearest friend ---
        if radar_blocks:
            enemies = []
            friends = []
            for j, other in enumerate(robots):
                if j == i or not other.alive:
                    continue
                dx = other.pos[0] - robot.pos[0]
                dy = other.pos[1] - robot.pos[1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < 0.001:
                    continue
                # Angle relative to robot's facing direction, normalized to [-1, 1]
                angle = math.atan2(dy, dx) - robot.angle
                # Wrap to [-pi, pi]
                angle = (angle + math.pi) % (2 * math.pi) - math.pi
                entry = (d, angle / math.pi)  # dist, normalized angle
                if other.team != robot.team:
                    enemies.append(entry)
                else:
                    friends.append(entry)
            enemies.sort()
            friends.sort()
            arena_diag = math.sqrt(800**2 + 400**2)  # normalization constant

            for r_idx, radar in enumerate(radar_blocks):
                if not radar.alive:
                    readings.extend([0.0, 0.0, 0.0, 0.0])
                    continue
                # Nth nearest enemy
                if r_idx < len(enemies):
                    e_dist, e_angle = enemies[r_idx]
                    readings.append(e_angle)                  # [-1, 1]
                    readings.append(1.0 - min(e_dist / arena_diag, 1.0))  # 1=close, 0=far
                else:
                    readings.extend([0.0, 0.0])
                # Nth nearest friend
                if r_idx < len(friends):
                    f_dist, f_angle = friends[r_idx]
                    readings.append(f_angle)
                    readings.append(1.0 - min(f_dist / arena_diag, 1.0))
                else:
                    readings.extend([0.0, 0.0])

        # --- Beacon: enemy base + friendly base ---
        has_beacon = any(
            b.block_type == BlockType.BEACON for b in robot.blocks
        )
        if has_beacon:
            # Virtual base positions: friendly at left, enemy at right (for team 0)
            friendly_base = np.array([0.0, 200.0], dtype=np.float32)
            enemy_base = np.array([800.0, 200.0], dtype=np.float32)
            if robot.team == 1:
                friendly_base, enemy_base = enemy_base, friendly_base
            for base_pos in (enemy_base, friendly_base):
                dx = base_pos[0] - robot.pos[0]
                dy = base_pos[1] - robot.pos[1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < 0.001:
                    readings.extend([0.0, 1.0])
                else:
                    angle = math.atan2(dy, dx) - robot.angle
                    angle = (angle + math.pi) % (2 * math.pi) - math.pi
                    readings.append(angle / math.pi)
                    readings.append(1.0 - min(d / arena_diag, 1.0))

        results[i] = np.array(readings, dtype=np.float32)
    return results


def _simple_robot_collisions(robots: list[Robot]):
    """Simple pairwise push — faster than NumPy for N<30."""
    n = len(robots)
    for i in range(n):
        if not robots[i].alive:
            continue
        for j in range(i + 1, n):
            if not robots[j].alive:
                continue
            dx = robots[i].pos[0] - robots[j].pos[0]
            dy = robots[i].pos[1] - robots[j].pos[1]
            dist_sq = dx * dx + dy * dy
            min_d = robots[i].radius + robots[j].radius
            if dist_sq >= min_d * min_d or dist_sq < 0.001:
                continue
            dist = math.sqrt(dist_sq)
            overlap = min_d - dist
            push = overlap * 0.5 / dist
            px, py = dx * push, dy * push
            robots[i].pos[0] += px
            robots[i].pos[1] += py
            robots[j].pos[0] -= px
            robots[j].pos[1] -= py


def _simple_bullet_collisions(bullets: list[Bullet], robots: list[Robot]):
    """Simple bullet-robot collision. Returns list of (bullet_idx, robot, bullet_pos)."""
    hits = []
    for bi, bullet in enumerate(bullets):
        if not bullet.alive:
            continue
        bx, by = bullet.pos[0], bullet.pos[1]
        for robot in robots:
            if not robot.alive or robot.team == bullet.team:
                continue
            dx = bx - robot.pos[0]
            dy = by - robot.pos[1]
            hit_r = settings.BULLET_RADIUS + robot.radius
            if dx * dx + dy * dy < hit_r * hit_r:
                hits.append((bi, robot, bullet.pos.copy()))
                break  # bullet hits first robot only
    return hits


# --- Fitness parameter definitions ------------------------------------------

FITNESS_PARAMS = [
    # (key, label, default, min_val, max_val, step)
    ('hit_enemy',     'Hit enemy',    50.0, -200.0, 200.0, 10.0),
    ('hit_friend',    'Hit friend',  -30.0, -200.0, 200.0, 10.0),
    ('survival',      'Survival',      0.1,   -2.0,   2.0,  0.1),
    ('damage_taken',  'Damage taken', -5.0,  -50.0,  50.0,  5.0),
    ('dist_to_enemy', 'Dist to enemy', 0.0,  -10.0,  10.0,  1.0),
    ('dist_to_friend','Dist to friend',0.0,  -10.0,  10.0,  1.0),
    ('dist_to_ebase', 'Dist to eBase', 0.0,  -10.0,  10.0,  1.0),
    ('dist_to_fbase', 'Dist to fBase', 0.0,  -10.0,  10.0,  1.0),
    ('collect',       'Collect res',   0.0, -100.0, 100.0, 10.0),
    ('scan_enemy',    'Scan enemy',    0.0, -200.0, 200.0, 10.0),
]

DEFAULT_FITNESS_WEIGHTS = {p[0]: p[2] for p in FITNESS_PARAMS}


# --- Training zone config ---------------------------------------------------

@dataclass
class TrainingZoneConfig:
    """Configuration for a training zone — sent to the worker process."""
    active_slot: int = 0
    enemy_slot: int = 0       # which of player's 3 designs for enemy sparring
    enemy_count: int = 3
    friend_slot: int = -1     # -1 = none, 0-2 = design index
    friend_count: int = 0
    resource_count: int = 0   # scattered resource drops in the arena for gatherer training
    spawn_distance: int = 2   # 0=close, 1=medium, 2=far (default: far, original behavior)
    fitness_weights: dict = field(default_factory=lambda: dict(DEFAULT_FITNESS_WEIGHTS))

    def to_dict(self) -> dict:
        return {
            'active_slot': self.active_slot,
            'enemy_slot': self.enemy_slot,
            'enemy_count': self.enemy_count,
            'friend_slot': self.friend_slot,
            'friend_count': self.friend_count,
            'resource_count': self.resource_count,
            'spawn_distance': self.spawn_distance,
            'fitness_weights': dict(self.fitness_weights),
        }

    @staticmethod
    def from_dict(d: dict) -> TrainingZoneConfig:
        return TrainingZoneConfig(
            active_slot=d['active_slot'],
            enemy_slot=d['enemy_slot'],
            enemy_count=d['enemy_count'],
            friend_slot=d['friend_slot'],
            friend_count=d['friend_count'],
            resource_count=d.get('resource_count', 0),
            spawn_distance=d.get('spawn_distance', 2),
            fitness_weights=dict(d['fitness_weights']),
        )


# ---------------------------------------------------------------------------
# TrainingArena — the actual simulation (runs inside worker processes)
# ---------------------------------------------------------------------------

@dataclass
class _ResourceDrop:
    """A collectible resource drop in the training arena."""
    pos: np.ndarray
    alive: bool = True


def _simple_gather_resources(robots: list[Robot], resources: list[_ResourceDrop]):
    """Magnetic gathering: resources fly toward nearby gatherer blocks, collected on contact."""
    for res in resources:
        if not res.alive:
            continue
        best_pull_robot = None
        best_pull_dist = float('inf')
        for robot in robots:
            if not robot.alive:
                continue
            gatherer_blocks = [b for b in robot.blocks if b.block_type == BlockType.GATHERER]
            if not gatherer_blocks:
                continue
            dx = robot.pos[0] - res.pos[0]
            dy = robot.pos[1] - res.pos[1]
            dist = math.sqrt(dx * dx + dy * dy)
            # Collect on contact
            if dist < robot.radius + settings.RESOURCE_RADIUS:
                robot.resources_collected += 1
                res.alive = False
                break
            # Pull toward nearest gatherer-equipped robot within range
            if dist < settings.GATHERER_RANGE * len(gatherer_blocks) and dist < best_pull_dist:
                best_pull_dist = dist
                best_pull_robot = robot
        else:
            # No contact collection happened — apply magnetic pull
            if best_pull_robot is not None and res.alive:
                dx = best_pull_robot.pos[0] - res.pos[0]
                dy = best_pull_robot.pos[1] - res.pos[1]
                dist = best_pull_dist
                pull = settings.GATHERER_PULL_SPEED / max(dist, 1.0)
                res.pos[0] += dx * pull
                res.pos[1] += dy * pull


def _simple_scan_enemies(robots: list[Robot]):
    """Check scanner blocks on each robot for enemy-team robots in range/cone.

    Only counts scans against enemies (different team).  Each scanner block
    can fire once per its scan_rate cooldown.  A successful scan increments
    the scanning robot's ``scans_enemy`` counter.
    """
    for robot in robots:
        if not robot.alive:
            continue
        scanner_blocks = [b for b in robot.blocks
                          if b.block_type == BlockType.SCANNER
                          and b.scan_cooldown <= 0]
        if not scanner_blocks:
            continue
        for scanner in scanner_blocks:
            s_pos = robot.get_block_world_pos(scanner)
            s_angle = robot.get_block_world_angle(scanner)
            cos_fov = math.cos(scanner.sensor_fov / 2)
            s_dx = math.cos(s_angle)
            s_dy = math.sin(s_angle)
            s_range = scanner.scan_range

            for other in robots:
                if not other.alive or other.team == robot.team:
                    continue
                dx = other.pos[0] - s_pos[0]
                dy = other.pos[1] - s_pos[1]
                d = math.sqrt(dx * dx + dy * dy)
                if d >= s_range or d < 0.001:
                    continue
                dot = (dx * s_dx + dy * s_dy) / d
                if dot < cos_fov:
                    continue
                # Successful scan
                robot.scans_enemy += 1
                scanner.scan_cooldown = scanner.scan_rate
                break  # one scan per scanner per cooldown


class TrainingArena:
    """Single training zone — trains one design at a time, holds populations for all 3."""

    def __init__(self, player_id: int, blueprints: list[RobotBlueprint],
                 config: TrainingZoneConfig):
        self.player_id = player_id
        self.blueprints = blueprints
        self.config = config
        self.width = settings.TRAINING_ZONE_SIM_WIDTH
        self.height = settings.TRAINING_ZONE_SIM_HEIGHT

        # Per-slot state
        self.populations: dict[int, Population] = {}
        self.best_brains: dict[int, Brain | None] = {}
        self.last_best_fitness: dict[int, float] = {}
        self.last_avg_fitness: dict[int, float] = {}

        # Initialize all valid slots
        for slot in range(3):
            if blueprints[slot].blocks:
                self._init_slot(slot)

        # Simulation state
        self.students: list[Robot] = []
        self.sparring: list[Robot] = []
        self.bullets: list[Bullet] = []
        self.resources: list[_ResourceDrop] = []
        self.gen_tick = 0
        self.paused = True  # starts paused (15s setup)

        if self.active_slot in self.populations:
            self._start_generation()

    @property
    def active_slot(self) -> int:
        return self.config.active_slot

    @property
    def blueprint(self) -> RobotBlueprint:
        return self.blueprints[self.active_slot]

    @property
    def population(self) -> Population:
        return self.populations[self.active_slot]

    @property
    def generation(self) -> int:
        if self.active_slot in self.populations:
            return self.population.generation
        return 0

    @property
    def all_robots(self) -> list[Robot]:
        return [r for r in self.students + self.sparring if r.alive]

    def _init_slot(self, slot: int):
        if slot in self.populations:
            return
        bp = self.blueprints[slot]
        self.populations[slot] = Population(
            size=settings.POPULATION_SIZE,
            input_size=bp.brain_input_size,
            hidden_size=bp.hidden_size,
            output_size=bp.brain_output_size,
            elite_count=settings.ELITE_COUNT,
            mutation_rate=settings.MUTATION_RATE,
            mutation_decay=settings.MUTATION_DECAY,
        )
        self.best_brains[slot] = None
        self.last_best_fitness[slot] = 0.0
        self.last_avg_fitness[slot] = 0.0

    def apply_config(self, new_config: TrainingZoneConfig):
        """Apply new config. If active slot changed, restart generation."""
        old_slot = self.config.active_slot
        self.config = new_config
        if new_config.active_slot != old_slot and new_config.active_slot in self.populations:
            self._start_generation()

    def _start_generation(self):
        self.gen_tick = 0
        self.bullets = []
        self.students = []
        self.sparring = []

        slot = self.active_slot
        if slot not in self.populations:
            return

        bp = self.blueprint
        pop = self.population
        n_students = min(settings.TRAINING_STUDENT_COUNT, pop.size)

        # Spawn students
        for i in range(n_students):
            pos = self._random_spawn_pos(team=0)
            robot = Robot(
                pos=pos, angle=0.0, team=0,
                blueprint=bp,
                brain=pop.brains[i].copy(),
            )
            self.students.append(robot)

        # Spawn enemy sparring partners
        enemy_slot = self.config.enemy_slot
        if 0 <= enemy_slot < 3 and self.blueprints[enemy_slot].blocks:
            enemy_bp = self.blueprints[enemy_slot]
            for _ in range(self.config.enemy_count):
                pos = self._random_spawn_pos(team=1)
                brain = self._get_sparring_brain(enemy_slot, enemy_bp)
                robot = Robot(
                    pos=pos, angle=math.pi, team=1,
                    blueprint=enemy_bp, brain=brain,
                )
                self.sparring.append(robot)

        # Spawn friend sparring partners
        friend_slot = self.config.friend_slot
        if 0 <= friend_slot < 3 and self.blueprints[friend_slot].blocks:
            friend_bp = self.blueprints[friend_slot]
            for _ in range(self.config.friend_count):
                pos = self._random_spawn_pos(team=0)
                brain = self._get_sparring_brain(friend_slot, friend_bp)
                robot = Robot(
                    pos=pos, angle=0.0, team=0,
                    blueprint=friend_bp, brain=brain,
                )
                self.sparring.append(robot)

        # Spawn resource drops (scattered randomly for gatherer training)
        self.resources = []
        margin = 40.0
        for _ in range(self.config.resource_count):
            pos = np.array([
                np.random.uniform(margin, self.width - margin),
                np.random.uniform(margin, self.height - margin),
            ], dtype=np.float32)
            self.resources.append(_ResourceDrop(pos=pos))

    def _get_sparring_brain(self, slot: int, bp: RobotBlueprint) -> Brain:
        best = self.best_brains.get(slot)
        if best is not None:
            return best.copy()
        return Brain(bp.brain_input_size, bp.hidden_size, bp.brain_output_size)

    def _random_spawn_pos(self, team: int) -> np.ndarray:
        margin = 30.0
        # spawn_distance: 0=close, 1=medium, 2=far
        sd = self.config.spawn_distance
        # Team 0 (students) on left, team 1 (enemies) on right
        # Ranges narrow as spawn_distance decreases
        if sd == 0:    # close: both teams near center
            t0_lo, t0_hi = 0.3, 0.45
            t1_lo, t1_hi = 0.55, 0.7
        elif sd == 1:  # medium
            t0_lo, t0_hi = 0.15, 0.4
            t1_lo, t1_hi = 0.6, 0.85
        else:          # far (original)
            t0_lo, t0_hi = 0.04, 0.4
            t1_lo, t1_hi = 0.6, 0.96
        if team == 0:
            x = np.random.uniform(self.width * t0_lo + margin, self.width * t0_hi)
        else:
            x = np.random.uniform(self.width * t1_lo, self.width * t1_hi - margin)
        y = np.random.uniform(margin, self.height - margin)
        return np.array([x, y], dtype=np.float32)

    def tick(self):
        if self.paused:
            return
        if self.active_slot not in self.populations:
            return

        self.gen_tick += 1

        alive_all = [r for r in self.students if r.alive] + \
                    [r for r in self.sparring if r.alive]

        if not alive_all:
            self._end_generation()
            return

        sensor_results = _simple_sensor_readings(alive_all)
        for i, robot in enumerate(alive_all):
            if sensor_results[i] is not None:
                robot.think(sensor_results[i])

        new_bullets = []
        for robot in alive_all:
            new_bullets.extend(robot.try_shoot())
        self.bullets.extend(new_bullets)

        w, h = self.width, self.height
        for robot in alive_all:
            robot.update()
            clamp_to_arena(robot.pos, robot.radius, w, h)

        _simple_robot_collisions(alive_all)

        # Accumulate distances for fitness
        alive_enemies = [r for r in self.sparring if r.alive and r.team != 0]
        alive_friends = [r for r in self.sparring if r.alive and r.team == 0]
        # Virtual base positions (team 0 students: friendly=left, enemy=right)
        enemy_base_pos = np.array([float(w), float(h) * 0.5], dtype=np.float32)
        friend_base_pos = np.array([0.0, float(h) * 0.5], dtype=np.float32)
        for s in self.students:
            if not s.alive:
                continue
            if alive_enemies:
                enemy_positions = np.array([e.pos for e in alive_enemies])
                dists = np.linalg.norm(enemy_positions - s.pos, axis=1)
                s.cum_dist_to_enemy += float(np.min(dists))
            if alive_friends:
                friend_positions = np.array([f.pos for f in alive_friends])
                dists = np.linalg.norm(friend_positions - s.pos, axis=1)
                s.cum_dist_to_friend += float(np.min(dists))
            s.cum_dist_to_ebase += float(np.linalg.norm(enemy_base_pos - s.pos))
            s.cum_dist_to_fbase += float(np.linalg.norm(friend_base_pos - s.pos))

        hits = _simple_bullet_collisions(self.bullets, alive_all)
        for bi, target, bpos in hits:
            shooter_team = self.bullets[bi].team
            target.take_damage(self.bullets[bi].damage)

            if shooter_team == 0 and target.team == 1:
                self._credit_hit(bpos, 'hit_enemy')
            elif shooter_team == 0 and target.team == 0:
                self._credit_hit(bpos, 'hit_friend')

            self.bullets[bi].alive = False

        self.bullets = [b for b in self.bullets if b.alive]

        # Scanner enemy detection
        _simple_scan_enemies(alive_all)

        # Gatherer resource collection
        if self.resources:
            _simple_gather_resources(alive_all, self.resources)
            self.resources = [r for r in self.resources if r.alive]

        all_students_dead = not any(s.alive for s in self.students)
        time_up = self.gen_tick >= settings.TRAINING_TICKS_PER_GENERATION

        if all_students_dead or time_up:
            self._end_generation()

    def _credit_hit(self, bullet_pos: np.ndarray, hit_type: str):
        best_student = None
        best_dist = float('inf')
        for s in self.students:
            if not s.alive:
                continue
            d = np.linalg.norm(s.pos - bullet_pos)
            if d < best_dist:
                best_dist = d
                best_student = s
        if best_student is not None:
            if hit_type == 'hit_enemy':
                best_student.hits_dealt += 1
            elif hit_type == 'hit_friend':
                best_student.hits_friend += 1

    def _end_generation(self):
        w = self.config.fitness_weights
        slot = self.active_slot

        for i, robot in enumerate(self.students):
            arena_diag = np.sqrt(self.width**2 + self.height**2)
            if robot.ticks_alive > 0:
                t = robot.ticks_alive
                # All inverted: 1.0 = close, 0.0 = far
                dist_enemy = 1.0 - min(robot.cum_dist_to_enemy / t / arena_diag, 1.0)
                dist_friend = 1.0 - min(robot.cum_dist_to_friend / t / arena_diag, 1.0)
                dist_ebase = 1.0 - min(robot.cum_dist_to_ebase / t / arena_diag, 1.0)
                dist_fbase = 1.0 - min(robot.cum_dist_to_fbase / t / arena_diag, 1.0)
            else:
                dist_enemy = dist_friend = dist_ebase = dist_fbase = 0.0
            fitness = (
                robot.hits_dealt * w.get('hit_enemy', 0)
                + robot.hits_friend * w.get('hit_friend', 0)
                + robot.ticks_alive * w.get('survival', 0)
                + robot.hits_taken * w.get('damage_taken', 0)
                + dist_enemy * w.get('dist_to_enemy', 0)
                + dist_friend * w.get('dist_to_friend', 0)
                + dist_ebase * w.get('dist_to_ebase', 0)
                + dist_fbase * w.get('dist_to_fbase', 0)
                + robot.resources_collected * w.get('collect', 0)
                + robot.scans_enemy * w.get('scan_enemy', 0)
            )
            self.population.set_fitness(i, fitness)

        self.last_best_fitness[slot] = float(np.max(self.population.fitness))
        self.last_avg_fitness[slot] = float(np.mean(self.population.fitness))
        self.best_brains[slot] = self.population.get_best()

        self.population.evolve()
        self._start_generation()

    def get_best_brain(self, slot: int | None = None) -> Brain:
        if slot is None:
            slot = self.active_slot
        best = self.best_brains.get(slot)
        if best is not None:
            return best.copy()
        bp = self.blueprints[slot]
        return Brain(bp.brain_input_size, bp.hidden_size, bp.brain_output_size)

    def get_stats(self) -> dict:
        slot = self.active_slot
        base = {
            'active_slot': slot,
            'slot_generations': {s: p.generation for s, p in self.populations.items()},
        }
        if slot not in self.populations:
            base.update({
                'generation': 0, 'best_fitness': 0.0, 'avg_fitness': 0.0,
                'gen_tick': 0, 'alive_students': 0, 'total_students': 0,
                'alive_sparring': 0,
            })
            return base
        base.update({
            'generation': self.population.generation,
            'best_fitness': self.last_best_fitness.get(slot, 0.0),
            'avg_fitness': self.last_avg_fitness.get(slot, 0.0),
            'gen_tick': self.gen_tick,
            'alive_students': sum(1 for s in self.students if s.alive),
            'total_students': len(self.students),
            'alive_sparring': sum(1 for s in self.sparring if s.alive),
        })
        return base


# ---------------------------------------------------------------------------
# Multiprocessing: worker function + snapshot packing
# ---------------------------------------------------------------------------

_SNAPSHOT_MIN_INTERVAL = 0.03  # ~30fps worth of render data


def _pack_robots(arena: TrainingArena) -> list[tuple]:
    result = []
    for r in arena.students + arena.sparring:
        blocks = [(b.grid_x, b.grid_y, b.block_type.value, True, 1.0, 1.0)
                  for b in r.blocks]
        result.append((r.pos[0], r.pos[1], r.angle, r.team, r.alive, blocks,
                        r.hp, r.max_hp))
    return result


def _pack_bullets(arena: TrainingArena) -> list[tuple]:
    return [(b.pos[0], b.pos[1], b.team) for b in arena.bullets if b.alive]


def _pack_resources(arena: TrainingArena) -> list[tuple]:
    return [(r.pos[0], r.pos[1]) for r in arena.resources if r.alive]


def _zone_worker(blueprint_dicts: list[dict], config_dict: dict, conn: mp.connection.Connection):
    """Training zone worker — runs in a subprocess.

    Commands from main process (received via conn):
        'stop'              — exit the worker loop
        'pause'             — pause simulation
        'resume'            — resume simulation
        ('config', dict)    — update training zone config
    """
    import traceback
    try:
        blueprints = [RobotBlueprint.from_dict(d) for d in blueprint_dicts]
        config = TrainingZoneConfig.from_dict(config_dict)
        arena = TrainingArena(config_dict['player_id'], blueprints, config)

        last_gen = -1
        last_snapshot_time = 0.0
        tick_interval = 1.0 / (settings.FPS * settings.TRAINING_TICKS_PER_FRAME)
        last_tick_time = time.monotonic()

        while True:
            # Check for commands (non-blocking)
            while conn.poll():
                cmd = conn.recv()
                if cmd == 'stop':
                    conn.close()
                    return
                elif cmd == 'pause':
                    arena.paused = True
                elif cmd == 'resume':
                    arena.paused = False
                elif isinstance(cmd, tuple) and cmd[0] == 'config':
                    new_config = TrainingZoneConfig.from_dict(cmd[1])
                    arena.apply_config(new_config)

            # Rate-limit to 3x game speed (FPS * TRAINING_TICKS_PER_FRAME ticks/sec)
            now = time.monotonic()
            sleep_time = tick_interval - (now - last_tick_time)
            if sleep_time > 0.001:
                time.sleep(sleep_time)
            last_tick_time = time.monotonic()

            # Run one tick
            arena.tick()

            # Decide whether to send a snapshot
            now = time.monotonic()
            gen_changed = arena.generation != last_gen
            render_due = (now - last_snapshot_time) >= _SNAPSHOT_MIN_INTERVAL

            if gen_changed or render_due:
                snapshot = {
                    'stats': arena.get_stats(),
                    'robots': _pack_robots(arena),
                    'bullets': _pack_bullets(arena),
                    'resources': _pack_resources(arena),
                }
                if gen_changed:
                    last_gen = arena.generation
                    slot = arena.active_slot
                    best = arena.best_brains.get(slot)
                    if best is not None:
                        snapshot['best_brain'] = {
                            'slot': slot,
                            'weights': best.get_flat_weights(),
                            'input_size': best.input_size,
                            'hidden_size': best.hidden_size,
                            'output_size': best.output_size,
                        }
                try:
                    conn.send(snapshot)
                except (BrokenPipeError, OSError):
                    return
                last_snapshot_time = now
    except Exception:
        traceback.print_exc()
        try:
            conn.send({'error': traceback.format_exc()})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Lightweight render proxy objects
# ---------------------------------------------------------------------------

@dataclass
class _RenderBlock:
    grid_x: int
    grid_y: int
    block_type: BlockType


@dataclass
class _RenderRobot:
    pos: np.ndarray
    angle: float
    team: int
    alive: bool
    blocks: list  # list[_RenderBlock]


@dataclass
class _RenderBullet:
    pos: np.ndarray
    team: int
    alive: bool = True


# ---------------------------------------------------------------------------
# TrainingZoneProxy — main-process stand-in for a worker zone
# ---------------------------------------------------------------------------

class TrainingZoneProxy:
    """Receives snapshots from a worker process and exposes rendering data."""

    def __init__(self, player_id: int, blueprints: list[RobotBlueprint],
                 conn: mp.connection.Connection):
        self.player_id = player_id
        self.blueprints = blueprints
        self.width = settings.TRAINING_ZONE_SIM_WIDTH
        self.height = settings.TRAINING_ZONE_SIM_HEIGHT
        self._conn = conn

        # Cached render state
        self._robots: list[_RenderRobot] = []
        self._bullets: list[_RenderBullet] = []
        self._resources: list[np.ndarray] = []  # list of [x, y] positions
        self._stats: dict = {
            'generation': 0, 'best_fitness': 0.0, 'avg_fitness': 0.0,
            'gen_tick': 0, 'alive_students': 0, 'total_students': 0,
            'alive_sparring': 0, 'active_slot': 0, 'slot_generations': {},
        }
        self._best_brains: dict[int, dict] = {}  # slot -> brain data

    @property
    def generation(self) -> int:
        return self._stats.get('generation', 0)

    @property
    def all_robots(self) -> list[_RenderRobot]:
        return [r for r in self._robots if r.alive]

    @property
    def bullets(self) -> list[_RenderBullet]:
        return self._bullets

    @property
    def resources(self) -> list[np.ndarray]:
        return self._resources

    def get_stats(self) -> dict:
        return self._stats

    def get_best_brain(self, slot: int | None = None) -> Brain:
        if slot is None:
            slot = self._stats.get('active_slot', 0)
        if slot in self._best_brains:
            d = self._best_brains[slot]
            brain = Brain(d['input_size'], d['hidden_size'], d['output_size'])
            brain.set_flat_weights(d['weights'].copy())
            return brain
        bp = self.blueprints[slot]
        return Brain(bp.brain_input_size, bp.hidden_size, bp.brain_output_size)

    def send_config(self, config: TrainingZoneConfig):
        try:
            self._conn.send(('config', config.to_dict()))
        except (BrokenPipeError, OSError):
            pass

    def send_command(self, cmd):
        try:
            self._conn.send(cmd)
        except (BrokenPipeError, OSError):
            pass

    def poll_updates(self):
        latest = None
        try:
            while self._conn.poll():
                latest = self._conn.recv()
        except (BrokenPipeError, EOFError, OSError):
            pass
        if latest is not None:
            self._apply_snapshot(latest)

    def _apply_snapshot(self, snapshot: dict):
        self._stats = snapshot['stats']

        if 'best_brain' in snapshot:
            bd = snapshot['best_brain']
            self._best_brains[bd['slot']] = bd

        self._robots = []
        for robot_data in snapshot['robots']:
            x, y, angle, team, alive, blocks_data = robot_data[:6]
            blocks = [
                _RenderBlock(gx, gy, BlockType(bt))
                for gx, gy, bt, *_ in blocks_data
            ]
            self._robots.append(_RenderRobot(
                pos=np.array([x, y], dtype=np.float32),
                angle=angle, team=team, alive=alive, blocks=blocks,
            ))

        self._bullets = []
        for (x, y, team) in snapshot['bullets']:
            self._bullets.append(_RenderBullet(
                pos=np.array([x, y], dtype=np.float32),
                team=team,
            ))

        self._resources = []
        for (x, y) in snapshot.get('resources', []):
            self._resources.append(np.array([x, y], dtype=np.float32))

    def stop(self):
        try:
            self._conn.send('stop')
        except (BrokenPipeError, OSError):
            pass


# ---------------------------------------------------------------------------
# TrainingZoneUI — per-player input handling for training zone config
# ---------------------------------------------------------------------------

class TrainingZoneUI:
    """Handles player input for training zone configuration during match."""

    # Column 0: config rows
    ROW_DESIGN = 0
    ROW_ENEMY_TYPE = 1
    ROW_ENEMY_COUNT = 2
    ROW_FRIEND_TYPE = 3
    ROW_FRIEND_COUNT = 4
    ROW_RESOURCES = 5
    ROW_SPAWN_DIST = 6
    NUM_CONFIG_ROWS = 7

    # Column 1: fitness rows (indexed from 0 within the column)
    NUM_FITNESS_ROWS = len(FITNESS_PARAMS)

    SPAWN_DIST_LABELS = ['Close', 'Medium', 'Far']

    # Labels for config column
    CONFIG_LABELS = [
        'Design',
        'Enemy type',
        'Enemy count',
        'Friend type',
        'Friend count',
        'Resources',
        'Spawn dist',
    ]

    # Labels for fitness column
    FITNESS_LABELS = [p[1] for p in FITNESS_PARAMS]

    def __init__(self, player_id: int, blueprints: list[RobotBlueprint]):
        self.player_id = player_id
        self.blueprints = blueprints
        self.cursor_col = 0   # 0=config, 1=fitness
        self.cursor_row = 0
        self.config = TrainingZoneConfig()
        self._config_dirty = True

        # Find first valid slot
        for i in range(3):
            if blueprints[i].blocks:
                self.config.active_slot = i
                self.config.enemy_slot = i
                break

        # Key repeat state
        self._held: dict[int, int] = {}
        self._repeat_delay = 18  # frames before repeat starts
        self._repeat_interval = 5  # frames between repeats

    def handle_input(self, keys_pressed) -> bool:
        """Process player input. Returns True if config changed."""
        pk = settings.PLAYER_KEYS[self.player_id]
        changed = False

        max_row = self.NUM_CONFIG_ROWS - 1 if self.cursor_col == 0 else self.NUM_FITNESS_ROWS - 1

        if self._key_event(keys_pressed, pk['up']):
            self.cursor_row = max(0, self.cursor_row - 1)
        if self._key_event(keys_pressed, pk['down']):
            self.cursor_row = min(max_row, self.cursor_row + 1)
        if self._key_event(keys_pressed, pk['left']):
            if self.cursor_col > 0:
                self.cursor_col = 0
                self.cursor_row = min(self.cursor_row, self.NUM_CONFIG_ROWS - 1)
        if self._key_event(keys_pressed, pk['right']):
            if self.cursor_col < 1:
                self.cursor_col = 1
                self.cursor_row = min(self.cursor_row, self.NUM_FITNESS_ROWS - 1)

        if self._key_event(keys_pressed, pk['primary']):
            changed = self._adjust(+1)
        if self._key_event(keys_pressed, pk['secondary']):
            changed = self._adjust(-1)

        if changed:
            self._config_dirty = True
        return changed

    def _key_event(self, keys_pressed, key: int) -> bool:
        if keys_pressed[key]:
            if key not in self._held:
                self._held[key] = 0
                return True
            self._held[key] += 1
            held = self._held[key]
            if held >= self._repeat_delay and (held - self._repeat_delay) % self._repeat_interval == 0:
                return True
            return False
        else:
            self._held.pop(key, None)
            return False

    def _adjust(self, direction: int) -> bool:
        row = self.cursor_row
        cfg = self.config

        if self.cursor_col == 1:
            # Fitness column
            if row < len(FITNESS_PARAMS):
                key, _label, default, lo, hi, step = FITNESS_PARAMS[row]
                old_val = cfg.fitness_weights.get(key, default)
                new_val = max(lo, min(hi, old_val + direction * step))
                if abs(new_val - old_val) > 1e-9:
                    cfg.fitness_weights[key] = round(new_val, 2)
                    return True
            return False

        # Config column
        if row == self.ROW_DESIGN:
            return self._cycle_slot('active_slot', direction)

        elif row == self.ROW_ENEMY_TYPE:
            return self._cycle_slot('enemy_slot', direction)

        elif row == self.ROW_ENEMY_COUNT:
            new = max(0, min(8, cfg.enemy_count + direction))
            if new != cfg.enemy_count:
                cfg.enemy_count = new
                return True

        elif row == self.ROW_FRIEND_TYPE:
            # Cycles through -1 (none), then valid slots
            options = [-1] + [i for i in range(3) if self.blueprints[i].blocks]
            cur_idx = options.index(cfg.friend_slot) if cfg.friend_slot in options else 0
            new_idx = (cur_idx + direction) % len(options)
            new_slot = options[new_idx]
            if new_slot != cfg.friend_slot:
                cfg.friend_slot = new_slot
                return True

        elif row == self.ROW_FRIEND_COUNT:
            new = max(0, min(8, cfg.friend_count + direction))
            if new != cfg.friend_count:
                cfg.friend_count = new
                return True

        elif row == self.ROW_RESOURCES:
            new = max(0, min(20, cfg.resource_count + direction))
            if new != cfg.resource_count:
                cfg.resource_count = new
                return True

        elif row == self.ROW_SPAWN_DIST:
            new = max(0, min(2, cfg.spawn_distance + direction))
            if new != cfg.spawn_distance:
                cfg.spawn_distance = new
                return True

        return False

    def _cycle_slot(self, attr: str, direction: int) -> bool:
        cur = getattr(self.config, attr)
        new_slot = (cur + direction) % 3
        for _ in range(3):
            if self.blueprints[new_slot].blocks:
                break
            new_slot = (new_slot + direction) % 3
        if new_slot != cur:
            setattr(self.config, attr, new_slot)
            return True
        return False

    def get_config_value_str(self, row: int) -> str:
        """Get display string for a config column row."""
        cfg = self.config
        if row == self.ROW_DESIGN:
            return f"Bot {cfg.active_slot + 1}"
        elif row == self.ROW_ENEMY_TYPE:
            return f"Bot {cfg.enemy_slot + 1}"
        elif row == self.ROW_ENEMY_COUNT:
            return str(cfg.enemy_count)
        elif row == self.ROW_FRIEND_TYPE:
            return "None" if cfg.friend_slot < 0 else f"Bot {cfg.friend_slot + 1}"
        elif row == self.ROW_FRIEND_COUNT:
            return str(cfg.friend_count)
        elif row == self.ROW_RESOURCES:
            return str(cfg.resource_count)
        elif row == self.ROW_SPAWN_DIST:
            return self.SPAWN_DIST_LABELS[cfg.spawn_distance]
        return ""

    def get_fitness_value_str(self, row: int) -> str:
        """Get display string for a fitness column row."""
        if row < len(FITNESS_PARAMS):
            key = FITNESS_PARAMS[row][0]
            val = self.config.fitness_weights.get(key, FITNESS_PARAMS[row][2])
            if abs(val) < 1.0 and val != 0:
                return f"{val:+.1f}"
            return f"{val:+.0f}"
        return ""

    def is_dirty(self) -> bool:
        dirty = self._config_dirty
        self._config_dirty = False
        return dirty


# ---------------------------------------------------------------------------
# TrainingManager — spawns and manages worker processes (1 per player)
# ---------------------------------------------------------------------------

class TrainingManager:
    """Manages one training zone worker per player."""

    def __init__(self, player_blueprints: list[list[RobotBlueprint]]):
        self.zones: list[TrainingZoneProxy] = []
        self.uis: list[TrainingZoneUI] = []
        self._workers: list[mp.Process] = []

        ctx = mp.get_context('spawn')

        for player_id in range(2):
            bps = player_blueprints[player_id]
            ui = TrainingZoneUI(player_id, bps)
            self.uis.append(ui)

            main_conn, worker_conn = ctx.Pipe()
            config_dict = ui.config.to_dict()
            config_dict['player_id'] = player_id

            p = ctx.Process(
                target=_zone_worker,
                args=([bp.to_dict() for bp in bps], config_dict, worker_conn),
                daemon=True,
            )
            p.start()
            worker_conn.close()

            proxy = TrainingZoneProxy(player_id, bps, main_conn)
            self.zones.append(proxy)
            self._workers.append(p)

    def tick(self):
        """Poll all workers for snapshot updates."""
        for zone in self.zones:
            zone.poll_updates()

    def handle_input(self, keys_pressed):
        """Process both players' training zone input."""
        for i, ui in enumerate(self.uis):
            ui.handle_input(keys_pressed)
            if ui.is_dirty():
                self.zones[i].send_config(ui.config)

    def resume_training(self):
        """Resume training on all zones (called after setup period)."""
        for zone in self.zones:
            zone.send_command('resume')

    def get_zone(self, player_id: int) -> TrainingZoneProxy:
        return self.zones[player_id]

    def get_ui(self, player_id: int) -> TrainingZoneUI:
        return self.uis[player_id]

    def stop(self):
        for zone in self.zones:
            zone.stop()
        for p in self._workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
