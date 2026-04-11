"""Training system — isolated arenas where robot brains evolve.

Each player has 3 training arenas (one per robot design).
Each arena runs in its own subprocess via multiprocessing:
  1. Spawn students + sparring partners
  2. Simulate for N ticks
  3. Evaluate fitness, evolve, repeat
  4. Periodically send snapshots back to the main process for rendering

The main process holds lightweight proxy objects that duck-type match
the TrainingArena interface so the renderer works unchanged.
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
        if not sensor_blocks:
            results[i] = np.zeros(robot.blueprint.brain_input_size, dtype=np.float32)
            continue

        readings = []
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


# Fitness weight presets
DEFAULT_FITNESS_WEIGHTS = {
    'hit_enemy': 50.0,
    'hit_friend': -30.0,
    'survival': 0.1,
    'damage_taken': -5.0,
}


# ---------------------------------------------------------------------------
# TrainingArena — the actual simulation (runs inside worker processes)
# ---------------------------------------------------------------------------

class TrainingArena:
    """An isolated arena where one robot design evolves."""

    def __init__(self, player_id: int, slot_index: int, blueprint: RobotBlueprint):
        self.player_id = player_id
        self.slot_index = slot_index
        self.blueprint = blueprint
        self.width = settings.TRAINING_ARENA_WIDTH
        self.height = settings.TRAINING_ARENA_HEIGHT

        # Evolution
        self.population = Population(
            size=settings.POPULATION_SIZE,
            input_size=blueprint.brain_input_size,
            hidden_size=blueprint.hidden_size,
            output_size=blueprint.brain_output_size,
            elite_count=settings.ELITE_COUNT,
            mutation_rate=settings.MUTATION_RATE,
            mutation_decay=settings.MUTATION_DECAY,
        )

        # Simulation state
        self.students: list[Robot] = []
        self.sparring: list[Robot] = []
        self.bullets: list[Bullet] = []
        self.gen_tick = 0
        self.paused = False

        # Fitness config
        self.fitness_weights = dict(DEFAULT_FITNESS_WEIGHTS)

        # Sparring config: how many copies of self to spar against
        self.sparring_count = 3

        # Best brain from latest generation (used for spawning on battlefield)
        self.best_brain: Brain | None = None
        # Stats from last completed generation (evolve() resets the population stats)
        self.last_best_fitness: float = 0.0
        self.last_avg_fitness: float = 0.0

        self._start_generation()

    @property
    def generation(self) -> int:
        return self.population.generation

    @property
    def all_robots(self) -> list[Robot]:
        """All alive robots in the arena (for simulation and rendering)."""
        return [r for r in self.students + self.sparring if r.alive]

    def _start_generation(self):
        """Spawn students and sparring partners for a new generation."""
        self.gen_tick = 0
        self.bullets = []
        self.students = []
        self.sparring = []

        n_students = min(settings.TRAINING_STUDENT_COUNT, self.population.size)

        # Spawn students — each gets a brain from the population
        for i in range(n_students):
            pos = self._random_spawn_pos(team=0)
            robot = Robot(
                pos=pos,
                angle=0.0,
                team=0,  # students are team 0 in the arena
                blueprint=self.blueprint,
                brain=self.population.brains[i].copy(),
            )
            self.students.append(robot)

        # Spawn sparring partners (copies of the same design with best brain or random)
        for i in range(self.sparring_count):
            pos = self._random_spawn_pos(team=1)
            if self.best_brain is not None:
                brain = self.best_brain.copy()
            else:
                brain = Brain(
                    self.blueprint.brain_input_size,
                    self.blueprint.hidden_size,
                    self.blueprint.brain_output_size,
                )
            robot = Robot(
                pos=pos,
                angle=math.pi,
                team=1,  # sparring partners are team 1
                blueprint=self.blueprint,
                brain=brain,
            )
            self.sparring.append(robot)

    def _random_spawn_pos(self, team: int) -> np.ndarray:
        """Random position in the arena, biased to left (team 0) or right (team 1)."""
        margin = 30.0
        if team == 0:
            x = np.random.uniform(margin, self.width * 0.4)
        else:
            x = np.random.uniform(self.width * 0.6, self.width - margin)
        y = np.random.uniform(margin, self.height - margin)
        return np.array([x, y], dtype=np.float32)

    def tick(self):
        """Run one simulation tick in this arena."""
        if self.paused:
            return

        self.gen_tick += 1

        # Build combined alive list
        alive_all = [r for r in self.students if r.alive] + \
                    [r for r in self.sparring if r.alive]

        if not alive_all:
            self._end_generation()
            return

        # Lightweight sensors (pure Python, no NumPy overhead)
        sensor_results = _simple_sensor_readings(alive_all)
        for i, robot in enumerate(alive_all):
            if sensor_results[i] is not None:
                robot.think(sensor_results[i])

        # Shooting
        new_bullets = []
        for robot in alive_all:
            new_bullets.extend(robot.try_shoot())
        self.bullets.extend(new_bullets)

        # Movement
        w, h = self.width, self.height
        for robot in alive_all:
            robot.update()
            clamp_to_arena(robot.pos, robot.radius, w, h)

        # Lightweight collisions
        _simple_robot_collisions(alive_all)

        # Lightweight bullet-robot collisions
        hits = _simple_bullet_collisions(self.bullets, alive_all)
        for bi, target, bpos in hits:
            shooter_team = self.bullets[bi].team
            target.take_damage_at(bpos, self.bullets[bi].damage)

            if shooter_team == 0 and target.team == 1:
                self._credit_hit(bpos, 'hit_enemy')
            elif shooter_team == 0 and target.team == 0:
                self._credit_hit(bpos, 'hit_friend')

            self.bullets[bi].alive = False

        # Cleanup dead bullets (keep dead robots for fitness tracking)
        self.bullets = [b for b in self.bullets if b.alive]

        # End generation?
        all_students_dead = not any(s.alive for s in self.students)
        time_up = self.gen_tick >= settings.TRAINING_TICKS_PER_GENERATION

        if all_students_dead or time_up:
            self._end_generation()

    def _credit_hit(self, bullet_pos: np.ndarray, hit_type: str):
        """Credit the nearest alive student for a hit."""
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
                best_student.hits_dealt -= 1  # negative to penalize

    def _end_generation(self):
        """Evaluate fitness, evolve, start next generation."""
        w = self.fitness_weights

        for i, robot in enumerate(self.students):
            fitness = (
                robot.hits_dealt * w['hit_enemy']
                + robot.ticks_alive * w['survival']
                + robot.hits_taken * w['damage_taken']
            )
            self.population.set_fitness(i, fitness)

        # Save stats before evolving (evolve resets fitness array)
        self.last_best_fitness = float(np.max(self.population.fitness))
        self.last_avg_fitness = float(np.mean(self.population.fitness))
        self.best_brain = self.population.get_best()

        # Evolve
        self.population.evolve()

        # Start next generation
        self._start_generation()

    def get_best_brain(self) -> Brain:
        """Get the best brain for spawning on the battlefield."""
        if self.best_brain is not None:
            return self.best_brain.copy()
        return self.population.get_best()

    def get_stats(self) -> dict:
        """Get training stats for UI display."""
        return {
            'generation': self.population.generation,
            'best_fitness': self.last_best_fitness,
            'avg_fitness': self.last_avg_fitness,
            'gen_tick': self.gen_tick,
            'alive_students': sum(1 for s in self.students if s.alive),
            'total_students': len(self.students),
            'alive_sparring': sum(1 for s in self.sparring if s.alive),
        }


# ---------------------------------------------------------------------------
# Multiprocessing: worker function + snapshot packing
# ---------------------------------------------------------------------------

# Minimum interval between render snapshots sent to main process (seconds).
# Generation-end snapshots (with best brain) are always sent immediately.
_SNAPSHOT_MIN_INTERVAL = 0.03  # ~30fps worth of render data


def _pack_robots(arena: TrainingArena) -> list[tuple]:
    """Pack robot state into picklable tuples."""
    result = []
    for r in arena.students + arena.sparring:
        blocks = [(b.grid_x, b.grid_y, b.block_type.value, b.alive, b.hp, b.max_hp)
                  for b in r.blocks]
        result.append((r.pos[0], r.pos[1], r.angle, r.team, r.alive, blocks))
    return result


def _pack_bullets(arena: TrainingArena) -> list[tuple]:
    """Pack bullet state into picklable tuples."""
    return [(b.pos[0], b.pos[1], b.team) for b in arena.bullets if b.alive]


def _arena_worker(blueprint_dict: dict, config: dict, conn: mp.connection.Connection):
    """Training arena worker — runs in a subprocess.

    Ticks the arena as fast as possible and sends snapshots back to the
    main process through ``conn`` (a multiprocessing Pipe endpoint).

    Commands from main process (received via conn):
        'stop'   — exit the worker loop
        'pause'  — pause simulation
        'resume' — resume simulation
    """
    import traceback
    try:
        blueprint = RobotBlueprint.from_dict(blueprint_dict)
        arena = TrainingArena(config['player_id'], config['slot_index'], blueprint)

        last_gen = -1
        last_snapshot_time = 0.0

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
                }
                if gen_changed:
                    last_gen = arena.generation
                    if arena.best_brain is not None:
                        snapshot['best_brain'] = {
                            'weights': arena.best_brain.get_flat_weights(),
                            'input_size': arena.best_brain.input_size,
                            'hidden_size': arena.best_brain.hidden_size,
                            'output_size': arena.best_brain.output_size,
                        }
                try:
                    conn.send(snapshot)
                except (BrokenPipeError, OSError):
                    return  # main process closed the connection
                last_snapshot_time = now
    except Exception:
        traceback.print_exc()
        try:
            conn.send({'error': traceback.format_exc()})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Lightweight render proxy objects (duck-type compatible with Robot/Bullet/Block)
# ---------------------------------------------------------------------------

@dataclass
class _RenderBlock:
    """Minimal block for rendering — matches the attrs the renderer reads."""
    grid_x: int
    grid_y: int
    block_type: BlockType
    alive: bool
    hp: float = 1.0
    max_hp: float = 1.0


@dataclass
class _RenderRobot:
    """Minimal robot for rendering — matches the attrs the renderer reads."""
    pos: np.ndarray
    angle: float
    team: int
    alive: bool
    blocks: list  # list[_RenderBlock]


@dataclass
class _RenderBullet:
    """Minimal bullet for rendering — matches the attrs the renderer reads."""
    pos: np.ndarray
    team: int
    alive: bool = True


# ---------------------------------------------------------------------------
# TrainingArenaProxy — main-process stand-in for a worker arena
# ---------------------------------------------------------------------------

class TrainingArenaProxy:
    """Receives snapshots from a worker process and exposes the same interface
    as TrainingArena so the renderer works unchanged."""

    def __init__(self, player_id: int, slot_index: int,
                 blueprint: RobotBlueprint, conn: mp.connection.Connection):
        self.player_id = player_id
        self.slot_index = slot_index
        self.blueprint = blueprint
        self.width = settings.TRAINING_ARENA_WIDTH
        self.height = settings.TRAINING_ARENA_HEIGHT
        self._conn = conn

        # Cached render state
        self._robots: list[_RenderRobot] = []
        self._bullets: list[_RenderBullet] = []
        self._stats: dict = {
            'generation': 0, 'best_fitness': 0.0, 'avg_fitness': 0.0,
            'gen_tick': 0, 'alive_students': 0, 'total_students': 0,
            'alive_sparring': 0,
        }
        self._best_brain_data: dict | None = None

    @property
    def generation(self) -> int:
        return self._stats.get('generation', 0)

    @property
    def all_robots(self) -> list[_RenderRobot]:
        return [r for r in self._robots if r.alive]

    @property
    def bullets(self) -> list[_RenderBullet]:
        return self._bullets

    def get_stats(self) -> dict:
        return self._stats

    def get_best_brain(self) -> Brain:
        """Reconstruct the best brain from cached worker data."""
        if self._best_brain_data is not None:
            d = self._best_brain_data
            brain = Brain(d['input_size'], d['hidden_size'], d['output_size'])
            brain.set_flat_weights(d['weights'].copy())
            return brain
        # Fallback: return a random brain matching the blueprint
        return Brain(
            self.blueprint.brain_input_size,
            self.blueprint.hidden_size,
            self.blueprint.brain_output_size,
        )

    def poll_updates(self):
        """Drain all pending snapshots from the worker, keep the latest."""
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
            self._best_brain_data = snapshot['best_brain']

        # Rebuild render robots
        self._robots = []
        for (x, y, angle, team, alive, blocks_data) in snapshot['robots']:
            blocks = [
                _RenderBlock(gx, gy, BlockType(bt), ba, hp, max_hp)
                for gx, gy, bt, ba, hp, max_hp in blocks_data
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

    def stop(self):
        """Tell the worker to exit."""
        try:
            self._conn.send('stop')
        except (BrokenPipeError, OSError):
            pass


# ---------------------------------------------------------------------------
# TrainingManager — spawns and manages worker processes
# ---------------------------------------------------------------------------

class TrainingManager:
    """Manages training arenas across worker processes for both players."""

    def __init__(self, player_blueprints: list[list[RobotBlueprint]]):
        self.arenas: list[list[TrainingArenaProxy | None]] = []
        self._workers: list[mp.Process] = []

        # Use 'spawn' context to avoid forking pygame/SDL state into workers,
        # which causes silent segfaults on Linux.
        ctx = mp.get_context('spawn')

        for player_id in range(2):
            player_arenas: list[TrainingArenaProxy | None] = []
            for slot in range(3):
                bp = player_blueprints[player_id][slot]
                if bp.blocks:
                    main_conn, worker_conn = ctx.Pipe()
                    config = {'player_id': player_id, 'slot_index': slot}
                    p = ctx.Process(
                        target=_arena_worker,
                        args=(bp.to_dict(), config, worker_conn),
                        daemon=True,
                    )
                    p.start()
                    # Main process doesn't use the worker's pipe end
                    worker_conn.close()

                    proxy = TrainingArenaProxy(player_id, slot, bp, main_conn)
                    player_arenas.append(proxy)
                    self._workers.append(p)
                else:
                    player_arenas.append(None)
            self.arenas.append(player_arenas)

    def tick(self, ticks_per_frame: int = None):
        """Poll all workers for snapshot updates.

        Workers run independently at full speed — this just collects their
        latest state for rendering and game logic.  The ``ticks_per_frame``
        argument is accepted for API compatibility but ignored.
        """
        for player_arenas in self.arenas:
            for arena in player_arenas:
                if arena is not None:
                    arena.poll_updates()

    def get_arena(self, player_id: int, slot: int) -> TrainingArenaProxy | None:
        return self.arenas[player_id][slot]

    def stop(self):
        """Shut down all worker processes."""
        for player_arenas in self.arenas:
            for arena in player_arenas:
                if arena is not None:
                    arena.stop()
        for p in self._workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
