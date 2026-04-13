"""Physics: movement, collision detection, sensors — vectorized with NumPy."""

from __future__ import annotations

import math
import numpy as np

from entities import Robot, Bullet, Base
from modules import BlockType, BLOCK_PIXEL_SIZE
import settings


# ---- Scalar helpers (still used for single-entity checks) ----

def clamp_to_arena(pos: np.ndarray, radius: float,
                   width: float = settings.ARENA_WIDTH,
                   height: float = settings.ARENA_HEIGHT):
    pos[0] = np.clip(pos[0], radius, width - radius)
    pos[1] = np.clip(pos[1], radius, height - radius)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


# ---- Vectorized batch operations ----

def batch_clamp_to_arena(positions: np.ndarray, radii: np.ndarray,
                         width: float = settings.ARENA_WIDTH,
                         height: float = settings.ARENA_HEIGHT):
    """Clamp N positions in-place. positions: (N,2), radii: (N,)"""
    np.clip(positions[:, 0], radii, width - radii, out=positions[:, 0])
    np.clip(positions[:, 1], radii, height - radii, out=positions[:, 1])


def batch_robot_collisions(robots: list[Robot]):
    """Vectorized robot-robot push using distance matrix."""
    alive = [r for r in robots if r.alive]
    n = len(alive)
    if n < 2:
        return

    # Build position and radius arrays
    positions = np.array([r.pos for r in alive], dtype=np.float32)  # (N, 2)
    radii = np.array([r.radius for r in alive], dtype=np.float32)   # (N,)

    # Pairwise differences: diff[i,j] = pos[i] - pos[j]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N, N, 2)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq, np.inf)  # ignore self

    # Min distances
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]  # (N, N)
    min_dist_sq = min_dist * min_dist

    # Find overlapping pairs (upper triangle only to avoid double-counting)
    overlap_mask = (dist_sq < min_dist_sq) & (dist_sq > 0.001)
    # Zero out lower triangle
    overlap_mask = np.triu(overlap_mask, k=1)

    # Process overlapping pairs
    pairs_i, pairs_j = np.where(overlap_mask)
    if len(pairs_i) == 0:
        return

    for idx in range(len(pairs_i)):
        i, j = pairs_i[idx], pairs_j[idx]
        d = diff[i, j]
        dist = math.sqrt(dist_sq[i, j])
        overlap = min_dist[i, j] - dist
        push = d * (overlap * 0.5 / dist)
        alive[i].pos += push
        alive[j].pos -= push


def batch_bullet_robot_collisions(bullets: list[Bullet], robots: list[Robot]):
    """Vectorized bullet-robot collision check.

    Returns list of (bullet_index, robot_index, bullet_pos) for hits.
    """
    alive_bullets = [(i, b) for i, b in enumerate(bullets) if b.alive]
    alive_robots = [(i, r) for i, r in enumerate(robots) if r.alive]

    if not alive_bullets or not alive_robots:
        return []

    nb = len(alive_bullets)
    nr = len(alive_robots)

    bullet_pos = np.array([b.pos for _, b in alive_bullets], dtype=np.float32)  # (nb, 2)
    bullet_teams = np.array([b.team for _, b in alive_bullets], dtype=np.int32)  # (nb,)
    robot_pos = np.array([r.pos for _, r in alive_robots], dtype=np.float32)    # (nr, 2)
    robot_teams = np.array([r.team for _, r in alive_robots], dtype=np.int32)   # (nr,)
    robot_radii = np.array([r.radius for _, r in alive_robots], dtype=np.float32)  # (nr,)

    # Pairwise distances: (nb, nr)
    diff = bullet_pos[:, np.newaxis, :] - robot_pos[np.newaxis, :, :]  # (nb, nr, 2)
    dist_sq = np.sum(diff * diff, axis=2)  # (nb, nr)

    # Hit radius: bullet radius + robot radius (approximate — uses bounding radius)
    hit_r = settings.BULLET_RADIUS + robot_radii  # (nr,)
    hit_r_sq = hit_r * hit_r  # (nr,)

    # Collision mask
    collision = dist_sq < hit_r_sq[np.newaxis, :]  # (nb, nr)

    # No friendly fire
    same_team = bullet_teams[:, np.newaxis] == robot_teams[np.newaxis, :]  # (nb, nr)
    collision &= ~same_team

    hits = []
    # For each bullet, find first robot hit
    for bi in range(nb):
        hit_robots = np.where(collision[bi])[0]
        if len(hit_robots) > 0:
            # Pick closest
            closest = hit_robots[np.argmin(dist_sq[bi, hit_robots])]
            orig_bi = alive_bullets[bi][0]
            orig_ri = alive_robots[closest][0]
            hits.append((orig_bi, orig_ri, bullets[orig_bi].pos.copy()))

    return hits


def batch_bullet_base_collisions(bullets: list[Bullet], bases: list[Base]):
    """Vectorized bullet-base collision. Returns (bullet_idx, base_idx, hit_type) tuples.
    hit_type: 'wall' or 'commander'.
    """
    alive_bullets = [(i, b) for i, b in enumerate(bullets) if b.alive]
    if not alive_bullets:
        return []

    nb = len(alive_bullets)
    bullet_pos = np.array([b.pos for _, b in alive_bullets], dtype=np.float32)
    bullet_teams = np.array([b.team for _, b in alive_bullets], dtype=np.int32)

    hits = []
    for base_idx, base in enumerate(bases):
        # Distances from all bullets to base center
        diff = bullet_pos - base.center[np.newaxis, :]  # (nb, 2)
        dist = np.sqrt(np.sum(diff * diff, axis=1))  # (nb,)

        # Skip same-team bullets
        enemy_mask = bullet_teams != base.team

        if base.wall_alive:
            wall_inner = base.radius - settings.BASE_WALL_THICKNESS
            wall_outer = base.radius + settings.BASE_WALL_THICKNESS
            wall_hit = enemy_mask & (dist >= wall_inner) & (dist <= wall_outer)
            for bi in np.where(wall_hit)[0]:
                hits.append((alive_bullets[bi][0], base_idx, 'wall'))
        elif base.commander_alive:
            cmd_hit = enemy_mask & (dist < settings.COMMANDER_RADIUS + settings.BULLET_RADIUS)
            for bi in np.where(cmd_hit)[0]:
                hits.append((alive_bullets[bi][0], base_idx, 'commander'))

    return hits


def robot_blocked_by_wall(robot: Robot, base: Base) -> bool:
    if not base.wall_alive:
        return False
    dist = distance(robot.pos, base.center)
    wall_outer = base.radius + settings.BASE_WALL_THICKNESS + robot.radius
    wall_inner = base.radius - settings.BASE_WALL_THICKNESS - robot.radius
    if wall_inner < dist < wall_outer:
        diff = robot.pos - base.center
        norm = diff / (np.linalg.norm(diff) + 0.001)
        robot.pos = base.center + norm * wall_outer
        return True
    return False


# ---- Vectorized sensor readings ----

def batch_sensor_readings(robots: list[Robot], bases: list[Base]) -> dict[int, np.ndarray]:
    """Compute sensor readings for all robots at once.

    Returns dict mapping robot list index -> sensor input array.
    Uses vectorized distance computation for the robot-to-robot checks.
    """
    alive = [(i, r) for i, r in enumerate(robots) if r.alive]
    if not alive:
        return {}

    n = len(alive)
    # Precompute all robot positions and teams
    all_pos = np.array([r.pos for _, r in alive], dtype=np.float32)  # (N, 2)
    all_teams = np.array([r.team for _, r in alive], dtype=np.int32)  # (N,)

    # Pairwise distance matrix (N, N)
    diff = all_pos[:, np.newaxis, :] - all_pos[np.newaxis, :, :]  # (N, N, 2)
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))  # (N, N)
    np.fill_diagonal(dist_matrix, np.inf)

    # Normalized direction vectors (N, N, 2) — avoid div by zero
    safe_dist = np.where(dist_matrix[:, :, np.newaxis] > 0.001,
                         dist_matrix[:, :, np.newaxis], 1.0)
    dir_matrix = diff / safe_dist  # (N, N, 2) — direction FROM other TO self... wait
    # diff[i,j] = pos[i] - pos[j], so dir from j to i
    # We want dir from self to other: -diff / dist = (pos[j] - pos[i]) / dist
    # Actually for sensor we need to_other = other.pos - sensor_pos
    # diff[i,j] = pos[i] - pos[j], so to_other from i's perspective toward j = -diff[i,j]
    to_other_dir = -diff / safe_dist  # (N, N, 2)

    # Base positions for sensor checks
    base_positions = np.array([b.center for b in bases], dtype=np.float32)  # (B, 2)
    base_teams = np.array([b.team for b in bases], dtype=np.int32)

    # Precompute arena diagonal for radar normalization
    arena_diag = math.sqrt(settings.SCREEN_WIDTH**2 + settings.SCREEN_HEIGHT**2)

    results = {}
    for ai, (orig_idx, robot) in enumerate(alive):
        sensor_blocks = [b for b in robot.blocks if b.block_type == BlockType.SENSOR]
        radar_blocks = [b for b in robot.blocks if b.block_type == BlockType.RADAR]

        readings = []

        # --- Directional sensors ---
        for sensor in sensor_blocks:
            if not sensor.alive:
                readings.extend([0.0, 0.0])
                continue

            sensor_pos = robot.get_block_world_pos(sensor)
            sensor_angle = robot.get_block_world_angle(sensor)
            cos_fov = math.cos(sensor.sensor_fov / 2)
            sensor_dir = np.array([math.cos(sensor_angle), math.sin(sensor_angle)],
                                  dtype=np.float32)

            best_dist = sensor.sensor_range
            best_type = 0.0

            # Use precomputed distances for robot-robot checks
            dists = dist_matrix[ai]  # (N,) distances from this robot to all others
            in_range = dists < sensor.sensor_range

            if np.any(in_range):
                dots = to_other_dir[ai, :, 0] * sensor_dir[0] + to_other_dir[ai, :, 1] * sensor_dir[1]
                in_fov = dots >= cos_fov
                candidates = in_range & in_fov

                if np.any(candidates):
                    cand_indices = np.where(candidates)[0]
                    cand_dists = dists[cand_indices]
                    nearest_idx = cand_indices[np.argmin(cand_dists)]
                    nearest_dist = dists[nearest_idx]
                    if nearest_dist < best_dist:
                        best_dist = nearest_dist
                        best_type = 1.0 if all_teams[nearest_idx] == robot.team else -1.0

            # Check bases (few, so loop is fine)
            for bi, base in enumerate(bases):
                to_base = base.center - sensor_pos
                bd = np.linalg.norm(to_base)
                if bd > sensor.sensor_range or bd < 0.001:
                    continue
                dot = np.dot(to_base / bd, sensor_dir)
                if dot < cos_fov:
                    continue
                if bd < best_dist:
                    best_dist = bd
                    best_type = 1.0 if base.team == robot.team else -1.0

            readings.append(best_dist / sensor.sensor_range)
            readings.append(best_type)

        # --- Radar blocks: Nth nearest enemy + Nth nearest friend ---
        if radar_blocks:
            enemies = []
            friends = []
            dists = dist_matrix[ai]
            for aj, (_, other) in enumerate(alive):
                if aj == ai:
                    continue
                d = dists[aj]
                if d == np.inf or d < 0.001:
                    continue
                dx = other.pos[0] - robot.pos[0]
                dy = other.pos[1] - robot.pos[1]
                angle = math.atan2(dy, dx) - robot.angle
                angle = (angle + math.pi) % (2 * math.pi) - math.pi
                entry = (float(d), angle / math.pi)
                if other.team != robot.team:
                    enemies.append(entry)
                else:
                    friends.append(entry)
            enemies.sort()
            friends.sort()

            for r_idx, radar in enumerate(radar_blocks):
                if not radar.alive:
                    readings.extend([0.0, 0.0, 0.0, 0.0])
                    continue
                if r_idx < len(enemies):
                    e_dist, e_angle = enemies[r_idx]
                    readings.append(e_angle)
                    readings.append(1.0 - min(e_dist / arena_diag, 1.0))
                else:
                    readings.extend([0.0, 0.0])
                if r_idx < len(friends):
                    f_dist, f_angle = friends[r_idx]
                    readings.append(f_angle)
                    readings.append(1.0 - min(f_dist / arena_diag, 1.0))
                else:
                    readings.extend([0.0, 0.0])

        # --- Beacon: enemy base + friendly base (order must match training) ---
        has_beacon = any(
            b.block_type == BlockType.BEACON for b in robot.blocks
        )
        if has_beacon:
            # Sort: enemy base first, then friendly base
            sorted_bases = sorted(bases, key=lambda b: b.team == robot.team)
            for base in sorted_bases:
                bx = base.center[0] - robot.pos[0]
                by = base.center[1] - robot.pos[1]
                d = math.sqrt(bx * bx + by * by)
                if d < 0.001:
                    readings.extend([0.0, 1.0])
                else:
                    angle = math.atan2(by, bx) - robot.angle
                    angle = (angle + math.pi) % (2 * math.pi) - math.pi
                    readings.append(angle / math.pi)
                    readings.append(1.0 - min(d / arena_diag, 1.0))

        # --- Intrinsic inputs: speed, health (always present) ---
        speed = float(np.linalg.norm(robot.velocity))
        readings.append(speed / settings.ROBOT_DEFAULT_SPEED)  # 0..1
        readings.append(robot.hp / robot.max_hp if robot.max_hp > 0 else 0.0)  # 0..1

        results[orig_idx] = np.array(readings, dtype=np.float32)

    return results
