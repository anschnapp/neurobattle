"""Robot Evolution - main entry point and game loop."""

from __future__ import annotations

import math
import sys
import numpy as np
import pygame

import settings
from brain import Brain
from modules import RobotBlueprint
from entities import Robot, Bullet, Base
from physics import (
    clamp_to_arena, robot_blocked_by_wall,
    batch_robot_collisions, batch_bullet_robot_collisions,
    batch_bullet_base_collisions, batch_sensor_readings,
)
from renderer import Renderer


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("Robot Evolution")
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.tick = 0
        self.winner: int | None = None

        self.bases = [Base.create(0), Base.create(1)]
        self.robots: list[Robot] = []
        self.bullets: list[Bullet] = []

        self._spawn_demo_robots()

    def _spawn_demo_robots(self):
        """Spawn robots with different blueprints for visual testing."""
        blueprints = [
            RobotBlueprint.default_fighter(),
            RobotBlueprint.default_tank(),
            RobotBlueprint.minimal_scout(),
        ]

        for team in (0, 1):
            base_pos = settings.BASE_POSITIONS[team]
            for i in range(5):
                bp = blueprints[i % len(blueprints)]
                brain = Brain(
                    input_size=bp.brain_input_size,
                    hidden_size=bp.hidden_size,
                    output_size=bp.brain_output_size,
                )
                # Scatter around base, outside the wall
                angle_offset = (2 * math.pi * i) / 5
                spawn_dist = settings.BASE_RADIUS + 50
                spawn_pos = base_pos + np.array([
                    math.cos(angle_offset) * spawn_dist,
                    math.sin(angle_offset) * spawn_dist,
                ], dtype=np.float32)

                # Face toward enemy base
                face_angle = 0.0 if team == 0 else math.pi

                robot = Robot(
                    pos=spawn_pos,
                    angle=face_angle,
                    team=team,
                    blueprint=bp,
                    brain=brain,
                )
                self.robots.append(robot)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self._restart()
                    elif event.key == pygame.K_SPACE:
                        self._spawn_demo_robots()

            if self.winner is None:
                self._update()

            self._draw()
            self.clock.tick(settings.FPS)

        pygame.quit()
        sys.exit()

    def _restart(self):
        self.tick = 0
        self.winner = None
        self.bases = [Base.create(0), Base.create(1)]
        self.robots.clear()
        self.bullets.clear()
        self._spawn_demo_robots()

    def _update(self):
        self.tick += 1

        # Robot AI: batch sensor readings, then think
        sensor_map = batch_sensor_readings(self.robots, self.bases)
        for i, robot in enumerate(self.robots):
            if not robot.alive:
                continue
            if i in sensor_map:
                robot.think(sensor_map[i])

        # Robot shooting
        new_bullets = []
        for robot in self.robots:
            new_bullets.extend(robot.try_shoot())

        # Turret AI
        for base in self.bases:
            enemies = [r for r in self.robots if r.team != base.team and r.alive]
            for turret in base.turrets:
                turret.update_simple_ai(enemies)
                bullet = turret.try_shoot()
                if bullet is not None:
                    new_bullets.append(bullet)

        self.bullets.extend(new_bullets)

        # Update positions
        for robot in self.robots:
            robot.update()
            clamp_to_arena(robot.pos, robot.radius)

        for bullet in self.bullets:
            bullet.update()

        # Vectorized robot-robot collisions
        batch_robot_collisions(self.robots)

        # Robot-wall collisions (few bases, loop is fine)
        for robot in self.robots:
            if not robot.alive:
                continue
            for base in self.bases:
                robot_blocked_by_wall(robot, base)

        # Vectorized bullet-robot collisions
        hits = batch_bullet_robot_collisions(self.bullets, self.robots)
        hit_bullet_indices = set()
        for bi, ri, bpos in hits:
            if bi in hit_bullet_indices:
                continue  # bullet already consumed
            self.robots[ri].take_damage_at(bpos, self.bullets[bi].damage)
            self.bullets[bi].alive = False
            hit_bullet_indices.add(bi)

        # Vectorized bullet-base collisions
        base_hits = batch_bullet_base_collisions(self.bullets, self.bases)
        for bi, base_idx, hit_type in base_hits:
            if bi in hit_bullet_indices:
                continue
            if not self.bullets[bi].alive:
                continue
            if hit_type == 'wall':
                self.bases[base_idx].take_wall_damage(self.bullets[bi].damage)
            elif hit_type == 'commander':
                self.bases[base_idx].commander_hit()
                self.winner = 1 - self.bases[base_idx].team
            self.bullets[bi].alive = False
            hit_bullet_indices.add(bi)

        # Cleanup
        self.robots = [r for r in self.robots if r.alive]
        self.bullets = [b for b in self.bullets if b.alive]

    def _draw(self):
        self.renderer.clear()
        self.renderer.draw_arena_border()

        for base in self.bases:
            self.renderer.draw_base(base)

        for robot in self.robots:
            self.renderer.draw_robot(robot)

        for bullet in self.bullets:
            self.renderer.draw_bullet(bullet)

        self.renderer.draw_hud(self.bases, self.robots, self.tick)

        if self.winner is not None:
            self.renderer.draw_game_over(self.winner)

        pygame.display.flip()


if __name__ == "__main__":
    game = Game()
    game.run()
