"""Robot Evolution - main entry point and game loop."""

from __future__ import annotations

import math
import sys
from enum import Enum, auto

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
from assembly import AssemblyScreen
from training import TrainingManager


class Phase(Enum):
    ASSEMBLY = auto()
    MATCH = auto()


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("Robot Evolution")
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)

        # Start in assembly phase
        self.phase = Phase.ASSEMBLY
        self.assembly = AssemblyScreen(self.screen)
        self.assembly_done_timer = 0  # brief delay before starting match

        # Match state (initialized on transition)
        self.tick = 0
        self.winner: int | None = None
        self.bases: list[Base] = []
        self.robots: list[Robot] = []
        self.bullets: list[Bullet] = []
        self.player_blueprints: list[list[RobotBlueprint]] = [[], []]
        self.training: TrainingManager | None = None
        self.training_started = False  # True after 15s setup period
        self.frame_count = 0

    def run(self):
        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False

                if self.phase == Phase.ASSEMBLY:
                    self._update_assembly()
                    self.assembly.draw()
                else:
                    if self.winner is None:
                        self._update_match()
                    self._draw_match()

                pygame.display.flip()
                self.clock.tick(settings.FPS)
        finally:
            if self.training is not None:
                self.training.stop()
            pygame.quit()
            sys.exit()

    # --- assembly phase ----------------------------------------------------

    def _update_assembly(self):
        self.assembly.update()

        if self.assembly.both_ready:
            self.assembly_done_timer += 1
            if self.assembly_done_timer >= settings.FPS:  # 1 second delay
                self._start_match()
        else:
            self.assembly_done_timer = 0

    def _start_match(self):
        self.phase = Phase.MATCH
        self.player_blueprints = self.assembly.get_blueprints()
        self.tick = 0
        self.winner = None
        self.bases = [Base.create(0), Base.create(1)]
        self.robots = []
        self.bullets = []
        self._spawn_initial_robots()

        # Start training
        self.training = TrainingManager(self.player_blueprints)

    def _spawn_initial_robots(self):
        """Spawn one robot per blueprint for each player."""
        for team in (0, 1):
            base_pos = settings.BASE_POSITIONS[team]
            bps = self.player_blueprints[team]
            for i, bp in enumerate(bps):
                if not bp.blocks:
                    continue
                brain = Brain(
                    input_size=bp.brain_input_size,
                    hidden_size=bp.hidden_size,
                    output_size=bp.brain_output_size,
                )
                angle_offset = (2 * math.pi * i) / max(len(bps), 1)
                spawn_dist = settings.BASE_RADIUS + 50
                spawn_pos = base_pos + np.array([
                    math.cos(angle_offset) * spawn_dist,
                    math.sin(angle_offset) * spawn_dist,
                ], dtype=np.float32)
                face_angle = 0.0 if team == 0 else math.pi
                self.robots.append(Robot(
                    pos=spawn_pos, angle=face_angle, team=team,
                    blueprint=bp, brain=brain,
                ))

    # --- match phase -------------------------------------------------------

    def _update_match(self):
        self.tick += 1

        # Training zone management
        if self.training is not None:
            # Handle player input for training config
            keys = pygame.key.get_pressed()
            self.training.handle_input(keys)

            # 15-second setup period — resume training once elapsed
            if not self.training_started and self.tick >= settings.TRAINING_SETUP_TICKS:
                self.training_started = True
                self.training.resume_training()

            # Poll worker snapshots
            self.training.tick()

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

        # Robot-wall collisions
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
                continue
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

    def _draw_match(self):
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

        # Draw training zones
        if self.training is not None:
            setup_remaining = max(0, settings.TRAINING_SETUP_TICKS - self.tick)
            for player_id in range(2):
                self.renderer.draw_training_zone(
                    player_id,
                    self.training.get_zone(player_id),
                    self.training.get_ui(player_id),
                    setup_remaining=setup_remaining,
                )


if __name__ == "__main__":
    game = Game()
    game.run()
