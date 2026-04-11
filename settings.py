"""Game constants and configuration."""

import numpy as np

# --- Display ---
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
FPS = 60

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (30, 30, 30)
MID_GRAY = (80, 80, 80)
LIGHT_GRAY = (160, 160, 160)

RED = (220, 50, 50)
BLUE = (50, 100, 220)
GREEN = (50, 200, 80)
YELLOW = (230, 200, 40)
ORANGE = (230, 130, 40)
CYAN = (50, 200, 220)

TEAM_COLORS = {
    0: BLUE,   # Player 1
    1: RED,    # Player 2
}

# --- Arena / Battlefield ---
ARENA_WIDTH = 1400
ARENA_HEIGHT = 900

# --- Base ---
BASE_RADIUS = 80
BASE_WALL_HP = 200
BASE_WALL_THICKNESS = 6
COMMANDER_RADIUS = 6

# Base positions (center of each base circle)
BASE_POSITIONS = {
    0: np.array([120.0, ARENA_HEIGHT / 2]),   # Player 1 - left
    1: np.array([ARENA_WIDTH - 120.0, ARENA_HEIGHT / 2]),  # Player 2 - right
}

# --- Robot defaults ---
ROBOT_RADIUS = 10
ROBOT_DEFAULT_HP = 50
ROBOT_DEFAULT_SPEED = 2.0
ROBOT_DEFAULT_TURN_RATE = 0.1  # radians per tick

# --- Bullet ---
BULLET_RADIUS = 3
BULLET_SPEED = 6.0
BULLET_DAMAGE = 10
BULLET_LIFETIME = 120  # ticks before despawn

# --- Weapon ---
WEAPON_COOLDOWN = 30  # ticks between shots

# --- Turret ---
TURRET_RANGE = 250
TURRET_COOLDOWN = 45

# --- Sensors ---
VIEW_SENSOR_RAYS = 16       # number of rays for ViewSensor
VIEW_SENSOR_RANGE = 200.0   # max distance per ray
POINT_SENSOR_RANGE = 150.0  # detection range for PointSensor

# --- Neural net defaults ---
DEFAULT_HIDDEN_SIZE = 16

# --- Evolution ---
POPULATION_SIZE = 30
ELITE_COUNT = 5
MUTATION_RATE = 0.3
MUTATION_DECAY = 0.995  # rate decreases slightly each generation

# --- Training ---
TRAINING_TICKS_PER_GENERATION = 300  # how many sim ticks per generation eval
TRAINING_ARENA_WIDTH = 400
TRAINING_ARENA_HEIGHT = 300

# --- Auto-fork ---
AUTO_FORK_COOLDOWN = 300  # ticks between auto-forks
