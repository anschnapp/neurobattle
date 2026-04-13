"""Game constants and configuration."""

import numpy as np

# --- Display ---
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 1000
FPS = 60

# --- Screen layout (top training / center battlefield / bottom training) ---
TRAINING_STRIP_HEIGHT = 200
BATTLEFIELD_Y = TRAINING_STRIP_HEIGHT
BATTLEFIELD_HEIGHT = SCREEN_HEIGHT - 2 * TRAINING_STRIP_HEIGHT  # 600
BATTLEFIELD_WIDTH = SCREEN_WIDTH

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
ARENA_WIDTH = BATTLEFIELD_WIDTH
ARENA_HEIGHT = BATTLEFIELD_HEIGHT

# --- Base ---
BASE_RADIUS = 80
BASE_WALL_HP = 200
BASE_WALL_THICKNESS = 6
COMMANDER_RADIUS = 6

# Base positions (center of each base circle)
BASE_POSITIONS = {
    0: np.array([120.0, ARENA_HEIGHT / 2], dtype=np.float32),   # Player 1 - left
    1: np.array([ARENA_WIDTH - 120.0, ARENA_HEIGHT / 2], dtype=np.float32),  # Player 2 - right
}

# --- Robot defaults ---
ROBOT_RADIUS = 10
ROBOT_DEFAULT_HP = 50
ROBOT_DEFAULT_SPEED = 2.0
ROBOT_DEFAULT_TURN_RATE = 0.1  # radians per tick

# --- Collision ---
COLLISION_DAMAGE = 2  # HP lost per tick of overlap (both parties)

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
POPULATION_SIZE = 20
ELITE_COUNT = 3
MUTATION_RATE = 0.1

# --- Training ---
TRAINING_TICKS_PER_GENERATION = 300  # how many sim ticks per generation eval
TRAINING_ZONE_SIM_WIDTH = 800   # sim coords for single training zone
TRAINING_ZONE_SIM_HEIGHT = 400
TRAINING_STUDENT_COUNT = 20
TRAINING_TICKS_PER_FRAME = 3  # training runs faster than real-time
TRAINING_RENDER_INTERVAL = 3  # only redraw training strips every N frames
TRAINING_SETUP_TICKS = 15 * FPS  # 15 seconds setup before training starts
TRAINING_CONFIG_PANEL_WIDTH = 480  # pixels for the config panel in the strip

# --- Gatherer / Resources ---
GATHERER_RANGE = 80.0       # magnetic pull radius per gatherer block
GATHERER_PULL_SPEED = 2.0   # how fast resources fly toward gatherer
RESOURCE_RADIUS = 4
RESOURCE_VALUE = 1           # fitness credit per pickup

# --- Resource / Spawn ---
RESOURCE_INCOME_PER_SECOND = 5.0   # resources gained per second (passive)
SPAWN_COST_PER_BLOCK = 10          # resource cost per block in the blueprint
RESOURCE_RECLAIM_FRACTION = 0.3    # fraction of spawn cost reclaimed on destroy

# --- Controls ---
# Input is read via pygame.key.get_pressed() (polling) to avoid modifier interference.
import pygame
PLAYER_KEYS = [
    {  # Player 1
        'up': pygame.K_w,
        'down': pygame.K_s,
        'left': pygame.K_a,
        'right': pygame.K_d,
        'primary': pygame.K_e,
        'secondary': pygame.K_q,
    },
    {  # Player 2
        'up': pygame.K_UP,
        'down': pygame.K_DOWN,
        'left': pygame.K_LEFT,
        'right': pygame.K_RIGHT,
        'primary': pygame.K_RCTRL,
        'secondary': pygame.K_RSHIFT,
    },
]
