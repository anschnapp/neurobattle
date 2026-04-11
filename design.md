# Robot Evolution - Game Design

## Concept

A 2-player competitive game where players don't fight directly. Instead, they design robot bodies, train robot brains through neural network evolution, and deploy them onto a shared battlefield. The core loop is: **build the body, evolve the brain, watch them fight, adapt.**

## Game Phases

### Phase 1: Pre-Game — Robot Design

Both players design **3 robot blueprints** from modular blocks. There is no time limit — the game starts when both players press "ready."

- Designs are **locked** after pre-game. Players cannot redesign robots mid-match.
- The only way to improve a design during the match is through training (better brains).
- More blocks = higher spawn cost + naturally harder/slower to train.

### Phase 2: Match — Train, Spawn, Fight

Once both players are ready, the match begins. Training arenas run continuously. Resources tick in. Robots auto-spawn and fight autonomously. The player's job is to manage training, allocate resources, scan enemies, and evolve counters.

## Players and Bases

- Two players, each with a static base on opposite sides of the screen (left and right).
- A base is a circular wall with a **commander** (red point) at the center.
- The commander is fragile: one hit and the player loses.
- The wall must be breached before the commander can be hit.
- Each base starts with one **turret** mounted on the circle wall. The turret rotates around the wall automatically and shoots at incoming threats.
- Turrets are trained using the same neural network system as robots.
- Players can potentially earn additional or improved turrets over time.
- Robots spawn **outside** the wall on the side facing the enemy base (left player: right of wall, right player: left of wall).

## Screen Layout

- **Top:** Player 1's 3 training arenas
- **Center:** Battlefield with bases on left and right
- **Bottom:** Player 2's 3 training arenas

Training arenas are rendered at the same scale as the battlefield. Each arena has slider controls for configuration (fitness weights, sparring partner counts, etc.).

## Robot Design (Modular)

Players assemble robots from small modular blocks. Each block adds to the robot's cost and complexity.

### Module Types

- **Armor / Body** — structural blocks, determines durability
- **Engine** — movement speed and maneuverability. No engine = can't move.
- **Weapon** — shooting device
- **Sensor** — perception modules (see Sensor Types below). No sensor = blind.
- **Scanner** — used to scan enemy robots (see Scanning below)
- **Gatherer** — magnetically collects resources from the battlefield in an area around the robot (dropped resources fly toward the gatherer like a magnet)

### Design Constraints

- More blocks = higher resource cost per spawn
- More blocks = more neural network inputs/outputs = harder to train effectively
- Players must balance capability against cost and trainability
- 3 designs total — forces strategic specialization (e.g., fighter, gatherer, scout)

### Sensor Types

Sensors are physical modules that determine what the robot's brain receives as input.

- **View Sensor (advanced)** — a ring of distance readings around the robot, like a radar sweep. Higher resolution (more rays) gives better awareness but requires more neurons.
- **Point Sensor (basic)** — detects the nearest entities in front of the robot. Returns a simple numeric encoding for enemy vs. friend. Cheap but limited awareness.

## Neural Network Architecture

- **Type:** small feedforward network (not recurrent, not deep learning)
- **Inputs:** sensor readings (float values, count depends on sensor modules)
- **Hidden layers:** 1-2 layers, size chosen by player (e.g. 8, 16, 32 neurons)
- **Outputs:** actions (move direction x/y, shoot yes/no, aim direction — ~4-6 floats)
- **Activation:** tanh (squashes values to -1..1)
- **Weights/biases:** plain NumPy arrays of floats — this IS the brain
- **Forward pass:** `hidden = tanh(inputs @ weights1 + bias1)` then `outputs = tanh(hidden @ weights2 + bias2)`
- **No backpropagation.** Training is purely evolutionary (genetic algorithm).

## Training System

Training runs continuously during the match and is free (no resource cost).

### Training Arenas

- Each player has **3 training arenas** (slots), one per robot design.
- Training arenas are isolated — hard borders, no connection to the battlefield or other arenas.
- Each arena has slider controls for configuring fitness function weights and sparring partner counts.

### Training Flow

1. **10 "student" robots** of the design being trained are spawned in the arena.
2. The player seeds the arena with **sparring partners**:
   - Own robots assigned as **friend** or **enemy** (any of the player's 3 designs)
   - Scanned enemy robots as enemies (locked to the generation that was scanned)
   - Player chooses how many sparring partners to include
3. The round plays out until a time limit or all robots are dead.
4. The best-performing students survive, are mutated, and form the next generation.
5. Repeat with the sparring partner count the player has configured.

### Fitness Function

The player configures fitness by adjusting weights on predefined components:

- Hitting enemies: +N
- Hitting friends: -N
- Survival time: +N per tick
- Distance toward enemy base: +N per unit closer
- Taking damage: -N
- **Collecting resources: +N** (for gatherer-type robots)

The mix of weights shapes what behavior evolves — aggressive, defensive, evasive, resource-focused, etc.

### Training Opponents

- **Own robots** — always available, player assigns them as friend or enemy
- **Same design** — training against copies of itself
- **Scanned enemy robots** — only available after scanning, locked to the scanned generation

## Resource System

The only resource is a single currency that accumulates over time.

### Income Sources

- **Passive income** — flat rate per second, always ticking
- **Recycling** — player can issue "kill all below generation N" to destroy obsolete units and reclaim some resources
- **Battlefield drops** — destroyed enemy robots drop partial resource value on the battlefield, must be **collected by a robot with a Gatherer module** (resources fly toward the gatherer magnetically)

### Spending

- The player allocates income rate across their 3 robot types (e.g., 60% to fighters, 30% to gatherers, 10% to scouts).
- When accumulated resources for a type reach the cost threshold of one unit, it **auto-spawns** at the base.
- Higher block count = higher cost = slower spawn rate.

### Strategic Tradeoffs

- Cheap robots (few blocks) spawn fast but are weak and dumb
- Expensive robots (many blocks) are capable but spawn slowly and train slowly
- Allocating income to gatherers creates a resource feedback loop but fewer combat units
- Recycling old generations frees resources but reduces battlefield presence

## Scanning

When a robot with a Scanner module successfully scans an enemy robot on the battlefield:

- The player receives a copy of that enemy robot at the **exact generation it was scanned at**.
- This copy can only be used as a **training dummy** in training areas.
- The player can then evolve their own robots specifically to counter that enemy design.

This creates a natural arms race:
1. Player A deploys a strong robot
2. Player B scans it, gets a training dummy
3. Player B trains a counter
4. Player A must evolve past that counter
5. Repeat

## Defense and Catch-Up Mechanics

### Auto-Forking

- The player's base automatically forks (clones) currently active bots near the base on a cooldown.
- Forks are copies of the current generation, not improved versions.
- This gives the defensive player a stream of reinforcements without spending resources.
- Buys time but doesn't win the game on its own, since forks don't evolve further.

### Defensive Advantage

- The base wall and turret(s) provide natural defensive strength.
- Auto-forking means an attacker must sustain pressure to overwhelm defenses.
- But a decisive lead should still convert to a fast win. No drawn-out inevitable losses.

## Win Condition

A player wins when the enemy **commander is hit**. This requires:

1. Breaching the circular wall
2. Landing a shot on the commander (one hit kill)

## Core Player Decisions (Real-Time)

During a match, the player's focus is on:

- Configuring training arenas (fitness weights, sparring partners)
- Watching training to assess behavior and readiness
- Allocating resource income across robot types
- Deciding when to recycle old generations
- Scanning enemy robots for training dummies
- Observing the battlefield and adapting strategy

The battlefield itself is autonomous. The player is an engineer and evolutionary biologist, not a soldier.

## Tech Stack

- **Python** — main language
- **Pygame** — rendering, input, game loop
- **NumPy** — neural net forward pass, genetic algorithm / evolution math (keeps the hot path in C)
- **dataclasses** — clean modular robot composition (body, weapon, engine, scanner, brain)
- Architecture: straightforward OOP / dataclass composition, no heavy framework

### Why Python

- Fast iteration, easy to debug and review
- NumPy handles the compute-heavy neural net and evolution math at near-native speed
- Pygame is sufficient for the visual scale (2D, modest entity counts)
- Training loops can run at pure compute speed (NumPy) without rendering every frame
- If the game outgrows Python, the architecture ports cleanly to Rust + Bevy later
