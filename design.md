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

- **Top:** Player 1's training zone (one long strip)
- **Center:** Battlefield with bases on left and right
- **Bottom:** Player 2's training zone (one long strip)

Each player has a single training zone spanning the full width of the screen. The zone trains one robot design at a time — the player selects which of their 3 designs to train and can hot-swap at any time. The zone has slider controls for configuration (fitness weights, sparring partner counts, etc.) and an indicator showing which design is currently training and its generation count.

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
- **Inputs:** auto-determined by sensor modules on the robot (float values)
- **Hidden layers:** 1-2 layers, size chosen by player during assembly (e.g. 8, 16, 32 neurons)
- **Outputs:** auto-determined by engine/weapon modules on the robot (move direction x/y, shoot yes/no, aim direction — ~4-6 floats)
- **Activation:** tanh (squashes values to -1..1)
- **Weights/biases:** plain NumPy arrays of floats — this IS the brain
- **Forward pass:** `hidden = tanh(inputs @ weights1 + bias1)` then `outputs = tanh(hidden @ weights2 + bias2)`
- **No backpropagation.** Training is purely evolutionary (genetic algorithm).

### Network Configuration (During Assembly)

The hidden layer size is locked during robot assembly — it cannot be changed after training begins, since changing the architecture invalidates all evolved weights.

The assembly screen shows a **live network visualization** for each robot design (implemented):
- **Input nodes** (left, green) — auto-determined by sensor modules. Each sensor contributes 2 inputs: distance + type. Labels show sensor direction, e.g. `S(R) dist`, `S(R) type`. Updates in real time as sensors are added/removed.
- **Hidden layer nodes** (center, blue) — player cycles through options (4, 8, 12, 16, 24, 32) using primary/secondary on the network row below the grid. A **suggested size** is shown based on `max(8, (inputs + outputs) * 1.5)`.
- **Output nodes** (right, orange) — auto-determined by engine/weapon modules. Each engine = 1 output, each weapon = 1 output. Labels show type and direction, e.g. `E(L)`, `W(R)`.
- **Connections** — faded lines drawn between layers, showing density. A compact 4-8-3 network looks clean. A sprawling 12-32-6 network looks visually dense — communicating "harder to train" through the visualization itself.
- **Overflow** — layers with more than 8 nodes cap the visual display and show `+N` for remaining nodes.
- **Param count** — total weight count displayed on the right (e.g. "156 params"), giving a concrete measure of brain complexity.

This makes the cost of complexity tangible before the match starts.

## Training System

Training is free (no resource cost). Each player has a **single training zone** (one long strip) that trains one robot design at a time.

### Training Zone Lifecycle

1. **Match start: 15-second setup period.** The training zone is inactive. The player uses this time to select which design to train, configure fitness weights, and set up sparring partners.
2. **After 15 seconds: training activates** and runs continuously for the rest of the match.
3. **Hot-swapping:** The player can switch which design is being trained or adjust parameters at any time. Switching designs preserves the previous design's evolved weights — training resumes from where it left off when that design is selected again.

### Training Zone

- The zone is isolated — hard borders, no connection to the battlefield.
- Layout: config panel on the left (350px), arena viewport on the right (remaining width).
- Config panel shows: active design selector, enemy/friend sparring type and count, resource drop count, and 6 fitness weight sliders. Player navigates rows with up/down, adjusts values with primary/secondary.
- Arena viewport shows the simulation with stats overlay (generation, fitness, alive count).
- Per-slot generation counts are displayed so the player can see progress on all 3 designs.
- **Resource injection:** The player can configure how many resource drops (0-20) spawn each generation. This allows training pure gatherers without needing the full battlefield resource system — bots evolve to magnetically collect yellow drops via GATHERER blocks.

### Training Flow

1. **10 "student" robots** of the currently selected design are spawned in the zone.
2. The player seeds the zone with **sparring partners**:
   - Own robots assigned as **friend** or **enemy** (any of the player's 3 designs)
   - Scanned enemy robots as enemies (locked to the generation that was scanned)
   - Player chooses how many sparring partners to include
3. The round plays out until a time limit or all robots are dead.
4. The best-performing students survive, are mutated, and form the next generation.
5. Repeat.

### Fitness Function

The player configures fitness by adjusting weights on predefined components. Every weight slider goes both positive and negative — the player can reward or punish any metric (e.g., reward taking damage for a blocker bot). Defaults are sensible but nothing is locked:

- **Hit enemy** (default +50): reward/punish landing hits on enemies
- **Hit friend** (default -30): reward/punish landing hits on friendlies (tracked separately from enemy hits)
- **Survival** (default +0.1): reward/punish per tick alive
- **Damage taken** (default -5): reward/punish per hit received
- **Dist to enemy** (default 0): reward/punish based on how far toward the enemy side of the arena the robot is (normalized 0..1 across arena width)
- **Collect resources** (default 0): reward/punish per resource collected

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

## Controls

Simple two-player keybindings, no configuration needed:

- **Player 1:** WASD (navigate) + E (primary action) + Q (secondary action)
- **Player 2:** Arrow keys (navigate) + Right Ctrl (primary action) + Right Shift (secondary action)

Input is read via `pygame.key.get_pressed()` (polling), not key events. This avoids modifier interference — e.g., Right Shift is just a button, not a modifier that changes other key behavior.

### Assembly Controls

- **Move:** navigate cursor on 9x9 grid
- **Primary:** place block (if empty) or cycle type (Armor → Engine → Weapon → Sensor → Scanner → Gatherer → remove)
- **Secondary:** cycle facing direction on directional blocks
- **Cursor above grid → slot tabs:** switch between 3 robot designs
- **Cursor below grid → network size control:** cycle hidden layer neuron count for the current design
- **Below network control → READY button:** lock in designs (requires all 3 to have blocks)

### Persistence

Last designs are saved to `last_designs.json` on match start and auto-loaded on next launch.

## Implementation Status

### Done
- `settings.py` — constants, colors, keybindings, screen layout (1400x1000: top 200px P1 training / center 600px battlefield / bottom 200px P2 training)
- `brain.py` — feedforward neural net (NumPy), save/load, copy
- `evolution.py` — genetic algorithm: Population with fitness, selection, mutation
- `modules.py` — block-based robot system (PLAIN/ENGINE/WEAPON/SENSOR/SCANNER/GATHERER), blueprints, serialize/deserialize
- `entities.py` — Robot (block-based), Bullet, Base, Turret
- `physics.py` — vectorized NumPy: batch sensors, collisions (robot-robot, bullet-robot, bullet-base)
- `renderer.py` — Pygame drawing: blocks, bases, bullets, HUD, training arena viewports
- `assembly.py` — pre-game robot assembly screen (side-by-side, grid editor, cursor, slot tabs, ready flow, save/load designs, live network visualization with input/output labels, hidden size selector with suggestions, param count)
- `main.py` — game loop with ASSEMBLY → MATCH phase state machine
- `training.py` — single training zone per player with configurable sparring and fitness. One zone trains one design at a time with hot-swap (populations preserved per design). Zone runs in its own subprocess via `multiprocessing` (spawn context). Worker ticks at 3× game speed (rate-limited), sends render snapshots + best brains through Pipes, receives config updates. Main process holds `TrainingZoneProxy` (render data) + `TrainingZoneUI` (player input handling with cursor navigation and key repeat). Config panel on left (350px), arena viewport on right. Lightweight pure-Python physics for small entity counts. **Training zone config:** active design selector, enemy sparring type/count, friend sparring type/count, resource drop count (for gatherer training), 6 fitness weight sliders (hit enemy, hit friend, survival, damage taken, distance to enemy, collect resources). 15-second setup period at match start (training paused, player configures). Resource drops are yellow collectibles gathered magnetically by GATHERER blocks.

### Needs Rework
- ~~**Training system** — restructure from 3 arenas per player to 1 training zone per player. Wider zone (full screen width), single active design with hot-swap, 15s setup delay at match start, generation count indicator.~~ (done)
- ~~**Assembly screen** — add hidden layer size selector per design, add live network visualization (inputs/hidden/outputs with connection lines), auto-derive input/output counts from placed modules.~~ (done)
- **Performance** — ~~parallelize training via multiprocessing~~ (done). Profile remaining bottlenecks (battlefield physics, sensor calculations at scale).

### Next Up
- **Resource system** — passive income, recycling, battlefield drops, auto-spawn
- **Battlefield & combat** — autonomous fighting, wall breach, win condition polish
- **Scanning & arms race** — scan enemies, use as training dummies

## Tech Stack

- **Python** — main language
- **Pygame** — rendering, input, game loop
- **NumPy** — neural net forward pass, genetic algorithm / evolution math (keeps the hot path in C)
- **dataclasses** — clean modular robot composition (body, weapon, engine, scanner, brain)
- Architecture: straightforward OOP / dataclass composition, no heavy framework

### Performance Strategy

- **Parallelization (implemented):** One training zone per player, each in its own subprocess via `multiprocessing` with `spawn` context. Workers tick at 3× game speed (180 ticks/sec), rate-limited via `TRAINING_TICKS_PER_FRAME`. Communication via `Pipe`: workers send render snapshots (~30fps) and best brain weights (each generation); main process sends commands (stop/pause/resume/config). Two worker processes total (one per player) instead of six.
- **Profile-first:** Identify actual bottlenecks before optimizing. Likely candidates: battlefield sensor calculations, collision detection at scale.

### Why Python

- Fast iteration, easy to debug and review
- NumPy handles the compute-heavy neural net and evolution math at near-native speed
- Pygame is sufficient for the visual scale (2D, modest entity counts)
- Training loops can run at pure compute speed (NumPy) without rendering every frame
- If the game outgrows Python, the architecture ports cleanly to Rust + Bevy later
