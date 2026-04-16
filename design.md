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
- The wall must be breached before the commander can be hit. The wall has 600 HP (3× standard).
- **No turrets** — bases have no active defenses. Players defend by spawning cheap bots near their base instead, creating emergent defensive strategies.
- Robots spawn **outside** the wall on the side facing the enemy base (left player: right of wall, right player: left of wall).

## Screen Layout

- **Top:** Player 1's training zone (one long strip)
- **Center:** Battlefield with bases on left and right
- **Bottom:** Player 2's training zone (one long strip)

Each player has a single training zone spanning the full width of the screen. The zone trains one robot design at a time — the player selects which of their 3 designs to train and can hot-swap at any time. The zone has slider controls for configuration (fitness weights, sparring partner counts, etc.) and an indicator showing which design is currently training and its generation count.

## Robot Design (Modular)

Players assemble robots from small modular blocks. Each block adds to the robot's cost and complexity.

### Module Types

- **Armor / Body** — structural blocks, contributes 2× HP to the robot's health pool
- **Engine** — movement speed and maneuverability. No engine = can't move.
- **Weapon** — shooting device
- **Sensor** — perception modules (see Sensor Types below). No sensor = blind.
- **Scanner** — used to scan enemy robots (see Scanning below)
- **Gatherer** — magnetically collects resources from the battlefield in an area around the robot (dropped resources fly toward the gatherer like a magnet)
- **Radar** — omnidirectional radar: each radar block detects the Nth nearest enemy (angle + distance) and Nth nearest friend (angle + distance). First radar = closest, second radar = 2nd closest, etc. Adds 4 brain inputs per block. No range limit — works across the entire arena.
- **Beacon** — base compass: detects enemy base (angle + distance) and friendly base (angle + distance). Adds 4 brain inputs. Max 1 per robot. Omnidirectional, unlimited range.

### Damage & Health

Robots use a **single HP pool** — no per-block damage. Every block contributes 20 HP to the pool, except Armor (PLAIN) blocks which contribute 40 HP (double). When HP reaches zero the robot dies. All modules remain fully functional until death — sensors, engines, weapons, radars all work at full capacity the entire time. This keeps the neural network's input/output contract stable throughout combat, which is critical for small GA-evolved brains that can't adapt to degraded modules mid-fight.

### Collision Damage

Robots take **2 HP per tick** of overlap when colliding with other robots or pressing into a base wall. Both parties take damage on robot-robot collisions. Collision damage counts toward `hits_taken` (penalized by the "Damage taken" fitness weight) but does **not** count as a hit dealt — no attacker gets credit. This creates a natural standoff distance: bots must get close enough to shoot but not so close they ram and take free damage. Prevents stacking on top of enemies or parking against the base wall.

### Fixed Orientation

Robots do **not rotate** during gameplay. Every robot always faces the enemy base (angle fixed at 0). Block directions set during assembly (right, left, up, down) determine where engines thrust, weapons aim, and sensors detect — all relative to this fixed orientation. This simplifies the neural network contract (no rotational state to learn) and makes block placement during assembly the sole source of directional strategy.

### Design Constraints

- More blocks = higher resource cost per spawn
- More blocks = more neural network inputs/outputs = harder to train effectively
- Players must balance capability against cost and trainability
- 3 designs total — forces strategic specialization (e.g., fighter, gatherer, scout)

### Perception Modules

Perception modules determine what the robot's brain receives as input. There are three tiers:

- **Sensor (directional, short-range)** — detects the nearest entity in a 60° cone in its facing direction. Returns distance (normalized) + type (+1 friend, -1 enemy). Range ~150px. Cheap (2 inputs per block) but requires line-of-sight and correct facing.
- **Radar (omnidirectional, unlimited range)** — detects the Nth nearest enemy and Nth nearest friend anywhere in the arena. Returns angle (−1..1, relative to the robot's fixed forward facing) + distance (inverted: 1=close, 0=far). 4 inputs per block. Multiple radars give awareness of multiple targets (1st radar = closest, 2nd = 2nd closest, etc.).
- **Beacon (omnidirectional, unlimited range, max 1)** — detects enemy base and friendly base. Returns angle + distance for each. 4 inputs total. Gives the brain a sense of strategic direction.

### Intrinsic Inputs

Every robot always receives 2 additional brain inputs regardless of modules:

- **Speed** (0..1) — current speed normalized to max speed. Lets the brain learn to brake, accelerate, or maintain speed.
- **Health** (0..1) — current HP / max HP. Lets the brain evolve retreat or kamikaze behavior based on remaining health.

These require no module — they are inherent to having a body.

## Robot Visual Feedback

Every robot on the battlefield and in training zones displays real-time perception overlays:

- **Sensor cones** — each SENSOR block projects a visible 60° cone (two edge lines + arc) in cyan, showing its detection range and facing direction. Multiple sensors show multiple cones.
- **Health bar** — thin bar above the robot, only shown after taking damage. Color shifts from green (full HP) through yellow to red (low HP) against a dark red background. Hidden at full HP to reduce visual noise.
- **Speed arrow** — gray line extending from the robot in its movement direction, length proportional to current speed (hidden when stationary).

These overlays help the player understand what their bots perceive and how they behave, both during training and on the battlefield.

## Neural Network Architecture

- **Type:** small feedforward network (not recurrent, not deep learning)
- **Inputs:** auto-determined by perception modules on the robot (sensors: 2 per block, radars: 4 per block, beacon: 4 if present) + 2 intrinsic inputs always present (speed, health)
- **Hidden layers:** 1-2 layers, size chosen by player during assembly (e.g. 8, 16, 32 neurons)
- **Outputs:** auto-determined by engine/weapon modules on the robot (move direction x/y, shoot yes/no, aim direction — ~4-6 floats)
- **Activation:** tanh (squashes values to -1..1)
- **Weights/biases:** plain NumPy arrays of floats — this IS the brain
- **Forward pass:** `hidden = tanh(inputs @ weights1 + bias1)` then `outputs = tanh(hidden @ weights2 + bias2)`
- **No backpropagation.** Training is purely evolutionary (genetic algorithm).
- **Evolution:** population of 20 brains per design. Each generation: top 3 elites survive unchanged, remaining 17 are produced by uniform crossover (each weight randomly from one of two top-half parents) + sparse mutation (gaussian noise with σ=0.1 applied to ~20% of weights). No mutation decay — constant exploration rate.

### Network Configuration (During Assembly)

The hidden layer size is locked during robot assembly — it cannot be changed after training begins, since changing the architecture invalidates all evolved weights.

The assembly screen shows a **live network visualization** for each robot design (implemented):
- **Input nodes** (left, green) — auto-determined by perception modules. Sensors: 2 inputs each, labels `S(R) dist`, `S(R) type`. Radars: 4 inputs each, labels `R1 eAng`, `R1 eDst`, `R1 fAng`, `R1 fDst`. Beacon: 4 inputs, labels `B eAng`, `B eDst`, `B fAng`, `B fDst`. Always includes 2 intrinsic inputs: `Speed` (0..1, normalized to max speed) and `Health` (0..1, current/max HP). Updates in real time as modules are added/removed.
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
- **Sim dimensions match the battlefield** (1400×600) so bots train at real distances. Base positions use the same x-coordinates as the battlefield.
- Layout: config panel on the left (350px), arena viewport on the right (sized to actual rendered area).
- Config panel is split into three columns: left column for spawn actions (spawn bot 1/2/3, destroy old), center column for training config (design selector, enemy/friend sparring type and count, resource drops, spawn distance), right column for all 10 fitness weight sliders. Player navigates with up/down within a column, left/right to switch columns, primary/secondary to adjust values.
- **Spawn distance:** configurable as Close/Medium/Far. Controls how far apart students and enemies spawn. Close puts them within sensor range from the start — useful for early training before radar-equipped bots learn long-range navigation.
- **Bases in training:** Each training zone contains a friendly base and an enemy base, positioned to match the player's actual game layout (Player 1: friendly left, enemy right; Player 2: friendly right, enemy left). Students spawn near their friendly base, enemies near the enemy base. **Bases have real walls** — robots are blocked by walls (with collision damage) and bullets damage the wall, just like on the battlefield. Wall HP resets each generation. This lets bots train to breach base walls. Hitting the enemy base wall earns "Hit eBase" fitness credit; hitting the friendly base wall earns "Hit fBase" fitness credit (penalized by default). Base walls dim as they take damage. Base positions feed into beacon inputs and distance fitness metrics, ensuring trained brains transfer correctly to the battlefield.
- Arena viewport shows the simulation with stats overlay (generation, fitness, alive count).
- Per-slot generation counts are displayed so the player can see progress on all 3 designs.
- **Resource injection:** The player can configure how many resource drops (0-20) spawn each generation. This allows training pure gatherers without needing the full battlefield resource system — bots evolve to magnetically collect yellow drops via GATHERER blocks.

### Training Flow

1. **20 "student" robots** of the currently selected design are spawned at the **same fixed point** near their friendly base. (All robots always face the enemy base — see Fixed Orientation.) Identical starting conditions ensure the GA evaluates brain quality, not spawn luck.
2. **Students are invisible to each other** — no collisions, no sensor/radar detection between students. Each student only perceives and interacts with sparring partners. This prevents crowding artifacts from 20 overlapping bots and ensures each student gets a clean training signal.
   - **No friendly fire in training** — bullets from students pass through same-team robots without dealing damage. This prevents the GA from evolving pacifist brains that avoid shooting to dodge friendly-hit penalties. Collision damage between same-team robots still applies to prevent stacking.
3. The player seeds the zone with **sparring partners**:
   - Own robots assigned as **friend** or **enemy** (any of the player's 3 designs)
   - Scanned enemy robots as enemies (locked to the generation that was scanned)
   - Player chooses how many sparring partners to include
4. The round plays out until a time limit (600 ticks) or all students are dead.
5. The best-performing students survive as elites. The rest are produced via crossover of two top-half parents + sparse mutation.
6. Repeat.

### Fitness Function

The player configures fitness by adjusting weights on predefined components. Every weight slider goes both positive and negative — the player can reward or punish any metric (e.g., reward taking damage for a blocker bot). Defaults are sensible but nothing is locked:

- **Hit enemy** (default +50): reward/punish landing hits on enemies
- **Hit eBase** (default +30): reward/punish landing hits on the enemy base wall
- **Survival** (default +0.1): reward/punish per tick alive
- **Damage taken** (default -5): reward/punish per hit received
- **Dist to enemy** (default 0): reward/punish based on how much closer the robot got to the nearest enemy compared to its spawn distance. Uses delta-based scoring: `sqrt((initial_dist - best_dist) / initial_dist)`. The square root curve rewards getting closer when already close more than initial approach. 0 = never moved closer, 1 = reached the target. A bot that spawns close and sits still scores 0.
- **Dist to friend** (default 0): same delta-based scoring toward nearest friendly robot. Positive = reward closing distance to allies.
- **Dist to eBase** (default 0): same delta-based scoring toward enemy base. Positive = reward pushing toward enemy base.
- **Dist to fBase** (default 0): same delta-based scoring toward friendly base. Positive = reward moving toward home / defending.
- **Collect resources** (default 0): reward/punish per resource collected
- **Scan enemy** (default 0): reward/punish per successful scanner hit on an enemy robot (friends cannot be scanned)

All distance metrics use delta-based scoring: they compare the robot's best (closest) distance achieved against its initial spawn distance. This rewards active movement toward targets rather than lucky spawn positions.

The mix of weights shapes what behavior evolves — aggressive, defensive, evasive, resource-focused, etc.

Fitness weight values are color-coded in the UI: green for positive, red for negative, gray for zero.

### Training Opponents

- **Own robots** — always available, player assigns them as friend or enemy
- **Same design** — training against copies of itself
- **Scanned enemy robots** — only available after scanning, locked to the scanned generation

## Resource System

The only resource is a single currency that accumulates over time. Robots are **manually spawned** by the player — no auto-spawning.

### Income Sources

- **Passive income** — flat rate per second, always ticking (5 resources/sec)
- **Destroying old generations** — player can destroy all robots of their oldest generation on the battlefield, reclaiming 30% of their spawn cost
- **Battlefield drops** — destroyed enemy robots drop partial resource value on the battlefield, must be **collected by a robot with a Gatherer module** (resources fly toward the gatherer magnetically)

### Spending

- **3 spawn buttons** (one per robot design) in the training zone config panel.
- Each button shows the cost (block count × 10 resources per block).
- The player manually decides when to spawn each type — press primary action on a spawn button to deploy one robot of that design with the latest trained brain.
- Spawn buttons are color-coded: **green** when affordable, **red** when too expensive.

### Destroy Oldest Generation

- **1 destroy button** in the training zone config panel.
- Kills all robots of the player's oldest generation currently on the battlefield.
- Reclaims 30% of their spawn cost as resources.
- Useful for clearing out obsolete bots that are wasting space and replacing them with better-trained ones.

### Generation Numbers

- Each robot on the battlefield displays a small **generation number** below it, showing which training generation its brain came from.
- This lets the player visually identify which bots are old/obsolete vs newly deployed.

### Strategic Tradeoffs

- Cheap robots (few blocks) are affordable but weak and dumb
- Expensive robots (many blocks) are capable but drain resources fast
- The player must choose **when** to spawn — deploying too early wastes resources on undertrained bots, waiting too long leaves the battlefield empty
- Destroying old generations frees partial resources but reduces battlefield presence

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

### Defensive Advantage

- The base wall (600 HP) provides natural defensive strength.
- Players can spawn cheap untrained bots near their base as an effective defensive screen — an emergent strategy that replaces the need for turrets.
- Manual spawning means the player can stockpile resources and deploy a wave of well-trained bots at a critical moment.
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
- **Secondary:** cycle block direction (right → down → left → up) on directional blocks — determines thrust, aim, or sensor facing relative to the robot's fixed orientation
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
- `modules.py` — block-based robot system (PLAIN/ENGINE/WEAPON/SENSOR/SCANNER/GATHERER/RADAR/BEACON), blueprints, serialize/deserialize
- `entities.py` — Robot (block-based), Bullet, Base
- `physics.py` — vectorized NumPy: batch sensors (directional sensors + radar + beacon), collisions (robot-robot, bullet-robot, bullet-base)
- `renderer.py` — Pygame drawing: blocks, bases, bullets, HUD, training arena viewports, robot perception overlays (sensor cones, health bars, speed arrows)
- `assembly.py` — pre-game robot assembly screen (side-by-side, grid editor, cursor, slot tabs, ready flow, save/load designs, live network visualization with input/output labels, hidden size selector with suggestions, param count)
- `main.py` — game loop with ASSEMBLY → MATCH phase state machine
- `training.py` — single training zone per player with configurable sparring and fitness. One zone trains one design at a time with hot-swap (populations preserved per design). Zone runs in its own subprocess via `multiprocessing` (spawn context). Worker ticks at 3× game speed (rate-limited), sends render snapshots + best brains through Pipes, receives config updates. Main process holds `TrainingZoneProxy` (render data) + `TrainingZoneUI` (player input handling with two-column cursor navigation and key repeat). Config panel on left (350px) split into three columns: spawn actions, training config, fitness sliders. Arena viewport on right (sized to actual rendered area, no misleading dead space). Lightweight pure-Python physics for small entity counts, including radar/beacon input computation with per-player base positions. **Real base walls in training:** training zones contain actual Base objects with walls — robots are blocked by walls (with collision damage), bullets damage walls, and wall HP resets each generation. This allows bots to train base-breaching behavior. **Student isolation:** students are completely invisible to each other — no collisions, no sensor detection, no radar detection. Each student only perceives and interacts with sparring partners. This prevents crowding artifacts and forces brains to learn from actual combat encounters. All students spawn at the **same fixed point** facing the enemy base each generation, ensuring identical starting conditions so the GA evaluates brain quality not spawn luck. **Training zone dimensions match the battlefield** (1400×600 sim coords) with base positions at the same x-coordinates as the real battlefield, so trained behavior transfers directly. Training zones include visible friendly and enemy bases matching the player's game-side layout (P1: friendly left/enemy right, P2: friendly right/enemy left); students spawn near their friendly base, enemies near the enemy base. Base positions feed into beacon inputs and distance fitness metrics. **Training zone config:** active design selector, enemy sparring type/count, friend sparring type/count, resource drop count (for gatherer training), spawn distance (close/medium/far), 10 fitness weight sliders (hit enemy, hit eBase, survival, damage taken, dist to enemy, dist to friend, dist to enemy base, dist to friendly base, collect resources, scan enemy). **No friendly fire in training** — same-team bullets pass through without damage, preventing pacifist evolution. Distance fitness metrics use per-tick accumulation of actual distances (not positional proxies). 15-second setup period at match start (training paused, player configures). Resource drops are yellow collectibles gathered magnetically by GATHERER blocks.

### Needs Rework
- ~~**Training system** — restructure from 3 arenas per player to 1 training zone per player. Wider zone (full screen width), single active design with hot-swap, 15s setup delay at match start, generation count indicator.~~ (done)
- ~~**Assembly screen** — add hidden layer size selector per design, add live network visualization (inputs/hidden/outputs with connection lines), auto-derive input/output counts from placed modules.~~ (done)
- **Performance** — ~~parallelize training via multiprocessing~~ (done). Profile remaining bottlenecks (battlefield physics, sensor calculations at scale).

### Next Up
- ~~**Resource system** — passive income, manual spawn buttons, destroy oldest generation, generation numbers on bots~~ (done)
- **Battlefield drops** — destroyed enemy robots drop partial resource value, collected by gatherer bots
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
