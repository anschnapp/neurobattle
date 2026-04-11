# Robot Evolution - Game Design

## Concept

A 2-player competitive game where players don't fight directly. Instead, they design robot bodies, train robot brains through neural network evolution, and deploy them onto a shared battlefield. The core loop is: **build the body, evolve the brain, deploy, adapt.**

## Players and Bases

- Two players, each with a static base on opposite sides of the screen (left and right).
- A base is a circular wall with a **commander** (red point) at the center.
- The commander is fragile: one hit and the player loses.
- The wall must be breached before the commander can be hit.
- Each base starts with one **turret** mounted on the circle wall. The turret rotates around the wall automatically and shoots at incoming threats.
- Turrets are trained using the same neural network system as robots.
- Players can potentially earn additional or improved turrets over time.

## Robot Design (Modular)

Players assemble robots from small modular parts:

- **Shape / Armor** - the body, determines durability
- **Shooting device** - weapon module
- **Engines** - movement speed and maneuverability
- **Sensors** - perception modules (see Sensor Types below)
- **Scanners** - used to scan enemy robots (see Scanning below)
- **Neurons** - the player selects how many neurons the robot's brain gets

More neurons means a smarter potential robot, but harder and slower to train effectively.

### Sensor Types

Sensors are physical modules that determine what the robot's brain receives as input. Sensor choice directly affects how many neurons are needed.

- **View Sensor (advanced)** - a ring of distance readings around the robot, like a radar sweep. Each value is the distance to the nearest object in that direction. Higher resolution (more rays) gives better awareness but requires more neurons to process.
- **Point Sensor (basic)** - detects the nearest entities in front of the robot. Returns a simple numeric encoding: different values for enemy vs. friend. Cheap, few inputs, but limited awareness. Works well with small brains.

## Neural Network Architecture

- **Type:** small feedforward network (not recurrent, not deep learning)
- **Inputs:** sensor readings (float values, count depends on sensor modules)
- **Hidden layers:** 1-2 layers, size chosen by player (e.g. 8, 16, 32 neurons)
- **Outputs:** actions (move direction x/y, shoot yes/no, aim direction — ~4-6 floats)
- **Activation:** tanh (squashes values to -1..1)
- **Weights/biases:** plain NumPy arrays of floats — this IS the brain
- **Forward pass:** `hidden = tanh(inputs @ weights1 + bias1)` then `outputs = tanh(hidden @ weights2 + bias2)`
- **No backpropagation.** Training is purely evolutionary (genetic algorithm).

### Evolution / Genetic Algorithm

1. Create a population of N brains (random weights)
2. Run them all in the training arena (virtual, fast, no rendering needed)
3. Score each brain using the player's fitness function
4. Keep the best, mutate copies to fill the next generation
5. Repeat

Mutation is simple: `child_weights = parent_weights + random_noise * mutation_rate`

## Neural Network Training

Robots are too dumb to do anything useful by default. Their behavior is entirely driven by a neural network that must be trained through evolution/generational training.

### Training Areas

- Each player has a limited number of **training slots** (small arenas).
- Training is virtual: spawning and killing bots in training costs no resources. Generations can iterate freely.
- The player watches training happen in real-time in these smaller arenas, observing how their bots behave, learn, and fail.
- This observation IS the primary feedback mechanism. No abstract stats needed - the player sees the behavior directly.

### Training Slot Allocation

Training slots are a core strategic resource. Players must decide:

- How many slots for offense bots vs. defense turrets
- Whether to deep-train one design (many generations) or broad-train several designs (fewer generations each)
- When a generation is "good enough" to deploy
- What training opponents to use (own bots, scanned enemies, etc.)

### Training Configuration

The player sets up each training slot by:

1. **Spawning robots** into the arena and assigning them roles: **friend** or **enemy**. These can be any of the player's own designs — the role assignment is per-spawn, not per-design.
2. **Selecting a fitness function** from predefined building blocks, e.g.:
   - Hitting enemies: +N
   - Hitting friends: -N
   - Survival time: +N/tick
   - Distance toward enemy base: +N per unit closer
   - Taking damage: -N
3. The player tunes the weights of these fitness components. The mix shapes what behavior evolves — aggressive, defensive, evasive, etc.

Two players with identical robot hardware can evolve completely different behaviors through different fitness tuning.

### Training Opponents

- **Own robots** - always available, player assigns them as friend or enemy in the arena
- **Same design** - training against copies of itself, always allowed
- **Scanned enemy robots** - only available after scanning (see below), locked to the generation that was scanned

## Scanning

When a robot with a scanner successfully scans an enemy robot on the battlefield:

- The player receives a copy of that enemy robot at the **exact generation it was scanned at**.
- This copy can only be used as a **training dummy** in training areas.
- The player can then evolve their own robots specifically to counter that enemy design.

This creates a natural arms race:
1. Player A deploys a strong robot
2. Player B scans it, gets a training dummy
3. Player B trains a counter
4. Player A must evolve past that counter
5. Repeat

## Deployment

- Deploying a trained robot onto the real battlefield costs **training time** (the robot leaves the virtual training and becomes a real unit).
- Once deployed, robots act autonomously based on their trained neural network. The player has no direct control.

## Resources

The only resource is **time**, expressed through:

- **Training time** - generations take time to evolve; better bots require more training
- **Training slots** - limited concurrent training capacity; allocation is a strategic choice
- **Deployment** - committing a bot to the field is an irreversible decision

No other resource systems (energy, currency, materials, etc.).

## Defense and Catch-Up Mechanics

### Auto-Forking

- The player's base automatically forks (clones) currently active bots near the base on a cooldown.
- Forks are copies of the current generation, not improved versions.
- This gives the defensive player a stream of reinforcements without spending training slots.
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

- Designing robot module loadouts
- Allocating training slots
- Choosing training opponents and swapping dummies
- Watching training to assess behavior and readiness
- Deciding when to deploy
- Deciding whether to scan or shoot incoming enemy bots

The battlefield itself is autonomous. The player is an engineer and evolutionary biologist, not a soldier.

## Tech Stack

- **Python** - main language
- **Pygame** - rendering, input, game loop
- **NumPy** - neural net forward pass, genetic algorithm / evolution math (keeps the hot path in C)
- **dataclasses** - clean modular robot composition (body, weapon, engine, scanner, brain)
- Architecture: straightforward OOP / dataclass composition, no heavy framework

### Why Python

- Fast iteration, easy to debug and review
- NumPy handles the compute-heavy neural net and evolution math at near-native speed
- Pygame is sufficient for the visual scale (2D, modest entity counts)
- Training loops can run at pure compute speed (NumPy) without rendering every frame
- If the game outgrows Python, the architecture ports cleanly to Rust + Bevy later
