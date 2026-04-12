# NeuroBattle

A 2-player competitive game where you don't fight directly — you design robot bodies, evolve their brains through neuroevolution, and watch them battle autonomously.

**Build the body. Evolve the brain. Watch them fight. Adapt.**

## Status

This project is a **work in progress** and not yet playable as a full game. Core systems like robot assembly, neural network training, and the genetic algorithm are functional, but major features (resource economy, battlefield combat loop, scanning/arms race, win condition) are still missing.

## What Works So Far

- **Robot Assembly** — modular block-based robot designer with armor, engines, weapons, sensors, radar, beacon, scanner, and gatherer modules
- **Neural Networks** — small feedforward nets with auto-derived inputs/outputs based on robot design, live network visualization during assembly
- **Neuroevolution** — genetic algorithm training in parallelized subprocesses, configurable fitness functions, hot-swappable training zones
- **Physics** — vectorized NumPy-based sensor calculations, collisions, and movement

## What's Still Missing

- Resource system (income, spawning, recycling)
- Battlefield combat loop and win condition
- Scanning enemies for use as training dummies
- General polish and balancing

## Tech

Python, Pygame, NumPy
