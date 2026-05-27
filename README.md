# Multi-Agent Drone Swarm Coordination using Deep Reinforcement Learning

A decentralized multi-UAV swarm coordination framework built using Multi-Agent Reinforcement Learning (MARL) for autonomous target pursuit, formation control, and collision-aware navigation in simulation.

## Overview

This project explores swarm intelligence and decentralized drone coordination using Deep Reinforcement Learning techniques such as MADDPG and MAPPO in PyBullet-based simulation environments.

The primary objective is to train multiple UAV agents to:
- cooperatively track moving targets
- maintain swarm cohesion
- avoid collisions
- exhibit emergent coordinated behavior
- operate without explicit inter-agent communication

The project is being developed as part of ongoing research work at the Robotics and Research Center (RRC), IIIT Hyderabad.

---

# Features

- Multi-agent drone swarm simulation in PyBullet
- Decentralized UAV coordination
- Moving-target pursuit
- Collision avoidance
- Formation control
- Reward shaping for cooperative behavior
- CTDE (Centralized Training, Decentralized Execution)
- MADDPG-based MARL pipeline
- Transitioning to recurrent MAPPO architectures

---

# Environment

The simulation environment includes:
- multiple quadrotor UAV agents
- dynamic/moving targets
- obstacle-aware navigation
- continuous control action space
- physics-based simulation using PyBullet

Agents receive local observations and independently generate actions while being trained under a centralized critic framework.

---

# Architecture

## Current Pipeline

```text
PyBullet Environment
        ↓
Observation Extraction
        ↓
MARL Policy (MADDPG)
        ↓
Continuous Actions
        ↓
Swarm Drone Dynamics
```

---

# Reinforcement Learning Setup

## Algorithm
- MADDPG (Multi-Agent Deep Deterministic Policy Gradient)

## Training Paradigm
- Centralized Training
- Decentralized Execution (CTDE)

## Current Research Extensions
- Recurrent MAPPO
- Temporal coordination
- Improved scalability
- Better partial observability handling

---

# Reward Engineering

The reward system is designed to encourage:

- cooperative target pursuit
- formation stability
- collision avoidance
- swarm cohesion
- smooth navigation

Example reward components include:

| Reward Component | Objective |
|---|---|
| Target Distance Reward | Encourage target pursuit |
| Collision Penalty | Avoid inter-agent crashes |
| Cohesion Reward | Maintain swarm connectivity |
| Formation Reward | Stabilize formations |
| Velocity Penalty | Smooth motion control |

---

# Formation Control

Implemented swarm formations include:
- Line Formation
- V Formation
- Square Formation

Formation tracking uses:
- PD Controllers
- Rotation-frame offsets
- Leader-follower trajectory tracking

---

# Collision Avoidance

Implemented:
- Potential-field repulsion
- Time-to-collision prediction
- Safety-aware trajectory correction

Optimizations:
- solver iteration tuning
- timestep optimization
- batched simulation execution

---

# Technologies Used

## Programming
- Python

## Simulation
- PyBullet
- Gymnasium

## Reinforcement Learning
- MADDPG
- MAPPO
- Stable-Baselines3

## Computer Vision
- OpenCV

---

# Current Progress

- [x] Multi-agent PyBullet environment
- [x] Formation control implementation
- [x] Collision avoidance
- [x] MADDPG training pipeline
- [x] Target pursuit behavior
- [ ] Recurrent MAPPO integration
- [ ] Sim-to-real transfer
- [ ] ROS2/PX4 integration
- [ ] Hardware deployment

---

# Results

The trained swarm demonstrates:
- decentralized coordination
- emergent cooperative behavior
- stable target tracking
- collision-aware navigation
- adaptive multi-agent pursuit

---

# Future Work

Future extensions include:
- ROS2 integration
- PX4 SITL support
- real drone deployment
- vision-based navigation
- communication-aware MARL
- multi-target cooperative pursuit
- sim-to-real transfer learning

---

# Research Inspiration

This project draws inspiration from recent research in:
- Multi-Agent Reinforcement Learning
- Swarm Robotics
- Autonomous UAV Coordination
- Decentralized Control Systems

Relevant concepts include:
- CTDE
- MAPPO
- coordinated exploration
- role-based swarm coordination
- league-play MARL systems

---

# Author

Prakhar Gupta  
B.Tech + M.S by Research  
IIIT Hyderabad

Research Areas:
- Multi-Agent Reinforcement Learning
- Autonomous UAV Systems
- Swarm Robotics
- Drone Intelligence

---

# Acknowledgements

- Robotics and Research Center (RRC), IIIT Hyderabad
- Dr. Harikumar Kandath
- PyBullet
- Stable-Baselines3
- OpenAI Gymnasium
