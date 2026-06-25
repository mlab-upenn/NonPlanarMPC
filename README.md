# NonPlanarMPC

Reference controller for **Nonplanar Model Predictive Control for Autonomous
Vehicles with Recursive Sparse Gaussian Process Dynamics** (Amine, Puri, Le,
Mangharam — IEEE Intelligent Vehicles Symposium, 2026).

The method is a sampling-based **MPPI** controller on a single-track dynamic
bicycle model, augmented with an online-learned **recursive sparse Gaussian-process
residual** that compensates for nonplanar terrain (slopes, banks, hills).

> [!IMPORTANT]
> **The controller implementation is being finalized (cleanup) and will be released here
> shortly.** For early access or questions in the meantime, please contact
> **Ahmad Amine** (aminea at upenn.edu).

## Simulation environment

The custom NVIDIA Isaac Sim environment originally developed for this project is
**deprecated**. The simulation project evolved into the open-source
**[Autoware off-road sim](https://github.com/autowarefoundation/autoware_off-road_sim)**,
included here as a submodule under [`third_party/`](third_party/). It ships the
nonplanar tracks used in the paper (L-shaped, kidney, oval) and a RoboRacer
vehicle with a ROS 2 interface. All credits for the sim go to the original authors of the Autoware off-road sim. The NonPlanarMPC controller is compatible with the sim and can be run on the tracks with the launch configurations in [`sim_configs/`](sim_configs/).

## Repository layout

| Path | Contents |
|---|---|
| [`third_party/autoware_off-road_sim/`](third_party/) | Simulation environment - submodule |
| [`third_party/f1tenth_planning/`](third_party/) | Motion-planning library the controller builds on - submodule |
| [`sim_configs/`](sim_configs/) | Off-road sim launch configs - *coming soon* |
| [`tracks/`](tracks/) | Per-track raceline + terrain data - *coming soon* |
| [`scripts/`](scripts/) | Run / training / extraction entry points - *coming soon* |
| [`docs/`](docs/) | Project website - *coming soon* |

The `npmpc` controller package will be added at the top level when released.

## Submodules

```bash
git clone --recurse-submodules https://github.com/mlab-upenn/NonPlanarMPC.git
# or, after a plain clone:
git submodule update --init --recursive
```

## Citation

```bibtex
@misc{amine2026nonplanarmodelpredictivecontrol,
      title={Nonplanar Model Predictive Control for Autonomous Vehicles with Recursive Sparse Gaussian Process Dynamics}, 
      author={Ahmad Amine and Kabir Puri and Viet-Anh Le and Rahul Mangharam},
      year={2026},
      eprint={2602.16206},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.16206}, 
}
```
