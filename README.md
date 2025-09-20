# SATOPT

SATOPT (Satellite Optimization) is a Python-based framework for optimizing satellite (or more generally, 3D mesh) shapes using Free‑Form Deformation (FFD) control points, drag or other physical‑models, and both global & local optimization techniques under geometric constraints.

---

## Table of Contents

- [Features](#features)  
- [Getting Started](#getting-started)  
- [Core Concepts](#core-concepts)  
- [Installation](#installation)  
- [Usage / Examples](#usage--examples)  
- [Optimization Workflow](#optimization-workflow)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Features

- Free‑Form Deformation (FFD) framework to deform 3D meshes via control points  
- Drag (or other physics) model(s) to evaluate shape performance  
- Mesh volume / non‑flip / orientation constraints to maintain valid deformations  
- Supports local optimization via SLSQP and potential for global methods (e.g. GA)  
- Example code / data for test meshes  

---

## Getting Started

To use SATOPT, the typical workflow is:

1. Load or craeate a base 3D mesh  
2. Define an FFD (control point grid) over the mesh  
3. Define objective function(s) (drag, plus penalties for volume loss, flipped cells, etc.)  
4. Export optimized mesh  

---

## Core Concepts

- **Mesh**: the 3D geometry you want to optimize (e.g. satellite body, wing, etc.).  
- **FFD (Free‑Form Deformation)**: a grid of control points; moving them deforms the embedded mesh smoothly.  
- **Control Points**: the variables of optimization; their positions determine deformation.  
- **Objective / Cost Function**: typically drag (or other aerodynamic/physical quantity) + penalty terms (e.g. minimum volume). 
- **DragModel**: Model used for calculating the drag. (can be extended) 
- **Constraints**: ensure mesh remains valid (elements not inverted), maintain volume, etc.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/schuettem/SATOPT.git
cd SATOPT

# Recommended: create & activate a virtual environment
conda create -n satopt python=3.12
conda activate satopt

# Install required packages
pip install -e ./satopt
```


---

## Usage / Examples

There are example scripts in the `examples/` directory. These show how to:

- Create initial meshes and FFDs  
- Define drag (or other) models  
- Run optimizations

A minimal usage snippet:

```python
from src.satopt import run

# Example call: specify model, parameters
run(
    drag_model_name = "MyDragModel",
    drag_kwargs = { ... },
    min_volume = 0.8,
    radius = 4.0,
    n_control_points = [2,2,2]
)
```

---

## Project Structure

```
SATOPT/
├── src/
│   ├── core modules: mesh, FFD, drag models, constraint definitions, objective functions
│   └── optimization routines (local / candidates for global)
├── examples/
│   ├── sample problems
│   └── demos of optimization
├── data/
│   └── meshes, test files
├── old_implementation/
│   └── legacy code (for reference)
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Contributing

Contributions are very welcome! You can help by:

- Adding global optimizers (GA, DE, etc.)  
- Improving constraints (e.g. more robust non‑flip checking)  
- Adding new drag or physics models  
- Writing more examples / tutorials  


---

## License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.
