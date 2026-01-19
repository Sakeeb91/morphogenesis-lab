# Morphogenesis Lab

Interactive GPU-accelerated simulations of pattern formation in nature. Explore reaction-diffusion systems and Laplacian growth instabilities through real-time 3D visualizations.

## Overview

This project implements two fundamental models of biological pattern formation:

1. **Gray-Scott Reaction-Diffusion** - Simulates chemical morphogenesis on curved surfaces
2. **Laplacian Growth (DLA)** - Models fractal aggregation processes like lightning and dendrites

Both simulations run in the browser using Three.js and WebGL, with Python backends for data generation.

---

## Turing Patterns on Sphere

**File:** `turing.html`

Real-time Gray-Scott reaction-diffusion on a spherical membrane. The simulation runs entirely on the GPU using ping-pong framebuffers, achieving 60fps with a 512x512 texture resolution.

### Gray-Scott Model

The system models two chemicals U (substrate) and V (catalyst) with the following PDEs:

```
du/dt = Du * laplacian(u) - u*v^2 + F*(1-u)
dv/dt = Dv * laplacian(v) + u*v^2 - (F+k)*v
```

Where:
- `Du, Dv` = diffusion coefficients
- `F` = feed rate (replenishes U)
- `k` = kill rate (removes V)

### Features

- GPU-accelerated simulation via WebGL fragment shaders
- Four color schemes: Bioluminescent, Thermal, Chlorophyll, Neural
- Fresnel edge glow effect
- Bloom post-processing
- Auto-rotation with orbit controls

---

## Laplacian Growth Simulation

**File:** `index.html` | **Backend:** `growth_sim.py`

Implements the Fuse Breakdown / Diffusion-Limited Aggregation algorithm for generating fractal growth patterns. The simulation uses a potential field approach where new particles attach preferentially to high-curvature regions.

### Algorithm

1. Initialize seed particle(s) at origin
2. Compute potential field: `Phi = Sum(1 - R1/r)` for all aggregate particles
3. Select growth site with probability `P(i) ~ Phi(i)^ETA`
4. Add particle and update potential incrementally
5. Repeat until target count reached

The `ETA` parameter controls branching:
- High ETA (5-6): Thin, lightning-like branches
- Medium ETA (3-4): Dendritic, neuron-like growth
- Low ETA (1-2): Compact, blob-like structures

### Presets

| Preset | ETA | Description |
|--------|-----|-------------|
| lightning | 6.0 | Electrical discharge patterns |
| neurons | 4.0 | Branching neural dendrites |
| moss | 1.5 | Organic blob growth |
| cities | 2.5 | Urban sprawl (2D mode) |

---

## Quick Start

### Run the visualizations

```bash
cd morphogenesis-lab
python3 -m http.server 8000
```

Then open:
- http://localhost:8000/turing.html - Reaction-diffusion on sphere
- http://localhost:8000/index.html - Laplacian growth visualization

### Generate new simulation data

```bash
# Laplacian growth with different presets
python growth_sim.py --preset neurons
python growth_sim.py --preset lightning --particles 5000

# Turing patterns (pre-computed mesh simulation)
python turing_sphere.py --preset stripes --subdivisions 5
```

---

## Project Structure

```
morphogenesis-lab/
├── turing.html          # Real-time GPU reaction-diffusion
├── index.html           # Laplacian growth visualizer
├── growth_sim.py        # Laplacian growth engine (numpy)
├── turing_sphere.py     # Gray-Scott on icosphere mesh (scipy)
├── presets.json         # Growth simulation configurations
└── output/              # Generated JSON data
    ├── growth_data.json
    └── turing_sphere.json
```

---

## Requirements

**Browser:**
- WebGL 2.0 support
- ES6 modules

**Python (for data generation):**
- Python 3.8+
- numpy
- scipy (for turing_sphere.py only)

```bash
pip install numpy scipy
```

---

## Technical Details

### GPU Simulation (turing.html)

The reaction-diffusion simulation uses a ping-pong technique with two WebGL render targets. Each frame:

1. Read chemical concentrations from texture A
2. Compute Laplacian via 5-point stencil with wrapping
3. Apply Gray-Scott reaction equations
4. Write results to texture B
5. Swap textures and repeat

The sphere visualization uses a custom shader that samples the simulation texture and applies:
- Triple smoothstep for soft Bouba-like transitions
- Fresnel rim lighting
- Emissive glow modulated by concentration

### Laplacian Discretization (turing_sphere.py)

For the mesh-based simulation, the Laplace-Beltrami operator is discretized using cotangent weights:

```
w_ij = (cot(alpha_ij) + cot(beta_ij)) / 2
```

Where alpha and beta are the angles opposite to edge (i,j) in adjacent triangles.

---

## References

- Pearson, J.E. (1993). "Complex Patterns in a Simple System." Science 261(5118):189-192.
- Niemeyer, L., Pietronero, L., Wiesmann, H.J. (1984). "Fractal Dimension of Dielectric Breakdown." Physical Review Letters 52:1033.
- Meyer, M. et al. (2003). "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds."
- Turing, A.M. (1952). "The Chemical Basis of Morphogenesis." Philosophical Transactions of the Royal Society B.

---

## License

MIT
