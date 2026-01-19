#!/usr/bin/env python3
"""
Turing Pattern Formation on a Sphere

Simulates reaction-diffusion (Gray-Scott model) on a spherical membrane
using the Laplace-Beltrami operator discretized via cotangent weights.

This models intracellular protein patterning on curved cell membranes,
such as Cdc42 polarization in yeast or PAR protein domains.

Gray-Scott equations:
    ∂u/∂t = Dᵤ Δu - uv² + F(1-u)
    ∂v/∂t = Dᵥ Δv + uv² - (F+k)v

Where Δ is the Laplace-Beltrami operator on the sphere.
"""

import json
import argparse
import time
from pathlib import Path

try:
    import numpy as np
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import spsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Install with: pip install scipy")


def create_icosphere(subdivisions=4):
    """
    Create an icosphere mesh by subdividing an icosahedron.
    Returns vertices (Nx3) and faces (Mx3).
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Icosahedron vertices
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float64)

    # Normalize to unit sphere
    vertices /= np.linalg.norm(vertices[0])

    # Icosahedron faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)

    # Subdivide
    for _ in range(subdivisions):
        vertices, faces = subdivide_icosphere(vertices, faces)

    return vertices, faces


def subdivide_icosphere(vertices, faces):
    """Subdivide each triangle into 4 triangles."""
    edge_midpoints = {}
    new_faces = []

    def get_midpoint(i1, i2):
        key = (min(i1, i2), max(i1, i2))
        if key in edge_midpoints:
            return edge_midpoints[key]

        mid = (vertices[i1] + vertices[i2]) / 2
        mid = mid / np.linalg.norm(mid)  # Project to sphere

        idx = len(vertices)
        edge_midpoints[key] = idx
        return idx

    vertices = list(vertices)

    for face in faces:
        v0, v1, v2 = face

        # Get midpoints
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)

        # Add new vertices
        while len(vertices) <= max(m01, m12, m20):
            key = [k for k, v in edge_midpoints.items() if v == len(vertices)][0]
            mid = (vertices[key[0]] + vertices[key[1]]) / 2
            mid = mid / np.linalg.norm(mid)
            vertices.append(mid)

        # Create 4 new faces
        new_faces.extend([
            [v0, m01, m20],
            [v1, m12, m01],
            [v2, m20, m12],
            [m01, m12, m20]
        ])

    return np.array(vertices), np.array(new_faces)


def compute_cotangent_laplacian(vertices, faces):
    """
    Compute the cotangent Laplacian (Laplace-Beltrami discretization).

    The cotangent formula gives weights:
        w_ij = (cot(α_ij) + cot(β_ij)) / 2

    Where α and β are the angles opposite to edge (i,j) in the two
    triangles sharing that edge.

    Returns sparse Laplacian matrix L where L @ f gives Δf.
    """
    n = len(vertices)

    # Build adjacency and cotangent weights
    row_indices = []
    col_indices = []
    values = []

    # For each face, compute cotangent contributions
    for face in faces:
        for i in range(3):
            # Vertices of the triangle
            i0 = face[i]
            i1 = face[(i + 1) % 3]
            i2 = face[(i + 2) % 3]

            v0 = vertices[i0]
            v1 = vertices[i1]
            v2 = vertices[i2]

            # Edge vectors from v0
            e1 = v1 - v0
            e2 = v2 - v0

            # Cotangent of angle at v0
            cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10)
            cos_angle = np.clip(cos_angle, -0.999, 0.999)
            sin_angle = np.sqrt(1 - cos_angle ** 2)
            cot_angle = cos_angle / (sin_angle + 1e-10)

            # This cotangent contributes to edge (i1, i2)
            weight = cot_angle / 2

            row_indices.extend([i1, i2])
            col_indices.extend([i2, i1])
            values.extend([weight, weight])

    # Create sparse matrix
    L = csr_matrix((values, (row_indices, col_indices)), shape=(n, n))

    # Diagonal: negative sum of row
    diag_values = -np.array(L.sum(axis=1)).flatten()
    L = L + diags(diag_values)

    return L


def compute_vertex_areas(vertices, faces):
    """Compute the Voronoi area associated with each vertex."""
    n = len(vertices)
    areas = np.zeros(n)

    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        # Triangle area
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        # Distribute equally to vertices (barycentric)
        for idx in face:
            areas[idx] += area / 3

    return areas


class GrayScottSphere:
    """
    Gray-Scott reaction-diffusion on a spherical mesh.

    Parameters:
    - Du: Diffusion rate of U (substrate)
    - Dv: Diffusion rate of V (catalyst)
    - F: Feed rate (replenishes U)
    - k: Kill rate (removes V)
    """

    def __init__(self, vertices, faces, Du=0.16, Dv=0.08, F=0.035, k=0.065):
        self.vertices = vertices
        self.faces = faces
        self.n = len(vertices)

        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k

        # Compute Laplace-Beltrami operator
        print("Computing Laplace-Beltrami operator...")
        self.L = compute_cotangent_laplacian(vertices, faces)
        self.areas = compute_vertex_areas(vertices, faces)

        # Mass matrix (diagonal with vertex areas)
        self.M_inv = diags(1.0 / (self.areas + 1e-10))

        # Initialize concentrations
        self.u = np.ones(self.n)  # U starts at 1
        self.v = np.zeros(self.n)  # V starts at 0

        # Add random perturbation to seed pattern
        self._add_initial_perturbation()

    def _add_initial_perturbation(self):
        """Add localized perturbations to break symmetry."""
        # Add several spots of V
        n_spots = 5
        for _ in range(n_spots):
            # Random vertex
            center_idx = np.random.randint(self.n)
            center = self.vertices[center_idx]

            # Find nearby vertices
            distances = np.linalg.norm(self.vertices - center, axis=1)
            mask = distances < 0.3  # Spot radius

            # Perturb
            self.u[mask] = 0.5 + 0.1 * np.random.random(np.sum(mask))
            self.v[mask] = 0.25 + 0.1 * np.random.random(np.sum(mask))

    def step(self, dt=1.0):
        """Advance simulation by one time step using explicit Euler."""
        # Laplacian of concentrations (mass-weighted)
        Lu = self.M_inv @ (self.L @ self.u)
        Lv = self.M_inv @ (self.L @ self.v)

        # Reaction terms
        uvv = self.u * self.v * self.v

        # Gray-Scott equations
        du = self.Du * Lu - uvv + self.F * (1 - self.u)
        dv = self.Dv * Lv + uvv - (self.F + self.k) * self.v

        # Update
        self.u += dt * du
        self.v += dt * dv

        # Clamp to valid range
        self.u = np.clip(self.u, 0, 1)
        self.v = np.clip(self.v, 0, 1)

    def run(self, steps=5000, dt=1.0, save_interval=100):
        """Run simulation and save frames."""
        frames = []
        start_time = time.time()

        for i in range(steps):
            self.step(dt)

            if i % save_interval == 0:
                frames.append({
                    'step': i,
                    'u': self.u.copy().tolist(),
                    'v': self.v.copy().tolist()
                })
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"Step {i}/{steps} ({rate:.1f} steps/sec)")

        # Save final frame
        frames.append({
            'step': steps,
            'u': self.u.copy().tolist(),
            'v': self.v.copy().tolist()
        })

        return frames, time.time() - start_time


# Preset parameters for different pattern types
PRESETS = {
    'spots': {
        'description': 'Isolated spots (like Cdc42 polarization)',
        'Du': 0.16, 'Dv': 0.08, 'F': 0.035, 'k': 0.065,
        'steps': 8000, 'dt': 1.0
    },
    'stripes': {
        'description': 'Stripes/labyrinth patterns',
        'Du': 0.16, 'Dv': 0.08, 'F': 0.04, 'k': 0.06,
        'steps': 10000, 'dt': 1.0
    },
    'maze': {
        'description': 'Dense maze/coral patterns',
        'Du': 0.16, 'Dv': 0.08, 'F': 0.029, 'k': 0.057,
        'steps': 15000, 'dt': 1.0
    },
    'mitosis': {
        'description': 'Splitting/mitotic patterns',
        'Du': 0.16, 'Dv': 0.08, 'F': 0.037, 'k': 0.06,
        'steps': 8000, 'dt': 1.0
    },
    'waves': {
        'description': 'Propagating waves (like Min oscillations)',
        'Du': 0.12, 'Dv': 0.08, 'F': 0.014, 'k': 0.054,
        'steps': 10000, 'dt': 1.0, 'save_interval': 150
    },
    'flowing': {
        'description': 'Smooth organic flowing patterns',
        'Du': 0.18, 'Dv': 0.09, 'F': 0.025, 'k': 0.055,
        'steps': 12000, 'dt': 0.8, 'save_interval': 120
    },
    'plasma': {
        'description': 'Dynamic plasma-like undulations',
        'Du': 0.14, 'Dv': 0.06, 'F': 0.022, 'k': 0.051,
        'steps': 15000, 'dt': 0.9, 'save_interval': 100
    },
    'bouba': {
        'description': 'Soft rounded blob patterns (smooth curves)',
        'Du': 0.25, 'Dv': 0.12, 'F': 0.042, 'k': 0.067,
        'steps': 8000, 'dt': 0.6, 'save_interval': 100
    }
}


def main():
    if not HAS_SCIPY:
        print("Error: scipy is required. Install with: pip install scipy")
        return

    parser = argparse.ArgumentParser(description="Turing Patterns on Sphere")
    parser.add_argument(
        '--preset',
        choices=list(PRESETS.keys()),
        default='spots',
        help='Pattern preset'
    )
    parser.add_argument('--subdivisions', type=int, default=5,
                        help='Icosphere subdivision level (4-6)')
    parser.add_argument('--steps', type=int, help='Override simulation steps')
    parser.add_argument('--output', default='output/turing_sphere.json',
                        help='Output file path')

    args = parser.parse_args()

    preset = PRESETS[args.preset]
    print(f"Using preset: {args.preset} - {preset['description']}")

    # Create mesh
    print(f"\nCreating icosphere (subdivisions={args.subdivisions})...")
    vertices, faces = create_icosphere(args.subdivisions)
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")

    # Create simulation
    print("\nInitializing Gray-Scott simulation...")
    sim = GrayScottSphere(
        vertices, faces,
        Du=preset['Du'],
        Dv=preset['Dv'],
        F=preset['F'],
        k=preset['k']
    )

    # Run
    steps = args.steps or preset['steps']
    print(f"\nRunning {steps} steps...")
    save_interval = preset.get('save_interval', 200)  # More frames for smoother animation
    frames, elapsed = sim.run(steps=steps, dt=preset['dt'], save_interval=save_interval)

    # Export
    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'metadata': {
            'preset': args.preset,
            'description': preset['description'],
            'num_vertices': len(vertices),
            'num_faces': len(faces),
            'steps': steps,
            'Du': preset['Du'],
            'Dv': preset['Dv'],
            'F': preset['F'],
            'k': preset['k'],
            'generation_time': round(time.time(), 3)
        },
        'vertices': vertices.tolist(),
        'faces': faces.tolist(),
        'frames': frames
    }

    with open(output_path, 'w') as f:
        json.dump(data, f)

    print(f"\nSimulation complete in {elapsed:.2f} seconds")
    print(f"Exported to {output_path}")
    print(f"  {len(frames)} frames saved")


if __name__ == "__main__":
    main()
