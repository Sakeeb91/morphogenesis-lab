#!/usr/bin/env python3
"""
Laplacian Growth Simulation Engine

Implements the Fuse Breakdown / Laplacian Growth algorithm for generating
fractal-like growth patterns (neurons, lightning, moss, cities).

Key Equations:
- Phi = Sum(1 - R1/r) for aggregate particles (Eqn 10)
- Incremental update: Phi_new = Phi_old + (1 - R1/r_new) (Eqn 11)
- Selection probability: P(i) ∝ Phi(i)^ETA (Eqn 12)

Optimized with numpy for vectorized distance calculations.
"""

import json
import argparse
import time
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    import math
    import random


class LaplacianGrowthNumpy:
    """
    Numpy-optimized Laplacian growth simulation.
    """

    def __init__(
        self,
        num_particles=3500,
        eta=4.0,
        r1=0.5,
        mode_2d=False,
        attractors=None,
        seeds=None
    ):
        self.num_particles = num_particles
        self.eta = eta
        self.r1 = r1
        self.mode_2d = mode_2d
        self.attractors = np.array([
            [a['x'], a['y'], a['z'], a.get('strength', 50.0)]
            for a in (attractors or [])
        ]) if attractors else np.array([]).reshape(0, 4)

        # Pre-generate neighbor offsets
        self.offsets = self._generate_offsets()

        # Initialize particles array
        seeds = seeds or [[0, 0, 0]]
        self.particles = np.array(seeds, dtype=np.float64)

        # Initialize candidates
        self._init_candidates()

    def _generate_offsets(self):
        """Generate neighbor offsets based on 2D/3D mode."""
        spacing = self.r1 * 2
        offsets = []

        if self.mode_2d:
            for dx in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dz == 0:
                        continue
                    offsets.append([dx * spacing, 0, dz * spacing])
        else:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        offsets.append([dx * spacing, dy * spacing, dz * spacing])

        return np.array(offsets)

    def _init_candidates(self):
        """Initialize candidates around seed particles."""
        all_candidates = set()

        for p in self.particles:
            new_cands = p + self.offsets
            for c in new_cands:
                c_tuple = tuple(c)
                if not self._overlaps_any(c):
                    all_candidates.add(c_tuple)

        self.candidates = np.array(list(all_candidates))
        self.candidate_potentials = self._calculate_potentials(self.candidates)

    def _distances(self, points1, points2):
        """
        Calculate pairwise distances between two sets of points.
        Returns matrix of shape (len(points1), len(points2))
        """
        if len(points1) == 0 or len(points2) == 0:
            return np.array([]).reshape(len(points1), len(points2))

        # Broadcasting: points1[:, np.newaxis] is (N1, 1, 3), points2 is (N2, 3)
        diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def _calculate_potentials(self, candidates):
        """Calculate potentials for all candidates using vectorized operations."""
        if len(candidates) == 0:
            return np.array([])

        # Distance from each candidate to each particle
        dists = self._distances(candidates, self.particles)  # (N_cand, N_particles)

        # Potential contribution: max(0, 1 - r1/r)
        with np.errstate(divide='ignore', invalid='ignore'):
            contributions = np.maximum(0, 1 - self.r1 / dists)
            contributions = np.nan_to_num(contributions, nan=0.0, posinf=0.0)

        potentials = np.sum(contributions, axis=1)

        # Add attractor contributions
        if len(self.attractors) > 0:
            attr_dists = self._distances(candidates, self.attractors[:, :3])
            attr_contributions = self.attractors[:, 3] / (attr_dists + 1)
            potentials += np.sum(attr_contributions, axis=1)

        return potentials

    def _overlaps_any(self, point):
        """Check if a point overlaps with any existing particle."""
        if len(self.particles) == 0:
            return False
        dists = np.sqrt(np.sum((self.particles - point) ** 2, axis=1))
        return np.any(dists < self.r1 * 1.8)

    def _select_candidate(self):
        """Select next growth site using probability P(i) ∝ Phi(i)^ETA."""
        if len(self.candidates) == 0:
            return None

        if self.eta == 0:
            idx = np.random.randint(len(self.candidates))
        else:
            # Normalize and apply eta exponent
            max_phi = np.max(self.candidate_potentials)
            normalized = self.candidate_potentials / (max_phi + 1e-10)
            weights = np.power(normalized, self.eta)
            weights = np.maximum(weights, 1e-10)

            # Weighted random selection
            probs = weights / np.sum(weights)
            idx = np.random.choice(len(self.candidates), p=probs)

        return idx

    def _add_particle(self, idx):
        """Add the selected candidate as a new particle."""
        new_particle = self.candidates[idx].copy()
        self.particles = np.vstack([self.particles, new_particle])

        # Remove selected candidate
        mask = np.ones(len(self.candidates), dtype=bool)
        mask[idx] = False
        self.candidates = self.candidates[mask]
        self.candidate_potentials = self.candidate_potentials[mask]

        # Update existing candidate potentials (incremental)
        if len(self.candidates) > 0:
            dists = np.sqrt(np.sum((self.candidates - new_particle) ** 2, axis=1))
            with np.errstate(divide='ignore', invalid='ignore'):
                delta = np.maximum(0, 1 - self.r1 / dists)
                delta = np.nan_to_num(delta, nan=0.0, posinf=0.0)
            self.candidate_potentials += delta

        # Generate new candidates around new particle
        new_cands = new_particle + self.offsets

        # Filter out overlapping and existing candidates
        valid_new = []
        existing = set(map(tuple, self.candidates)) if len(self.candidates) > 0 else set()

        for c in new_cands:
            c_tuple = tuple(c)
            if c_tuple not in existing and not self._overlaps_any(c):
                valid_new.append(c)

        if valid_new:
            valid_new = np.array(valid_new)
            new_potentials = self._calculate_potentials(valid_new)

            if len(self.candidates) > 0:
                self.candidates = np.vstack([self.candidates, valid_new])
                self.candidate_potentials = np.concatenate([
                    self.candidate_potentials, new_potentials
                ])
            else:
                self.candidates = valid_new
                self.candidate_potentials = new_potentials

        # Remove candidates that now overlap with the new particle
        if len(self.candidates) > 0:
            dists = np.sqrt(np.sum((self.candidates - new_particle) ** 2, axis=1))
            mask = dists >= self.r1 * 1.8
            self.candidates = self.candidates[mask]
            self.candidate_potentials = self.candidate_potentials[mask]

    def grow(self):
        """Run the growth simulation."""
        start_time = time.time()

        while len(self.particles) < self.num_particles:
            idx = self._select_candidate()

            if idx is None:
                print(f"Warning: No candidates at {len(self.particles)} particles")
                break

            self._add_particle(idx)

            if len(self.particles) % 100 == 0:
                elapsed = time.time() - start_time
                rate = len(self.particles) / elapsed if elapsed > 0 else 0
                print(f"Particles: {len(self.particles)}/{self.num_particles} "
                      f"({rate:.1f}/sec, {len(self.candidates)} candidates)")

        return time.time() - start_time

    def export_json(self, filepath, preset_name="custom"):
        """Export simulation data to JSON."""
        data = {
            "metadata": {
                "preset": preset_name,
                "num_particles": len(self.particles),
                "eta": self.eta,
                "r1": self.r1,
                "mode_2d": self.mode_2d,
                "generation_time": round(time.time(), 3)
            },
            "particles": self.particles.tolist(),
            "attractors": [
                {"x": float(a[0]), "y": float(a[1]), "z": float(a[2]), "strength": float(a[3])}
                for a in self.attractors
            ] if len(self.attractors) > 0 else []
        }

        with open(filepath, 'w') as f:
            json.dump(data, f)

        print(f"Exported {len(self.particles)} particles to {filepath}")


# Fallback pure Python implementation
class LaplacianGrowthPure:
    """Pure Python implementation (slower, used when numpy unavailable)."""

    def __init__(self, num_particles=3500, eta=4.0, r1=0.5, mode_2d=False,
                 attractors=None, seeds=None):
        self.num_particles = num_particles
        self.eta = eta
        self.r1 = r1
        self.mode_2d = mode_2d
        self.attractors = attractors or []
        self.particles = []
        self.candidates = {}

        seeds = seeds or [[0, 0, 0]]
        for seed in seeds:
            self._add_particle(seed[0], seed[1], seed[2])

    def _distance(self, p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _overlaps(self, pos):
        for p in self.particles:
            if self._distance(pos, p) < self.r1 * 1.8:
                return True
        return False

    def _potential(self, pos):
        phi = 0
        for p in self.particles:
            r = self._distance(pos, p)
            if r > 0:
                phi += max(0, 1 - self.r1 / r)
        for a in self.attractors:
            r = self._distance(pos, (a['x'], a['y'], a['z']))
            if r > 0:
                phi += a.get('strength', 50) / (r + 1)
        return phi

    def _get_offsets(self):
        s = self.r1 * 2
        if self.mode_2d:
            return [(dx * s, 0, dz * s) for dx in [-1, 0, 1] for dz in [-1, 0, 1]
                    if not (dx == 0 and dz == 0)]
        return [(dx * s, dy * s, dz * s)
                for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]
                if not (dx == 0 and dy == 0 and dz == 0)]

    def _add_particle(self, x, y, z):
        new_p = [x, y, z]
        self.particles.append(new_p)

        key = (x, y, z)
        if key in self.candidates:
            del self.candidates[key]

        for c in list(self.candidates):
            delta = max(0, 1 - self.r1 / self._distance(c, new_p))
            self.candidates[c] += delta

        for o in self._get_offsets():
            nc = (x + o[0], y + o[1], z + o[2])
            if nc not in self.candidates and not self._overlaps(nc):
                self.candidates[nc] = self._potential(nc)

        for c in list(self.candidates):
            if self._distance(c, new_p) < self.r1 * 1.8:
                del self.candidates[c]

    def _select(self):
        if not self.candidates:
            return None
        items = list(self.candidates.items())
        if self.eta == 0:
            return random.choice(items)[0]
        max_phi = max(p for _, p in items)
        weights = [(p / (max_phi + 1e-10)) ** self.eta for _, p in items]
        total = sum(weights)
        r = random.random() * total
        cum = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return items[i][0]
        return items[-1][0]

    def grow(self):
        start = time.time()
        while len(self.particles) < self.num_particles:
            sel = self._select()
            if sel is None:
                break
            self._add_particle(sel[0], sel[1], sel[2])
            if len(self.particles) % 100 == 0:
                elapsed = time.time() - start
                rate = len(self.particles) / elapsed if elapsed > 0 else 0
                print(f"Particles: {len(self.particles)}/{self.num_particles} "
                      f"({rate:.1f}/sec, {len(self.candidates)} candidates)")
        return time.time() - start

    def export_json(self, filepath, preset_name="custom"):
        data = {
            "metadata": {
                "preset": preset_name,
                "num_particles": len(self.particles),
                "eta": self.eta,
                "r1": self.r1,
                "mode_2d": self.mode_2d,
                "generation_time": round(time.time(), 3)
            },
            "particles": self.particles,
            "attractors": self.attractors
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Exported {len(self.particles)} particles to {filepath}")


# Choose implementation based on numpy availability
LaplacianGrowth = LaplacianGrowthNumpy if HAS_NUMPY else LaplacianGrowthPure


def load_presets():
    """Load preset configurations from presets.json."""
    presets_path = Path(__file__).parent / "presets.json"
    if presets_path.exists():
        with open(presets_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Laplacian Growth Simulation")
    parser.add_argument(
        "--preset",
        choices=["lightning", "neurons", "moss", "cities", "custom"],
        default="neurons",
        help="Preset configuration to use"
    )
    parser.add_argument("--particles", type=int, help="Override number of particles")
    parser.add_argument("--eta", type=float, help="Override eta value")
    parser.add_argument("--mode-2d", action="store_true", help="Enable 2D mode")
    parser.add_argument(
        "--output",
        default="output/growth_data.json",
        help="Output file path"
    )

    args = parser.parse_args()

    presets = load_presets()

    if args.preset in presets:
        config = presets[args.preset]
        print(f"Using preset: {args.preset} - {config.get('description', '')}")
    else:
        config = {
            "num_particles": 3500,
            "eta": 4.0,
            "r1": 0.5,
            "mode_2d": False,
            "seeds": [[0, 0, 0]],
            "attractors": []
        }

    if args.particles:
        config["num_particles"] = args.particles
    if args.eta is not None:
        config["eta"] = args.eta
    if args.mode_2d:
        config["mode_2d"] = True

    impl = "numpy" if HAS_NUMPY else "pure Python (slower)"
    print(f"\nUsing {impl} implementation")
    print(f"\nStarting simulation:")
    print(f"  Particles: {config['num_particles']}")
    print(f"  ETA: {config['eta']}")
    print(f"  R1: {config['r1']}")
    print(f"  2D Mode: {config['mode_2d']}")
    print(f"  Attractors: {len(config.get('attractors', []))}")
    print()

    sim = LaplacianGrowth(
        num_particles=config["num_particles"],
        eta=config["eta"],
        r1=config["r1"],
        mode_2d=config["mode_2d"],
        attractors=config.get("attractors", []),
        seeds=config.get("seeds", [[0, 0, 0]])
    )

    elapsed = sim.grow()

    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sim.export_json(output_path, args.preset)

    print(f"\nSimulation complete in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
