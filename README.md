# `compute ws` command for LAMMPS

A custom LAMMPS compute command for on-the-fly **Wigner-Seitz (WS) defect analysis**, identifying and classifying point defects such as vacancies, interstitials, replacements, and antisites.

## Table of Contents

* [Syntax](#syntax)
* [Examples](#examples)
* [Description](#description)
  * [Classification of Defects](#classification-of-defects)
  * [Voronoi Container and Skin Buffer](#voronoi-container-and-skin-buffer)
* [Output Information](#output-information)
  * [Global Vector (Defect Tallies)](#1-global-vector-defect-tallies)
  * [Per-Atom Array (Site Properties)](#2-per-atom-array-site-properties)
  * [Local Array (Vacancy Coordinates)](#3-local-array-vacancy-coordinates)
* [Prerequisites & Restrictions](#prerequisites--restrictions)
* [Related Commands](#related-commands)
* [Default Values](#default-values)

---

## Syntax

```lammps
compute ID group-ID ws
compute ID group-ID ws skin
```

* **`ID`**, **`group-ID`**: Documented in the standard LAMMPS [`compute`](https://docs.lammps.org/compute.html) command.
* **`ws`**: Style name of this compute command.
* **`skin`**: *(Optional)* Voronoi container expansion buffer (distance units).

---

## Examples

```lammps
# 1. Default usage (automatically synchronizes with LAMMPS neighbor skin)
compute 1 all ws

# 2. Manual skin usage (e.g., for sputtering or non-periodic boundary condition simulations)
compute 2 all ws 5.0

# Force the compute to evaluate and cache the reference lattice before dynamics
run 0 

# Output total defect tallies: [Interstitials, Vacancies, Replacements, Antisites]
thermo_style custom step pe ke c_1[1] c_1[2] c_1[3] c_1[4]

# Per-atom dump for WS site properties (occupancy, site index, site ID, site type)
dump 1 all custom 100 dump.atoms.lammpstrj id type x y z c_1[1] c_1[2] c_1[3] c_1[4]

# Local dump for reference coordinates and properties of identified vacancies
dump 2 all local 100 dump.vacancies.lammpstrj c_1[1] c_1[2] c_1[3] c_1[4] c_1[5]
```

---

## Description

The `compute ws` command defines a computation that performs Wigner-Seitz (WS) defect analysis to identify and classify point defects such as vacancies, interstitials, replacements, and antisites on-the-fly.

This compute uses the [voro++](http://math.lbl.gov/voro++/) library to calculate the Voronoi polyhedra (Wigner-Seitz cells) based on a reference lattice.

### How It Works

1. **Initialization:** When initialized (e.g., at `run 0`), the compute captures and stores the current atomic positions and types as the **reference lattice** on their respective MPI processors.
2. **Defect Tracking:** In all subsequent timesteps, the compute maps the instantaneous atomic positions into the cached Wigner-Seitz cells of the reference lattice to determine WS site properties.

### Classification of Defects

Based on the occupancy and the types of atoms within each WS cell, the compute classifies defects into four categories:

* **Interstitials:** WS cells containing more than one atom. If a cell contains $N$ atoms, it contributes $N - 1$ to the total interstitial count.
* **Vacancies:** WS cells containing no atoms (empty sites, occupancy = $0$).
* **Replacements:** WS cells containing exactly one atom (occupancy = $1$), but with a different original ID than the reference atom.
* **Antisites:** WS cells containing exactly one atom (occupancy = $1$), but of a different atom type than the original reference atom.

### Voronoi Container and Skin Buffer

To construct mathematically rigorously bounded Voronoi cells, the compute creates a local container around the reference atoms:

* **Dynamic Neighbor Skin:** By default, this compute dynamically synchronizes with the LAMMPS [`neighbor`](https://docs.lammps.org/neighbor.html) skin distance to expand the container boundaries.
* **Manual Skin Override:** Users can optionally override the dynamic boundary by specifying a manual `skin` value.
* **Non-Periodic Boundaries:** The compute supports systems with non-periodic boundaries. Atoms that evaporate or are sputtered beyond the boundaries of the established Voronoi container are safely ignored.

---

## Output Information

This compute calculates a **global vector** of length 4, a **per-atom array** with 4 columns, and a **local array** with 5 columns.

### 1. Global Vector (Defect Tallies)

Values are accessible via index notation `c_ID[i]` ($1 \le i \le 4$), representing total defect counts across the system:

| Index | Name | Description |
| :---: | :--- | :--- |
| `c_ID[1]` | **Interstitials** | Total number of interstitial atoms |
| `c_ID[2]` | **Vacancies** | Total number of vacancies (empty sites) |
| `c_ID[3]` | **Replacements** | Total number of replacement collisions |
| `c_ID[4]` | **Antisites** | Total number of antisite defects |

### 2. Per-Atom Array (Site Properties)

Calculated for **every** atom in the compute group. 

* For atoms that remain in their original lattice sites, `Occupancy` is typically 1, and `SiteIdentifier` and `SiteType` match the atom's own ID and type.
* For atoms that have moved outside the reference Voronoi container, all values are set to `0`.

| Column | Property | Description |
| :---: | :--- | :--- |
| `c_ID[1]` | **Occupancy** | Number of atoms currently residing in the WS cell where this atom is located |
| `c_ID[2]` | **SiteIndex** | The MPI-local array index of the WS cell.<br>*(**Note:** This is an internal processor-specific index and is **not** globally unique. For post-processing and visualization, users should rely on `SiteIdentifier` (column 3) instead).* |
| `c_ID[3]` | **SiteIdentifier** | Original atom ID of the reference atom defining this WS cell |
| `c_ID[4]` | **SiteType** | Original atom type of the reference atom defining this WS cell |

### 3. Local Array (Vacancy Coordinates)

Outputs the reference coordinates and properties of identified **Vacancies** (empty WS cells):

| Column | Property | Description |
| :---: | :--- | :--- |
| `c_ID[1]` | **ID** | Reference atom ID of the vacancy site |
| `c_ID[2]` | **Type** | Reference atom type of the vacancy site |
| `c_ID[3]` | **x** | Reference $x$-coordinate of the vacancy site |
| `c_ID[4]` | **y** | Reference $y$-coordinate of the vacancy site |
| `c_ID[5]` | **z** | Reference $z$-coordinate of the vacancy site |

---

## Prerequisites & Restrictions

* **VORONOI Package:** This compute is part of the `VORONOI` package. It is only enabled if LAMMPS was built with that package. See the [LAMMPS Build package](https://docs.lammps.org/Build_package.html) page for build instructions.
* **Static Processor Domain:** Assumes a static processor domain and simulation box after initialization.
  * It **cannot** be used in simulations where the simulation box is deformed continuously (e.g., via [`fix deform`](https://docs.lammps.org/fix_deform.html)).
  * It **cannot** be used with dynamic load balancing (e.g., via [`fix balance`](https://docs.lammps.org/fix_balance.html)), because reference lattice coordinates are frozen on their original MPI ranks.
  * The compute will throw a runtime error if a simulation box change is detected.

---

## Related Commands

* [`compute voronoi/atom`](https://docs.lammps.org/compute_voronoi_atom.html)
* [`dump local`](https://docs.lammps.org/dump.html)
* [`compute`](https://docs.lammps.org/compute.html)

---

## Default Values

* **`skin`**: LAMMPS neighbor skin distance (with a minimum of $0.5$ distance units).
