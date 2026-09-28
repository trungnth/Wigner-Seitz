# `compute ws` command for LAMMPS

A custom LAMMPS compute command for on-the-fly **Wigner-Seitz (WS) defect analysis**, identifying and classifying point defects such as vacancies, interstitials, replacements, and antisites.

## Table of Contents

- [Syntax](#syntax)
- [Examples](#examples)
- [Description](#description)
  - [Classification of Defects](#classification-of-defects)
  - [Voronoi Container and Skin](#voronoi-container-and-skin)
- [Output Information](#output-information)
  - [1. Global Vector (Defect Tallies)](#1-global-vector-defect-tallies)
  - [2. Per-Atom Array (Site Properties)](#2-per-atom-array-site-properties)
  - [3. Local Array (Vacancy Coordinates)](#3-local-array-vacancy-coordinates)
- [Installation Instructions](#installation-instructions)
  - [Prerequisites: Positioning Source Files](#prerequisites-positioning-the-source-files)
  - [CMake Build (Recommended)](#cmake-build-recommended)
- [Restrictions](#restrictions)
- [Related Commands](#related-commands)
- [Default](#default)


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

## Installation Instructions

This custom compute relies on the exact polyhedral cell algorithms provided by the `voro++` library. Therefore the simplest and most straightforward way to compile the custom `compute ws` style is to build LAMMPS with the `VORONOI` package enabled.

### Prerequisites: Positioning the Source Files

Before beginning the build process, place the custom compute source files into the LAMMPS source tree.

Copy the following source files into the `src/VORONOI` directory of your LAMMPS installation:

- `compute_ws.cpp`
- `compute_ws.h`

---

### CMake Build (Recommended)

CMake is the primary and recommended build system for modern LAMMPS installations. It handles package dependencies and the downloading of external libraries automatically. 

> **Notice:** Starting with LAMMPS version **10Sep2025**, the `VORONOI` package no longer supports traditional GNU make builds. You must build LAMMPS using CMake.

#### 1. Create a build directory

It is best practice to compile LAMMPS in a separate directory outside of the main source tree:

```bash
cd lammps
mkdir build
cd build
```

#### 2. Configure the build environment

Enable the `VORONOI` package, and use the `-D DOWNLOAD_VORO=yes` flag to instruct CMake to automatically fetch and statically link the correct version of the `voro++` library:

```bash
cmake -D PKG_VORONOI=ON -D DOWNLOAD_VORO=yes -D BUILD_MPI=ON ../cmake
```

You can append any additional CMake flags depending on your required configuration. Below is an example of building LAMMPS with `compute ws` for a DGX node equipped with NVIDIA V100 GPUs on the HYBRILIT computing platform at MLIT, JINR using the `most.cmake` preset:

```bash
module load gcc/v12.3.0 cuda/v12.8 openmpi/v4.1.8_gcc1230 CMake/v4.2.3 LAPACK/v3.12.0_gcc1230

cmake -C ../cmake/presets/most.cmake \
      -D PKG_VORONOI=ON \
      -D DOWNLOAD_VORO=yes \
      -D BUILD_MPI=ON \
      -D PKG_GPU=ON \
      -D GPU_API=cuda \
      -D GPU_ARCH=sm_70 \
      -D PKG_OPENMP=ON \
      -D CMAKE_C_COMPILER=gcc \
      -D CMAKE_CXX_COMPILER=g++ \
      ../cmake
```

#### 3. Compile the executable

Compile the code using multiple threads to accelerate the build:

```bash
make -j 8
```

Upon successful completion, the `lmp` executable will be generated in your `build` directory.

---

## Restrictions

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
