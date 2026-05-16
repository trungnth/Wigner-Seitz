# LAMMPS Documentation: `compute ws`

## Syntax

```lammps
compute ID group-ID ws
```

- `ID`, `group-ID` are documented in the `compute` command.
- `ws` = name of this compute style.

---

## Examples

```lammps
compute 1 all ws

dump defects all custom 1000 dump.defects id type x y z c_1[1] c_1[2] c_1[3] c_1[4]

dump vacs all ws/vac 1000 dump.vacs c_1[1] c_1[2] c_1[3] c_1[4] c_1[5]

thermo_style custom step temp c_1[1] c_1[2] c_1[3] c_1[4]
```

---

## Description

The `compute ws` command identifies and classifies point defects on-the-fly using exact Wigner-Seitz cell analysis via the Voro++ library.

The compute evaluates the system upon its first invocation (typically via a `run 0` command) and stores the perfect pristine lattice coordinates, atom IDs, and atom types into a permanent reference memory. During subsequent timesteps, the compute maps the coordinates of currently displaced atoms onto the static reference cells.

The compute functions polymorphically, simultaneously generating three distinct data streams depending on the calling command.

---

## 1. Per-Atom Array (Displaced Atom Data)

Accessed by atom-style commands such as `dump custom` or `variable atom`. It can be used to define a LAMMPS dynamic group to track these specific atoms on-the-fly.

Produces a 4-column array:

| Column | Description |
|---|---|
| `c_ID[1]` | **Occupancy** - Total number of current atoms residing in the mapped reference cell. Values `>= 2` indicate an interstitial or co-location cluster. |
| `c_ID[2]` | **Site Index** - Internal memory index of the reference cell. |
| `c_ID[3]` | **Site Identifier** - Original Atom ID that defined this geometric site at Step 0. Comparing `id != c_ID[3]` identifies mixing/replacement atoms. |
| `c_ID[4]` | **Site Type** - Original element type of the site. Comparing `type != c_ID[4]` identifies anti-site defects. |

---

## 2. Local Array (Vacancy Dummy Atoms)

Accessed by local-style commands.

To correctly visualize vacancies as physical particles in visualization tools, it is highly recommended to use the custom `dump ws/vac` style command.

Produces a 5-column local array ordered for standard particle visualization:

| Column | Description |
|---|---|
| `c_ID[1]` | Original ID of the atom in reference lattice that previously occupied this vacancy. |
| `c_ID[2]` | Original type of the atom in reference lattice that previously occupied this vacancy. |
| `c_ID[3]` | X coordinate of the vacancy. |
| `c_ID[4]` | Y coordinate of the vacancy. |
| `c_ID[5]` | Z coordinate of the vacancy. |

---

## 3. Global Vector (Statistical Data)

Accessed by global-style commands such as `thermo_style`.

Outputs system-wide defect totals integrated across all MPI processors.

| Column | Description |
|---|---|
| `c_ID[1]` | Total interstitials in the simulation at current timestep. Includes lattice interstitial defects and atoms sputtered in case of sputtering simulation with open surface. |
| `c_ID[2]` | Total vacancies in the simulation at current timestep. |
| `c_ID[3]` | Total replacements (mixing atoms) in the simulation at current timestep. |
| `c_ID[4]` | Total antisites defects (chemical mixing in multi-element materials) in the simulation at current timestep.

---

## Restrictions

- This compute requires the **VORONOI** package to be installed.
- Currently supports only:
  - 3D periodic boundaries
  - fixed simulation boundaries
- Load balancing (`fix balance`) that alters MPI domain boundaries after Step 0 is **not supported**, because it breaks the static ghost-atom reference maps.

---


# Installation Instructions

This custom compute and dump style relies on the exact polyhedral cell algorithms provided by the `voro++` library. Therefore, although it does not require `VORONOI` package code to function, the simplest and most straightforward way to compile the custom `compute ws` style is to build LAMMPS with the `VORONOI` package enabled.


## Prerequisites: Positioning the Source Files

Before beginning either build process, place the custom compute and dump source files into the LAMMPS source tree.

Copy the following four files into the `src/VORONOI` directory of your LAMMPS installation:

- `compute_ws.cpp`
- `compute_ws.h`
- `dump_ws_vac.cpp`
- `dump_ws_vac.h`

---

## CMake Build (Recommended)

CMake is the primary and recommended build system for modern LAMMPS installations. It handles package dependencies and the downloading of external libraries automatically. From LAMMPS version 10Sep2025, the VORONOI package no longer supports the traditional make build. You need to build LAMMPS with CMake. 

### 1. Create a build directory

It is best practice to compile LAMMPS in a separate directory outside of the main source tree.

```bash
cd lammps
mkdir build
cd build
```

### 2. Configure the build environment

Enable the `VORONOI` package, use the `DOWNLOAD_VORO=yes` flag to instruct CMake to automatically fetch and statically link the correct version of the library.

```bash
cmake -D PKG_VORONOI=ON -D DOWNLOAD_VORO=yes -D BUILD_MPI=ON ../cmake
```

You can append any additional CMake flags here depending on your required configuration. Below is an example of building LAMMPS with the custom `compute ws` style for a DGX node equipped with NVIDIA V100 GPUs on the HYBRILIT computing platform at MLIT using the `most.cmake` preset.

```bash
module load gcc/v12.3.0 cuda/v12.8 openmpi/v4.1.8_gcc1230 CMake/v4.2.3 LAPACK/v3.12.0_gcc1230
cmake -C ../cmake/presets/most.cmake -D BUILD_MPI=ON -D PKG_GPU=ON -D GPU_API=cuda -D GPU_ARCH=sm_70 -D PKG_OPENMP=ON -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ ../cmake
```
### 3. Compile the executable

Compile the code using multiple threads to speed up the process.

```bash
make -j 8
```

Upon completion, the `lmp` executable will be generated in your `build` directory.
