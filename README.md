# compute ws: Wigner-Seitz defect analysis for LAMMPS

`compute ws` identifies point defects in LAMMPS simulations, such as
displacement cascades, by a Wigner-Seitz analysis relative to a reference
configuration. In one invocation it gives:

- the numbers of interstitials, vacancies, replacements and antisites;
- the occupancy, site ID and site type of every atom;
- the ID, type and position of every vacant site.

The reference is either the group atoms at the start of the run or a dump
file. With `affine yes` the box may change size and shape, e.g. under
NPT. With `drift yes` the compute follows a rigid translation of the
crystal relative to the reference, e.g. from the momentum of a primary
knock-on atom. The compute needs no external library. Between MPI
processes it exchanges only one sum of nine integer counters per
assignment of the atoms (one per invocation, a few with `drift yes`),
whatever the number of atoms. Without `drift yes`, its results agree
exactly with the Wigner-Seitz modifier of OVITO.

## Contents

```
src/EXTRA-COMPUTE/compute_ws.h, compute_ws.cpp   source code
doc/src/compute_ws.rst                            documentation page (LAMMPS format)
unittest/commands/test_compute_ws.cpp             unit tests (GoogleTest, 60 tests)
examples/PACKAGES/ws/                             example: displacement cascade in SiC
    in.ws                                         input script
    ws_sublattice.py                              per-type and sublattice statistics
    log.30Sep26.ws.g++.1, log.30Sep26.ws.g++.4    reference output with 1 and 4 MPI ranks
tools/ovito_compare.py                            comparison with OVITO for a frame
install.sh                                        installs everything into a LAMMPS source tree
LICENSE                                           GNU GPL v2, as LAMMPS
```

The files are laid out as in the LAMMPS source tree.

## Method

Each site of the reference configuration has a Wigner-Seitz cell: the
region of space closer to that site than to any other site, i.e. its
Voronoi cell. An atom inside the cell of a site occupies that site, so
each atom belongs to its nearest site. The number of atoms on site s is
its occupancy n_s.

| Site state   | Condition                                   | Counted as                                          |
|--------------|---------------------------------------------|-----------------------------------------------------|
| vacant       | n_s = 0                                     | 1 vacancy                                           |
| unchanged    | n_s = 1, the atom that defined the site     | nothing                                             |
| replaced     | n_s = 1, another atom of the site type      | 1 replacement                                       |
| antisite     | n_s = 1, an atom of another type            | 1 antisite, and 1 replacement if it is another atom |
| interstitial | n_s >= 2                                    | n_s - 1 interstitials                               |

Vacancies and interstitials follow the definitions of the Wigner-Seitz
analysis in OVITO. Replacements and antisites are counted on singly
occupied sites only.

The nearest site is found exactly; no cells are constructed:

- An atom closer than r_min/2 to a site lies inside that site's cell,
  because the sphere of radius r_min/2 fits inside the cell. Here r_min is
  the smallest distance between two sites.
- This test is applied first to the likely site of each atom.
- All other atoms are searched on a cell list of the sites.

Each MPI rank keeps the sites of its local and ghost atoms only. Defects
are classified on the sites it owns, and the counts are summed in one
`MPI_Allreduce` of nine 64-bit integer counters (72 bytes) per assignment,
whatever the number of atoms.

### Drift of the crystal (`drift yes`)

The crystal can move as a whole relative to the box: by the momentum of a
primary knock-on atom that is not compensated, by a thermostat that does
not conserve the total momentum, or by defects that migrate while the
center of mass stays fixed (Wang et al. 2024). With a fixed reference, a
drift by a sizable fraction of the distance between sites produces false
Frenkel pairs. With `drift yes`:

- The translation t is measured at every analysis as a weighted mean of
  the displacements of the atoms from their sites. An atom closer than
  r_min/4 to its site has weight 1, the weight falls smoothly to 0 at
  r_min/2, and farther atoms (interstitials, displaced atoms of a cascade)
  do not count. Such atoms certainly belong to their site, so t changes
  continuously with the positions.
- The atoms are assigned at x - t, starting from t of the previous
  analysis, and t is corrected until it changes by less than
  1e-6 r_min/2 (one assignment if the crystal has not moved, two or
  more if it has).
- The sums are 64-bit integers in fixed point, so t is the same on any
  number of MPI ranks, and a rigidly moved copy of the crystal gives the
  same result.
- t is added to the global vector, and vacancies are reported at the
  moved sites.

Without the keyword, every 16th atom (by ID) estimates t, only for a
warning that is printed once if t exceeds 25% of r_min/2.

## Requirements

- LAMMPS built with the EXTRA-COMPUTE package (tested on LAMMPS 30 Sep 2026 (stable release))
- For the example: the MANYBODY package (pair style `gw/zbl`).
- For `tools/ovito_compare.py`: Python 3 with NumPy and the OVITO Python
  module (`pip install ovito`).

## Installation

With the script:

```
git clone https://github.com/trungnth/Wigner-Seitz.git
bash Wigner-Seitz/install.sh /path/to/lammps
```

The script does the following:

- copies the files to their places in the LAMMPS tree;
- links the potential file of the example from `potentials/`;
- registers the unit test in `unittest/commands/CMakeLists.txt`;
- adds the compute to the lists in `doc/src/Commands_compute.rst` and
  `doc/src/compute.rst`, and to `src/.gitignore`.

Running it again only updates the files.

By hand: copy `src/EXTRA-COMPUTE/compute_ws.h` and `compute_ws.cpp` to
`src/EXTRA-COMPUTE/` of the LAMMPS tree. Nothing else is needed to use the
compute.

Then build LAMMPS with the package:

```
# CMake
cmake -S cmake -B build -D PKG_EXTRA-COMPUTE=on [other options]
cmake --build build

# legacy make
cd src
make yes-extra-compute
make mpi
```

If the package was already installed with legacy make, run
`make yes-extra-compute` again to copy the new files. With CMake, run
`cmake` again so that it finds the new files.

An earlier version of `compute ws`, based on the Voro++ library, used the
same style name in `src/VORONOI/`. Remove `src/VORONOI/compute_ws.h` and
`compute_ws.cpp` before installing this version.

## Usage

```
compute ID group-ID ws keyword value ...
```

Keywords (all optional):

- `reference file`: read the sites from the first snapshot of a dump
  file. The default is the group atoms at the start of the first run.
- `affine yes|no`: with `yes`, map the positions into the box of the
  reference before the analysis. The default is `no`, which requires the
  box of the reference.
- `drift yes|no`: with `yes`, follow a rigid translation of the crystal
  relative to the reference. The default is `no`.

Examples:

```
compute 1 all ws
compute 2 metal ws affine yes
compute 3 all ws reference ref.dump affine yes
compute 4 all ws drift yes
```

### Output

| Output                       | Values                                                        | Use in                                   |
|------------------------------|---------------------------------------------------------------|------------------------------------------|
| global vector (4, or 7)      | interstitials, vacancies, replacements, antisites; with `drift yes` also the translation tx, ty, tz | `thermo_style`, equal-style variables    |
| per-atom array (3 columns)   | occupancy of the atom's site, site ID, site type              | `dump custom`, atom-style variables      |
| local array (5 columns)      | site ID, site type, x, y, z of each vacant site (moved by the translation with `drift yes`) | `dump local`                             |

All per-atom values are 0 for atoms outside the group or not assigned to a
site. The global values are intensive: `thermo_modify norm` does not scale
them. If every group atom is assigned to a site,
N_int - N_vac = N_group - N_sites.

### Typical use during a cascade

```
compute         ws all ws drift yes
thermo_style    custom step temp pe c_ws[1] c_ws[2] c_ws[3] c_ws[4] c_ws[5] c_ws[6] c_ws[7]

# vacancies
dump            vac all local 1000 vac.*.dump index c_ws[1] c_ws[2] c_ws[3] c_ws[4] c_ws[5]

# only the atoms whose site differs from the reference
variable        def atom "c_ws[1] >= 2 || type != c_ws[3] || id != c_ws[2]"
dump            def all custom 1000 def.*.dump id type x y z c_ws[*]
dump_modify     def thresh v_def == 1
```

The two dumps together describe the occupancy of every site. Every site
missing from both is occupied by its original atom. Statistics per atom
type and per sublattice can therefore be done in post-processing; see
`examples/PACKAGES/ws/ws_sublattice.py`.

### Points to note

- The reference is stored at the start of the first run (or the first
  `rerun` snapshot) after the compute is defined. It should be a
  defect-free crystal. To store a new reference, define the compute again
  with the same ID (`uncompute`, then `compute`).
- A reference file written with `write_dump` should keep the full
  precision of the positions, e.g.
  `write_dump all custom ref.dump id type x y z modify format float %20.15g`.
  The site IDs should be the atom IDs of the simulation, so that
  replacements are meaningful.
- A restart file does not store the reference. A job started from a
  restart file uses the crystal at its start, unless it reads a reference
  file. With `drift yes`, the translation is stored in the restart file;
  a compute with the same ID that reads the same reference file continues
  with it.
- With `drift yes`:
  - the crystal must move by less than r_min/2 between two analyses, so
    invoke the compute often enough (LAMMPS warns above r_min/4);
  - the communication cutoff must exceed the cutoff needed for the sites
    (see below) by the accumulated translation (LAMMPS warns otherwise);
  - only a translation is followed, not a rotation or local motions of
    the lattice, e.g. near moving dislocations or grain boundaries.
- A crystal with extended defects, such as dislocations, should be
  referenced to its own relaxed initial configuration, not to a perfect
  lattice.
- Between runs, the output is that of the last analysis. After
  `create_atoms`, `delete_atoms` or `displace_atoms`, use `run 0` before
  using the compute again.
- Box:
  - Without `affine yes`, the box must not change.
  - With `affine yes`, the atoms must deform with the box: e.g. `fix npt`,
    or `fix deform` with `remap x`.
  - Box flips are not supported (use `flip no`).
  - Under continuous shear, invoke the compute often enough that a tilt
    changes by less than half a box length between two invocations.
- Processor sub-domains:
  - They must not change during a run, so `fix balance` is not allowed.
  - `balance` between runs is allowed only with a reference file.
  - Shrink-wrapped boundaries are not allowed.
- The communication cutoff must be at least the diameter of a sphere with
  the volume per site plus half the neighbor skin. This is about 1.1
  nearest-neighbor distances in fcc, hcp and bcc, and 1.4 in diamond.
  Pair style cutoffs usually suffice; LAMMPS warns otherwise.
- Replacements are detected by atom ID, so commands that renumber atoms
  (e.g. `delete_atoms` with `compress yes`) invalidate them.
- The compute requires a 3d system and atom IDs; dynamic groups are not
  allowed. A static group selects both sites and atoms, so one sublattice
  can be analyzed with a group by type.

The complete description is in `doc/src/compute_ws.rst`.

## Example

`examples/PACKAGES/ws/in.ws` follows a 200 eV Si recoil in 3C-SiC:

- 1000 atoms, pair style `gw/zbl`;
- a Berendsen thermostat on the border;
- `drift yes`, with the translation printed in the thermo output;
- about 4 s on one core.

It writes the vacancy and defect dumps of the workflow above at the first
and the last step.

```
cd examples/PACKAGES/ws                      # in the LAMMPS tree after install.sh
lmp -in in.ws
python3 ws_sublattice.py .
```

Outside the LAMMPS tree, set `LAMMPS_POTENTIALS` to the `potentials/`
directory of LAMMPS, so that `SiC.gw.zbl` is found. The setup is
deterministic: with 1 and 4 MPI ranks the defects are the same (5
interstitials, 5 vacancies, 3 replacements, 2 antisites at the end), and
the thermo output differs only by round-off.

The momentum of the primary knock-on atom is not removed in the example,
so the crystal drifts by 0.28 A along the direction of the recoil by the
last step. Without `drift yes` (`compute ws_map all ws`), the last step
has 6 interstitials, 6 vacancies, 3 replacements and 1 antisite: a few
atoms of the damaged region, near the boundary between two sites, are
assigned to the other site.

## Tests

```
cmake -S cmake -B build -D PKG_EXTRA-COMPUTE=on -D ENABLE_TESTING=on -D BUILD_SHARED_LIBS=on
cmake --build build --target lmp test_compute_ws
ctest --test-dir build -R ComputeWS -V
mpirun -np 4 build/test_compute_ws
```

The 60 tests check the following:

- every atom against a brute-force nearest-site search, in orthogonal,
  triclinic, `comm_style tiled` and non-periodic boxes;
- a known set of defects, with exact global, per-atom and local output;
- reference files, the affine mapping, `rerun`, restart, `balance`, and
  use between runs;
- the `drift` keyword (10 tests): the same output as without it for a
  crystal that does not move; sudden and gradual rigid moves, up to 2.5
  times the distance from a site to its cell faces, without false defects
  and with a translation equal to the move; known defects moving with the
  crystal; drift together with the affine mapping; continuation after a
  restart; independence of the history of the translation;
- the warnings and the errors.

They pass with 1, 2, 4 and 8 MPI ranks, also with `LAMMPS_SIZES=bigbig`.
With one rank, 3 tests that need more ranks are skipped.

## Comparison with OVITO

`tools/ovito_compare.py` runs the Wigner-Seitz analysis of OVITO on a
reference and a current configuration. The current file must contain the
per-atom array of the compute. The script compares the output for every
atom and every vacant site, and writes the OVITO result in the format of
LAMMPS dumps:

The script compares the analysis without `drift yes`: OVITO uses the
reference sites as they are.

```
write_dump all custom ref.dump id type x y z modify format float %20.15g      # before the run
...
write_dump all custom final.dump id type x y z c_ws[1] c_ws[2] c_ws[3] &
    modify format float %20.15g colname c_ws[1] Occupancy &
    colname c_ws[2] SiteIdentifier colname c_ws[3] SiteType               # after the run

python3 tools/ovito_compare.py --reference ref.dump --current final.dump --digits 15 \
    [--log log.lammps] [--defects def.dump] [--vacancies vac.dump] [--affine]
```

With the LAMMPS default format `%g` (6 significant digits), a few atoms
within about 0.001 A of the boundary between two sites can be assigned
differently by OVITO, which sees the rounded coordinates. The script lists
such atoms as ties. The exit code is 0 for identical results, 2 for
differences only at ties, and 1 for other differences.

## Validation

- **OVITO comparison** (OVITO 3.15.4 and 3.17.1, 4 MPI ranks):
  - configurations: fcc with 5 atom types, hcp, bcc, zinc blende, a
    triclinic fcc box, an fcc slab with two free surfaces, strained and
    sheared fcc and zinc blende boxes, 111 snapshots of a 1 keV cascade in
    SiC, and Cu heated to 1000 K in NPT with a reference at 0 K;
  - reference from the atoms and from a file, with and without
    `affine yes`;
  - in 369 snapshots, all 2 984 469 site assignments, all vacant sites and
    all counts are identical.
- **Production run:** a 100 keV cascade in equiatomic FeNiCrCoCu,
  15 185 664 atoms, 131 700 steps, 40 MPI ranks. The site of every atom
  and all 221 vacancies at the last step are identical to OVITO.
- **Reruns:** a cascade trajectory rerun with 1, 4 and 36 MPI ranks gives
  byte-identical dump files.
- **`drift yes`** (LAMMPS develop, 2-core virtual machine):
  - the trajectory of the example and a copy moved by a rigid,
    time-dependent translation of up to 4.9 A give the same counts and
    the same site for every atom in all 16 frames; the measured
    translations differ by the added shift within 6e-7 A. Without the
    keyword the moved copy has up to 448 Frenkel pairs where the original
    has 9;
  - the same with 1, 2, 4 and 8 MPI ranks: identical counts, translations
    and per-atom output;
  - a thermal crystal moved rigidly in 30 steps by up to three times the
    site-to-face distance (Cu at 300 K, W at 1500 K): no defects, and a
    translation equal to the move. Without the keyword, up to 5727 (Cu)
    and 2928 (W) false Frenkel pairs;
  - the drifted configurations of Wang et al. (2024): W 19 interstitials
    and no vacancy (66 / 47 without the keyword), Cu 3 and none
    (224 / 221);
  - without `drift yes` the results are bit-identical to the version
    without the keyword.
- Cost of `drift yes` (zinc blende, 10^6 atoms, 1 rank, 2-core virtual
  machine): 32 ms per invocation for static atoms and 49 ms when the
  crystal moves between invocations, against 28 ms without the keyword.

Time per invocation, measured on one node of the HybriLIT cluster (JINR):

| System                                  | `compute ws` 1 / 36 ranks | `compute frenkel` 1 / 36 ranks |
|-----------------------------------------|---------------------------|--------------------------------|
| fcc, 131 072 atoms                      | 3.79 / 0.27 ms            | 1692 / 148 ms                  |
| zinc blende, 125 000 atoms              | 2.35 / 0.20 ms            | 892 / 46.8 ms                  |

For comparison, on a 2-core virtual machine (zinc blende, 125 000 atoms,
1 rank), `compute voronoi/atom` with `occupation` takes 15.3 ms per
invocation and `compute ws` 3.6 ms.

## Related LAMMPS commands

- `compute frenkel` also performs a Wigner-Seitz analysis.
  - It generates its sites from the `lattice` command and uses one cell
    shape derived from the lattice.
  - It reports defect clusters.
  - It counts one interstitial per site with two or more atoms.
  - `compute ws` takes the sites from a reference configuration of any
    structure (also hcp, diamond, compounds, relaxed or distorted
    crystals). It counts n_s - 1 interstitials, as OVITO does, and reports
    replacements, antisites and the site of every atom.
- `compute voronoi/atom` with `occupation` gives per-atom occupancies. Its
  bookkeeping and communication grow with the total number of atoms, and
  it does not classify defects.

## References

- K. Nordlund, S. J. Zinkle, A. E. Sand, F. Granberg, R. S. Averback,
  R. E. Stoller, T. Suzudo, L. Malerba, F. Banhart, W. J. Weber,
  F. Willaime, S. L. Dudarev, D. Simeone, *Primary radiation damage: A
  review of current understanding and models*, J. Nucl. Mater. 512,
  450-479 (2018).
- A. Stukowski, *Visualization and analysis of atomistic simulation data
  with OVITO - the Open Visualization Tool*, Modelling Simul. Mater. Sci.
  Eng. 18, 015012 (2010).
- J. Wang, T. Li, Z. Fu, B. Fu, C. Gou, *Identification of
  self-interstitial atoms and vacancies in crystalline materials in
  atomistic simulation*, Comput. Phys. Commun. 299, 109138 (2024).

## Changes

- Keyword `drift` (follow a rigid translation of the crystal). Without
  it the analysis is unchanged: the counts, per-atom values and vacancies
  are bit-identical to the first version, and a warning is printed once
  if the crystal has drifted. The global vector has 7 elements with
  `drift yes`. The example uses `drift yes`.
- 6 November 2026: first version in this repository (exact nearest-site
  search without Voro++; keywords `reference` and `affine`). It replaces
  an earlier version based on the Voro++ library.

## Author

Trung Nguyen, Dalat Nuclear Research Institute (DNRI), Vietnam Atomic
Energy Institute (VINATOM). Contact: trungnth@gmail.com

## License

GNU General Public License version 2, the license of LAMMPS (see
`LICENSE`).
