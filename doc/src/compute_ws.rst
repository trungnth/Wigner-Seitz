.. index:: compute ws

compute ws command
==================

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID ws keyword value ...

* ID, group-ID are documented in :doc:`compute <compute>` command
* ws = style name of this compute command
* zero or more keyword/value pairs may be appended
* keyword = *reference* or *affine*

  .. parsed-literal::

       *reference* value = file
         file = name of a dump file whose first snapshot defines the sites
       *affine* value = *yes* or *no*
         *yes* = map the positions into the box of the reference
         *no* = require the box of the reference

Examples
""""""""

.. code-block:: LAMMPS

   compute 1 all ws
   compute 2 metal ws affine yes
   compute 3 all ws reference ref.dump affine yes

   compute ws all ws
   thermo_style custom step temp c_ws[1] c_ws[2] c_ws[3] c_ws[4]
   dump vac all local 1000 vac.dump index c_ws[1] c_ws[2] c_ws[3] c_ws[4] c_ws[5]
   variable def atom "c_ws[1] >= 2 || type != c_ws[3] || id != c_ws[2]"
   dump def all custom 1000 def.dump id type x y z c_ws[*]
   dump_modify def thresh v_def == 1

Description
"""""""""""

.. versionadded:: TBD

Define a computation that identifies point defects by Wigner-Seitz (WS)
analysis :ref:`(Nordlund) <ws-Nordlund>`.

The group atoms of a reference configuration define the lattice
*sites*.  Each current group atom is assigned to the site whose WS cell,
i.e. Voronoi cell in the reference configuration, contains it.  This is
its nearest site.  The number of atoms on site :math:`s` is its
*occupancy* :math:`n_s`.

Each site is in exactly one of the following states:

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Site state
     - Condition
     - Counted as
   * - vacant
     - :math:`n_s = 0`
     - 1 vacancy
   * - unchanged
     - :math:`n_s = 1`, the atom that defined the site
     - nothing
   * - replaced
     - :math:`n_s = 1`, another atom of the site type
     - 1 replacement
   * - antisite
     - :math:`n_s = 1`, an atom of another type
     - 1 antisite, and 1 replacement if it is another atom
   * - interstitial
     - :math:`n_s \ge 2`
     - :math:`n_s - 1` interstitials

* An interstitial is a property of a site, not of an atom: a dumbbell
  counts as one interstitial, and the compute does not decide which of
  its atoms is the interstitial.
* Replacements and antisites are counted only on singly occupied sites.
  A multiply occupied site contributes only interstitials, even if its
  original atom has left or none of its atoms has the site type.
* Atom types do not change in MD, so every antisite is then also a
  replacement.  An antisite without replacement occurs only if types
  are changed in place, e.g. by :doc:`fix atom/swap <fix_atom_swap>`.

Vacancies and interstitials follow the conventions of the WS analysis in
OVITO :ref:`(Stukowski) <ws-Stukowski>`, which does not count
replacements or antisites.  Replacements measure atomic mixing rather
than lattice defects.  The nearest site is found by an exact search;
no Voronoi cells are constructed.

The :doc:`compute frenkel <compute_frenkel>` command also performs a
Wigner-Seitz analysis.  It generates its sites from a :doc:`lattice
<lattice>` (or reads them from a file), derives one cell shape from the
lattice for all sites, and reports vacancies, interstitials, and
clusters of defects.  This compute instead uses the group atoms of a
reference configuration as sites, so it needs no lattice definition and
applies to any crystal structure, also when relaxed or distorted.  It
assigns every atom to its nearest site, also in crystals whose sites
have differently oriented surroundings, such as diamond and hcp, and it
reports replacements, antisites, and the site of every atom.  The two
computes count interstitials differently: compute frenkel counts one
interstitial for every site with two or more atoms, this compute counts
:math:`n_s - 1`.

The reference is stored at the start of the first run or minimization
after the compute is defined.  With :doc:`rerun <rerun>`, it is the
first snapshot.  The reference should be a defect-free crystal, e.g.
created with :doc:`create_atoms <create_atoms>` or relaxed with
:doc:`minimize <minimize>`.  A new reference is stored by defining the
compute again with the same ID (:doc:`uncompute <uncompute>`, then
*compute*); thermo output, dumps, and variables that use the compute
need no change.

Between runs, the output of the compute is that of its last analysis,
i.e. at the end of the previous run.  After commands that add, remove,
or move atoms, e.g. :doc:`create_atoms <create_atoms>`,
:doc:`delete_atoms <delete_atoms>`, or :doc:`displace_atoms
<displace_atoms>`, use ``run 0`` before the compute is used again; a
changed number of atoms is detected and is an error.

With the *reference* keyword, the sites are instead read at the start of
the first run from the first snapshot of a dump file in the native
format of :doc:`dump custom <dump>`, with the columns ``id``, ``type``,
and ``x``, ``y``, ``z`` (also unwrapped, e.g. ``zu``, or scaled, e.g.
``xs``, ``ys``, ``zs``, but not a mix of scaled and unscaled
coordinates).  Without *affine* = *yes*, the box in the file must be the
simulation box.  Compressed and binary dump files are read as by
:doc:`read_dump <read_dump>`.  Such a file is written, e.g., before the
run in which the reference would otherwise be stored:

.. code-block:: LAMMPS

   write_dump all custom ref.dump id type x y z modify format float %20.15g

The *modify* option writes the positions with 15 significant digits
instead of the default 6.  All sites in the file are used and the
compute group selects the current atoms, so the file should be written
for the compute group.  Site IDs and types are taken from the file;
replacements are meaningful if the IDs in the file are those of the
simulated atoms, e.g. from the same :doc:`create_atoms <create_atoms>`
command or chain of restart files.
A compute does not store its reference in a :doc:`restart file
<restart>`: a job started from a restart file uses the crystal at its
start, which may already contain defects, unless it reads the reference
from the file written in the first job.

With *affine* = *yes*, the box may change size and shape, e.g. under
:doc:`fix npt <fix_nh>`, :doc:`fix deform <fix_deform>` with *remap x*,
or by :doc:`change_box <change_box>` with *remap* between runs, and the
box of a reference file may differ from the simulation box, e.g. a
crystal at 0 K for a simulation at higher temperature.  The positions of
the atoms are then mapped into the box of the reference by the affine
transformation that maps the current box onto it, as with the affine
mapping to the reference configuration in OVITO :ref:`(Stukowski)
<ws-Stukowski>`, and each atom is assigned to its nearest site there.
This removes a homogeneous deformation, e.g. thermal expansion, so it is
only meaningful if the atoms were deformed together with the box, which
is not the case for :doc:`fix deform <fix_deform>` with *remap v* or
*remap none*, for :doc:`fix npt <fix_nh>` with *dilate partial*, or for
:doc:`change_box <change_box>` without *remap*.

Box flips of a triclinic box (*flip yes* of :doc:`fix deform
<fix_deform>` or :doc:`fix npt <fix_nh>`) are not supported.  An error
is raised when a tilt factor changed by more than half a box length
since the previous invocation of the compute, so under continuous shear
the compute must be invoked often enough, e.g. by the thermo output,
that the tilt changes by less than that between two invocations.  A
flip is missed if the tilt also changed continuously by about half a
box length in between.  LAMMPS therefore warns when the tilt can change
during a run with *affine* = *yes*; use *flip no* in that case.

Only group atoms are used, both as sites and as occupants.  A single
sublattice can thus be analyzed with a group defined by atom type.

The variable *def* in the examples selects all atoms whose site differs
from the reference: atoms sharing a site, antisites, atoms on another
atom's site, and atoms not assigned to any site (site ID 0).  These
atoms and the vacancies (local array) describe the occupancy of every
site, so type-resolved and sublattice statistics can be obtained in
post-processing.  Use the compute group for the dump; atoms outside it
have all-zero per-atom values.  The chemical composition of
interstitials is best reported per configuration, e.g. as A-A or A-B
dumbbells on A sites, by grouping the dumped atoms by site ID, rather
than by assigning a type to an individual interstitial atom.  The
example in ``examples/PACKAGES/ws`` follows a displacement cascade in
SiC this way and includes a Python script for these statistics.

.. note::

   Replacements are detected by atom ID.  Commands that renumber atoms,
   e.g. :doc:`delete_atoms <delete_atoms>` with its default
   ``compress yes``, invalidate the replacement count.

----------

Each processor stores its local and ghost group atoms as sites and owns
the sites of its local atoms, so every site is counted exactly once.
Sites read from a file are stored and owned in the same way, by
position.  Occupancies are determined from local and ghost atoms.  This
requires a communication cutoff of at least the diameter of a sphere
with the volume per site (about 1.1 nearest-neighbor distances in fcc,
hcp, and bcc crystals, 1.4 in diamond) plus half the :doc:`neighbor
skin <neighbor>`.  Pair style cutoffs usually suffice; otherwise use
:doc:`comm_modify cutoff <comm_modify>` before the reference is
stored.  A larger cutoff set later makes the compute read a reference
file again, whereas the sites taken from the atoms stay unchanged
(LAMMPS warns), so the compute must then be defined again.  LAMMPS
warns if the cutoff appears too short, and if the site occupancies do
not add up to the number of group atoms, e.g. for atoms in vacuum
farther than the cutoff from any site.

The processor sub-domains must not change during a run: :doc:`fix
balance <fix_balance>` and shrink-wrapped boundaries are rejected.  A
change of the sub-domains between runs, e.g. by :doc:`balance
<balance>`, is an error unless the sites are read from a file, which is
then read again.  Unless *affine* = *yes*, the box must not change while
a reference is in use: fixes that change it are rejected, and a change
between runs, e.g. by :doc:`change_box <change_box>`, is an error.

----------

Output info
"""""""""""

This compute calculates a global vector of length 4, a per-atom array
with 3 columns, and a local array with 5 columns, all in the same
invocation.  ``c_ID[I]`` refers to the global vector in
:doc:`thermo_style <thermo_style>` and equal-style variables, to the
per-atom array in :doc:`dump custom <dump>` and atom-style variables,
and to the local array in :doc:`dump local <dump>`.  See the
:doc:`Howto output <Howto_output>` page for an overview of LAMMPS output
options.

The global vector values are "intensive" (not normalized by the number
of atoms):

#. interstitials, :math:`\sum_s \max(n_s - 1, 0)`
#. vacancies
#. replacements
#. antisites

If every group atom is assigned to a site, :math:`N_\mathrm{int} -
N_\mathrm{vac} = N_\mathrm{group} - N_\mathrm{sites}`.  Without added
or removed atoms, interstitials and vacancies thus form Frenkel pairs.

The per-atom array values are:

#. occupancy of the atom's site (at least 1)
#. site ID, i.e. the ID of the atom that defined the site
#. site type

All values are 0 for atoms outside the group or not assigned to a site.

The local array has one row per vacant site.  Each vacancy is stored by
exactly one processor, so the total number of rows equals global vector
element 2.  The values are:

#. site ID
#. site type
#. x coordinate of the site
#. y coordinate of the site
#. z coordinate of the site

Coordinates are in distance :doc:`units <units>`; all other values are
counts or IDs.  With *affine* = *yes*, the coordinates are those of the
site mapped into the current box.

Restrictions
""""""""""""

This compute is part of the EXTRA-COMPUTE package.  It is only enabled
if LAMMPS was built with that package.  See the :doc:`Build package
<Build_package>` page for more info.

This compute requires a 3d system and atom IDs.  The processor
sub-domains must be static during a run, and so must the box unless
*affine* = *yes*.  Box flips of a triclinic box are not supported.
Shrink-wrapped boundaries and dynamic groups are not allowed.

Related commands
""""""""""""""""

:doc:`compute frenkel <compute_frenkel>`,
:doc:`compute voronoi/atom <compute_voronoi_atom>`,
:doc:`compute centro/atom <compute_centro_atom>`,
:doc:`compute cna/atom <compute_cna_atom>`,
:doc:`dump local <dump>`,
:doc:`rerun <rerun>`

Default
"""""""

The option defaults are no reference file (the group atoms at the start
of the first run) and affine = no.

----------

.. _ws-Nordlund:

**(Nordlund)** Nordlund, Zinkle, Sand, Granberg, Averback, Stoller,
Suzudo, Malerba, Banhart, Weber, Willaime, Dudarev, and Simeone,
J Nucl Mater, 512, 450-479 (2018).

.. _ws-Stukowski:

**(Stukowski)** Stukowski, Modelling Simul Mater Sci Eng, 18, 015012
(2010).
