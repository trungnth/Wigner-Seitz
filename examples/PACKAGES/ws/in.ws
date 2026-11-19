# Wigner-Seitz defect analysis of a displacement cascade in 3C-SiC
# with compute ws
#
# 3C-SiC has the zinc blende structure: two interpenetrating fcc
# sublattices, one occupied by Si (type 1) and one by C (type 2), so
# antisites (C on a Si site, Si on a C site) are well defined.  The
# Gao-Weber potential for SiC is joined to the ZBL repulsion at short
# distances (pair style gw/zbl), as needed for energetic collisions.
#
# A Si primary knock-on atom (PKA) at the box center starts the cascade.
# Atoms within one unit cell of the box faces are coupled to a Berendsen
# thermostat that removes the deposited energy (deterministic, so the
# result does not depend on the number of MPI ranks).
#
# Output:
#   thermo              interstitials, vacancies, replacements, antisites,
#                       the number of atoms written to defects.*.dump, and
#                       the translation of the crystal (drift yes, see below)
#   vacancies.*.dump    dump local: one line per vacant site
#   defects.*.dump      dump custom: only atoms whose site differs from the
#                       reference (interstitial configurations, antisites,
#                       and atoms that moved to another site)
#   both at the first step (perfect crystal, empty) and at the last step
#
# Together, the two dumps describe the occupancy of every site.  Per-type
# and sublattice statistics follow from them in post-processing:
#   python3 ws_sublattice.py
#
# Usage:  lmp -in in.ws [-var L 5] [-var epka 200.0] [-var seed 4928459]

variable        L     index 5          # box edge in unit cells (8*L^3 atoms)
variable        epka  index 200.0      # kinetic energy of the PKA (eV)
variable        seed  index 4928459    # random seed for the initial velocities

units           metal
atom_style      atomic
boundary        p p p

# zinc blende = diamond lattice with Si on basis atoms 1-4 and C on 5-8;
# 4.3632 A is the zero-pressure lattice constant of this potential at 300 K
lattice         diamond 4.3632
region          box block 0 ${L} 0 ${L} 0 ${L}
create_box      2 box
create_atoms    1 box basis 1 1 basis 2 1 basis 3 1 basis 4 1 &
                      basis 5 2 basis 6 2 basis 7 2 basis 8 2

variable        mSi equal 28.0855
mass            1 ${mSi}
mass            2 12.011

pair_style      gw/zbl
pair_coeff      * * SiC.gw.zbl Si C
neighbor        1.0 bin
neigh_modify    every 1 delay 0 check yes

region          inner block 1 $(v_L-1) 1 $(v_L-1) 1 $(v_L-1)
group           inner region inner
group           border subtract all inner

# ------------------------------------------------------------------------
# Wigner-Seitz analysis
#
# Defined before the first run: the perfect lattice above becomes the
# reference configuration at the beginning of that run.
#
# drift yes: the PKA velocity below is added to the thermal velocities
# (sum yes), so the whole crystal moves along [1 2 3] with the momentum of
# the PKA, by about 0.3 A until the end of the run.  The compute follows
# this translation instead of counting atoms near the faces of their sites
# as displaced.
#
# global vector   c_ws_map[1..4]  interstitials, vacancies, replacements,
#                                 antisites
#                 c_ws_map[5..7]  translation of the crystal (drift yes)
# per-atom array  c_ws_map[1..3]  occupancy, site ID, site type
# local array     c_ws_map[1..5]  site ID, site type, x, y, z of each
#                                 vacant site (moved with the crystal)
#
# Other references: "reference ref.dump" reads the sites from a dump file,
# "affine yes" removes a homogeneous deformation of the box (fix npt).
# ------------------------------------------------------------------------

compute         ws_map all ws drift yes

# an atom is part of a defect if it shares its site with other atoms,
# occupies another atom's site, or has a different type than its site
variable        def atom "(c_ws_map[1] >= 2) || (id != c_ws_map[2]) || (type != c_ws_map[3])"
compute         ndef all reduce sum v_def

thermo_style    custom step time dt temp pe c_ws_map[1] c_ws_map[2] c_ws_map[3] c_ws_map[4] c_ndef &
                c_ws_map[5] c_ws_map[6] c_ws_map[7]
thermo_modify   colname c_ws_map[1] Interstitials colname c_ws_map[2] Vacancies &
                colname c_ws_map[3] Replacements colname c_ws_map[4] Antisites &
                colname c_ndef DefectAtoms colname c_ws_map[5] DriftX &
                colname c_ws_map[6] DriftY colname c_ws_map[7] DriftZ
thermo          100

dump            defects all custom 1500 defects.*.dump id type x y z &
                c_ws_map[1] c_ws_map[2] c_ws_map[3]
dump_modify     defects thresh v_def == 1 &
                format 6 %.0f format 7 %.0f format 8 %.0f &
                colname c_ws_map[1] Occupancy colname c_ws_map[2] SiteIdentifier &
                colname c_ws_map[3] SiteType

dump            vacancies all local 1500 vacancies.*.dump &
                c_ws_map[1] c_ws_map[2] c_ws_map[3] c_ws_map[4] c_ws_map[5]
dump_modify     vacancies format 1 %.0f format 2 %.0f &
                colname c_ws_map[1] id colname c_ws_map[2] type &
                colname c_ws_map[3] x colname c_ws_map[4] y colname c_ws_map[5] z

# ---- thermalize the crystal (no defects expected) ----------------------
# NVE from 600 K: equipartition gives about 300 K

velocity        all create 600.0 ${seed} mom yes rot yes dist gaussian loop geom
fix             nve all nve
timestep        0.001
run             400

# ---- primary knock-on atom: the Si site at the box center --------------

variable        c equal floor(0.5*${L})
region          center sphere ${c} ${c} ${c} 0.25 units lattice
group           pka region center
variable        npka equal count(pka)
if              "${npka} != 1" then "print 'ERROR: PKA group must contain exactly one atom'" "quit 1"

# velocity in A/ps along [1 2 3], away from low-index channeling directions
variable        vpka equal sqrt(2.0*${epka}/(${mSi}*1.0364269e-4))
variable        vx   equal 1.0*${vpka}/sqrt(14.0)
variable        vy   equal 2.0*${vpka}/sqrt(14.0)
variable        vz   equal 3.0*${vpka}/sqrt(14.0)
velocity        pka set ${vx} ${vy} ${vz} sum yes units box

# ---- cascade: thermostat only at the box faces, adaptive time step -----

fix             tstat border temp/berendsen 300.0 300.0 0.1
fix             dtreset all dt/reset 1 NULL 0.001 0.05 units box
run             1100

# ---- summary ------------------------------------------------------------
# No atoms were added or removed, so (unless compute ws printed a
# warning) every interstitial has a vacancy partner (Frenkel pairs).

variable        ni equal c_ws_map[1]
variable        nv equal c_ws_map[2]
variable        nr equal c_ws_map[3]
variable        na equal c_ws_map[4]
variable        nd equal c_ndef

print           "Surviving defects: ${ni} interstitials, ${nv} vacancies, ${na} antisites, ${nr} replacements (${nd} atoms in defects.*.dump)"
