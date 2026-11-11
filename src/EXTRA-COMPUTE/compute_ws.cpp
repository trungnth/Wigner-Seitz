/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Trung Nguyen at DNRI (trungnth@gmail.com)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Implementation notes

   - The reference sites are the group atoms at the start of the first run
     after the compute was defined, or the atoms of the first snapshot of a
     dump file ("reference" keyword). Each current atom belongs to the
     Wigner-Seitz cell of its nearest site; no cell is constructed.
   - With "affine yes" the positions of the atoms are mapped into the box of
     the reference, x' = H_ref H^-1 (x - lo) + lo_ref, and the nearest site
     is sought there. Box flips are not supported: a flip moves the atoms
     of a rank away from its stored sites.
   - Nearest-site search: a stored site closer than r_min/2 to an atom is
     its nearest site, because every other stored site is then farther away
     (r_min = smallest distance between two stored sites, computed once per
     reference). Atoms and sites are stored in nearly the same spatial
     order, so this test is first applied to the sites with the atom's ID
     just after the previous atom's site, then to the sites in the atom's
     bin of a cell list of the stored sites. All other atoms are searched
     exactly on the cell list; positions outside it use its edge bins. The
     cell list keeps a copy of the site coordinates in bin order.
   - Each rank stores its local and ghost atoms as sites and owns the sites
     of its local atoms (index-based ownership, also valid for triclinic
     boxes). Occupancies are accumulated from local and ghost atoms, defects
     are classified on owned sites only, and all counters are summed by a
     single integer MPI_Allreduce() per assignment.
   - A reference file is read by rank 0 in chunks and broadcast. Each rank
     keeps the sites inside its sub-domain extended by the ghost cutoff, in
     fractional coordinates and with periodic images, and owns those inside
     its sub-domain. Owned and ghost sites are put into the order of the
     local and ghost atoms with the same ID, as for a reference taken from
     the atoms. The file is read again when the sub-domains changed between
     runs.
   - The per-atom array is filled only when it is requested, from the site
     assignment of the analysis of the same timestep.
   - One analysis per timestep serves all outputs and all invocations, e.g.
     by a dynamic group, thermo output and dumps. Only during setup is every
     invocation analyzed again, since a dynamic group may invoke the compute
     before its own setup. Atoms must not move after the first invocation of
     a timestep, as with the usual time integrators. Between runs, the
     per-atom output refers to the local atoms of the last analysis, so a
     changed number of local atoms is an error.
   - Internal check (not an output): the site occupancies must add up to the
     number of group atoms, otherwise a warning is printed once per reference.
   - drift keyword: the crystal can move as a whole relative to the box,
     e.g. by the momentum of a primary knock-on atom, by a thermostat, or by
     the mass that migrating defects carry while the center of mass stays
     fixed. Its translation t is a weighted mean displacement of the atoms
     from their sites (an M-estimate of location with a smooth, redescending
     weight): weight 1 closer than r_min/4 to the site, falling smoothly to
     0 at r_min/2, w = 1 - v^2 (3 - 2v) with v = 4 r / r_min - 1. Farther
     atoms (interstitials, displaced atoms of a cascade) do not count. An
     atom closer than r_min/2 to a site certainly belongs to it, so the
     weighted mean changes continuously with t and with the positions. A
     plain mean over the atoms alone on a site would change in steps, and an
     atom within about r_min/N of a cell face could then make two
     translations self-consistent.
   - The sums of the weights and of the weighted displacements are 64-bit
     integers, in units of 2^-27 and of 2^-27 times the largest power of two
     not above r_min/2 (fewer bits above 2^34 atoms, so that they cannot
     overflow). Each atom is counted on the rank that owns its site, and the
     sums travel in the same MPI_Allreduce() as the defect counters, so t is
     the same on every rank and for any number of ranks.
   - drift yes: atoms are assigned at x - t (after the affine mapping),
     starting with t of the previous analysis. t is corrected by the
     weighted mean displacement of the current assignment and the atoms are
     assigned again (iteratively reweighted), until the correction is below
     1e-6 r_min/2. After a correction, an atom closer than r_min/2 to its
     previous site keeps it; only the other atoms are searched again. The
     counters, per-atom values and vacancies are those of the last
     assignment, with the vacancies at the moved site positions. t is kept in
     an internal fix STORE/GLOBAL, so that a reference file used after a
     restart continues with the stored t.
   - drift no: every 16th atom (by ID) measures t, only for a warning that is
     printed once if t exceeds 25% of r_min/2.
------------------------------------------------------------------------- */

#include "compute_ws.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_store_global.h"
#include "group.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "reader_native.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>

using namespace LAMMPS_NS;
using MathConst::MY_PI;

namespace {

enum { INTERSTITIALS = 0, VACANCIES, REPLACEMENTS, ANTISITES, NVECTOR };
enum { UNACCOUNTED = NVECTOR, WEIGHT, SUMX, SUMY, SUMZ, NCOUNTER };
enum { BOX_CHANGED = 1, SUBDOMAIN_CHANGED = 2 };

constexpr int NDRIFT = 3;               // translation components in the global vector
constexpr int DRIFT_BITS = 27;          // fixed-point bits of the translation sums
constexpr int DRIFT_MAXITER = 10;       // most assignments in one analysis (drift yes)
constexpr double DRIFT_EPS = 1.0e-6;    // translation converged below this * r_min/2
constexpr double DRIFT_CORE = 0.5;      // full weight below this * r_min/2
constexpr double DRIFT_WARN = 0.25;     // warning (drift no) above this * r_min/2
constexpr double DRIFT_JUMP = 0.5;      // warning (drift yes) above this * r_min/2
constexpr int DRIFT_SAMPLE = 16;        // drift no: one atom in this many (by ID) measures t

constexpr int NPERATOM = 3;
constexpr int NLOCAL = 5;
constexpr int NFIELD = 5;
constexpr int DELTA = 1024;
constexpr int CHUNK = 16384;
constexpr int HINT_WINDOW = 8;
constexpr double SITES_PER_BIN = 8.0;
constexpr double BIN_GROWTH = 1.1;
constexpr double BOXTOL = 1.0e-10;
constexpr double BIG = std::numeric_limits<double>::max();

/* ----------------------------------------------------------------------
   upper triangular 3x3 matrices in Voigt order (xx, yy, zz, yz, xz, xy),
   as the box matrix domain->h: inverse, and y = a x + c
------------------------------------------------------------------------- */

void triangular_inverse(const double *h, double *hinv)
{
  hinv[0] = 1.0 / h[0];
  hinv[1] = 1.0 / h[1];
  hinv[2] = 1.0 / h[2];
  hinv[3] = -h[3] / (h[1] * h[2]);
  hinv[4] = (h[3] * h[5] - h[1] * h[4]) / (h[0] * h[1] * h[2]);
  hinv[5] = -h[5] / (h[0] * h[1]);
}

inline void triangular_apply(const double *a, const double *c, const double *x, double *y)
{
  y[0] = a[0] * x[0] + a[5] * x[1] + a[4] * x[2] + c[0];
  y[1] = a[1] * x[1] + a[3] * x[2] + c[1];
  y[2] = a[2] * x[2] + c[2];
}

/* ----------------------------------------------------------------------
   origin and shape matrix of the box in a dump file header; a triclinic
   header contains the bounding box and the tilt factors
------------------------------------------------------------------------- */

void file_box(const double box[3][3], int triclinic, double *lo, double *h)
{
  double xlo = box[0][0];
  double xhi = box[0][1];
  double ylo = box[1][0];
  double yhi = box[1][1];
  const double xy = triclinic ? box[0][2] : 0.0;
  const double xz = triclinic ? box[1][2] : 0.0;
  const double yz = triclinic ? box[2][2] : 0.0;
  if (triclinic) {
    xlo -= std::min(std::min(0.0, xy), std::min(xz, xy + xz));
    xhi -= std::max(std::max(0.0, xy), std::max(xz, xy + xz));
    ylo -= std::min(0.0, yz);
    yhi -= std::max(0.0, yz);
  }
  lo[0] = xlo;
  lo[1] = ylo;
  lo[2] = box[2][0];
  h[0] = xhi - xlo;
  h[1] = yhi - ylo;
  h[2] = box[2][1] - box[2][0];
  h[3] = yz;
  h[4] = xz;
  h[5] = xy;
}

}    // namespace

/* ---------------------------------------------------------------------- */

ComputeWS::ComputeWS(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), reffile(nullptr), affine_flag(0), drift_flag(0), drift_margin(BIG),
    rcert_all(0.0), drift_warned(0), jump_warned(0), converge_warned(0), margin_warned(0),
    id_fix(nullptr), fix_drift(nullptr), nref(0), nref_owned(0), maxref(0), xref(nullptr),
    tagref(nullptr), typeref(nullptr), nsites(0), ref_stored(false), occupancy(nullptr),
    occupant(nullptr), nmax(0), site_of_atom(nullptr), peratom(nullptr), nvacancy(0), maxvacancy(0),
    vacancy(nullptr), analyzed_step(-1), peratom_step(-1), analyzed_nlocal(0), cutoff_warned(0),
    unaccounted_warned(0), rcertsq(0.0), binsize(1.0), bininv(1.0), nbins(0), maxbins(0),
    binstart(nullptr), binsite(nullptr), xbin(nullptr)
{
  if (atom->tag_enable == 0) error->all(FLERR, "Compute ws requires atom IDs");

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "reference") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "compute ws reference", error);
      delete[] reffile;
      reffile = utils::strdup(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "affine") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "compute ws affine", error);
      affine_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "drift") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "compute ws drift", error);
      drift_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, iarg, "Unknown compute ws keyword: {}", arg[iarg]);
    }
  }

  for (int d = 0; d < 3; d++) {
    ref_boxlo[d] = 0.0;
    binlo[d] = 0.0;
    nbin[d] = 1;
  }
  for (int k = 0; k < 6; k++) ref_h[k] = ref_split[k] = map_h[k] = 0.0;
  ref_cutoff = 0.0;
  for (int d = 0; d < 3; d++) drift[d] = drift_used[d] = 0.0;

  dynamic_group_allow = 0;

  vector_flag = 1;
  size_vector = drift_flag ? NVECTOR + NDRIFT : NVECTOR;
  extvector = 0;

  // internal fix that stores the translation (x, y, z, set flag) in restart files;
  // created now so that a restart file can restore it

  if (drift_flag) {
    id_fix = utils::strdup(std::string(id) + "_COMPUTE_STORE");
    fix_drift = dynamic_cast<FixStoreGlobal *>(
        modify->add_fix(fmt::format("{} {} STORE/GLOBAL 4 1", id_fix, group->names[igroup])));
  }

  peratom_flag = 1;
  size_peratom_cols = NPERATOM;

  local_flag = 1;
  size_local_cols = NLOCAL;
  size_local_rows = 0;

  memory->create(vector, size_vector, "ws:vector");
  std::fill_n(vector, size_vector, 0.0);
}

/* ---------------------------------------------------------------------- */

ComputeWS::~ComputeWS()
{
  if (id_fix && modify->get_fix_by_id(id_fix)) modify->delete_fix(id_fix);
  delete[] id_fix;
  delete[] reffile;
  memory->destroy(vector);

  memory->destroy(xref);
  memory->destroy(tagref);
  memory->destroy(typeref);
  memory->destroy(occupancy);
  memory->destroy(occupant);
  memory->destroy(binsite);
  memory->destroy(xbin);
  memory->destroy(site_of_atom);
  memory->destroy(peratom);
  memory->destroy(vacancy);
  memory->destroy(binstart);
}

/* ---------------------------------------------------------------------- */

void ComputeWS::init()
{
  if (domain->dimension != 3)
    error->all(FLERR, Error::NOLASTLINE, "Compute ws requires a 3d system");

  if (domain->nonperiodic == 2)
    error->all(FLERR, Error::NOLASTLINE, "Compute ws does not support shrink-wrapped boundaries");
  if (domain->box_change_domain)
    error->all(FLERR, Error::NOLASTLINE,
               "Compute ws requires static processor sub-domains (no fix balance)");
  if (domain->box_change && !affine_flag)
    error->all(FLERR, Error::NOLASTLINE,
               "Compute ws requires a static simulation box unless 'affine yes' is used");
  if (affine_flag && domain->deform_vremap && (comm->me == 0))
    error->warning(FLERR,
                   "Compute ws with 'affine yes' assumes that the atoms are deformed with the "
                   "box, which fix deform with 'remap v' does not do");
  if (affine_flag && domain->triclinic && domain->box_change_shape && (comm->me == 0))
    error->warning(FLERR,
                   "Compute ws with 'affine yes' does not support box flips; use 'flip no' "
                   "if the box tilt can change");

  if (drift_flag) {
    fix_drift = dynamic_cast<FixStoreGlobal *>(modify->get_fix_by_id(id_fix));
    if (!fix_drift)
      error->all(FLERR, Error::NOLASTLINE, "Could not find compute ws fix with ID {}", id_fix);
  }
}

/* ----------------------------------------------------------------------
   called by Modify::setup() before every run, minimization, and rerun
   snapshot, after atoms were migrated and ghost atoms were acquired
------------------------------------------------------------------------- */

void ComputeWS::setup()
{
  analyzed_step = -1;

  if (!ref_stored) {
    if (reffile)
      read_reference();
    else
      store_reference();
    build_container();
    reset_drift();
  } else {
    check_reference();
  }

  check_ghost_cutoff();
}

/* ----------------------------------------------------------------------
   checks of a stored reference at every setup, also when the compute is
   invoked during setup before its own setup() (e.g. by a dynamic group):
   the box and the sub-domains must be unchanged, or the file is read
   again; a larger communication cutoff reads the file again or is reported
------------------------------------------------------------------------- */

void ComputeWS::check_reference()
{
  const int changed = geometry_changed();
  if ((changed & BOX_CHANGED) && !affine_flag)
    error->all(FLERR, Error::NOLASTLINE,
               "Box changed since compute ws stored its reference; use 'affine yes' or define "
               "the compute again");
  if ((changed & SUBDOMAIN_CHANGED) && !reffile)
    error->all(FLERR, Error::NOLASTLINE,
               "Processor sub-domains changed since compute ws stored its reference; use a "
               "reference file or define the compute again");

  const double cutoff = comm->get_comm_cutoff();
  if (reffile && ((changed & SUBDOMAIN_CHANGED) || (cutoff > ref_cutoff))) {
    read_reference();
    build_container();
  } else if (cutoff > ref_cutoff) {
    ref_cutoff = cutoff;
    if (comm->me == 0)
      error->warning(FLERR,
                     "Compute ws: communication cutoff increased after the reference was "
                     "stored; define the compute again or use a reference file");
  }
}

/* ----------------------------------------------------------------------
   make room for n stored sites
------------------------------------------------------------------------- */

void ComputeWS::grow_reference(int n)
{
  if (n <= maxref) return;

  maxref = n;
  memory->grow(xref, maxref, 3, "ws:xref");
  memory->grow(tagref, maxref, "ws:tagref");
  memory->grow(typeref, maxref, "ws:typeref");
  memory->grow(occupancy, maxref, "ws:occupancy");
  memory->grow(occupant, maxref, "ws:occupant");
  memory->grow(binsite, maxref, "ws:binsite");
  memory->grow(xbin, 3 * maxref, "ws:xbin");
}

/* ----------------------------------------------------------------------
   store the current local and ghost group atoms as reference sites
------------------------------------------------------------------------- */

void ComputeWS::store_reference()
{
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;

  grow_reference(nall);

  auto add_site = [&](int i) {
    xref[nref][0] = x[i][0];
    xref[nref][1] = x[i][1];
    xref[nref][2] = x[i][2];
    tagref[nref] = tag[i];
    typeref[nref] = type[i];
    nref++;
  };

  nref = 0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) add_site(i);
  nref_owned = nref;
  for (int i = nlocal; i < nall; i++)
    if (mask[i] & groupbit) add_site(i);

  bigint nowned = nref_owned;
  MPI_Allreduce(&nowned, &nsites, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  if ((nsites == 0) && (comm->me == 0))
    error->warning(FLERR, "Compute ws reference contains no sites (empty group?)");

  for (int d = 0; d < 3; d++) ref_boxlo[d] = domain->boxlo[d];
  for (int k = 0; k < 6; k++) ref_h[k] = map_h[k] = domain->h[k];
  sub_domain(ref_split);
  ref_cutoff = comm->get_comm_cutoff();
  ref_stored = true;
  unaccounted_warned = 0;

  if (comm->me == 0)
    utils::logmesg(lmp, "Compute ws {}: {} reference sites from the group atoms at step {}\n", id,
                   nsites, update->ntimestep);
}

/* ----------------------------------------------------------------------
   read the reference sites from the first snapshot of a dump file with
   the columns id, type, and x, y, z (also scaled or unwrapped)
------------------------------------------------------------------------- */

void ComputeWS::read_reference()
{
  int fieldtype[NFIELD] = {Reader::ID, Reader::TYPE, Reader::X, Reader::Y, Reader::Z};
  char *fieldlabel[NFIELD] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  double box[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  int info[5] = {0, 0, 0, 0, 0};
  bigint ntotal = 0;

  std::unique_ptr<ReaderNative> reader;
  if (comm->me == 0) {
    reader = std::make_unique<ReaderNative>(lmp);
    reader->open_file(reffile);
    bigint step = 0;
    if (reader->read_time(step)) {
      ntotal = -1;
    } else {
      int boxinfo = 0;
      ntotal = reader->read_header(box, boxinfo, info[0], 1, NFIELD, fieldtype, fieldlabel, 0, 0,
                                   info[1], info[2], info[3], info[4]);
    }
  }
  MPI_Bcast(&ntotal, 1, MPI_LMP_BIGINT, 0, world);
  if (ntotal < 0)
    error->all(FLERR, Error::NOLASTLINE, "Compute ws reference file {} contains no snapshot",
               reffile);
  MPI_Bcast(info, 5, MPI_INT, 0, world);
  MPI_Bcast(&box[0][0], 9, MPI_DOUBLE, 0, world);

  if ((info[1] < 0) || (info[2] == Reader::UNSET) || (info[3] == Reader::UNSET) ||
      (info[4] == Reader::UNSET))
    error->all(FLERR, Error::NOLASTLINE,
               "Compute ws reference file {} must contain the columns id, type, x, y, z", reffile);
  bool scaledxyz[3];
  for (int d = 0; d < 3; d++)
    scaledxyz[d] = (info[2 + d] == Reader::SCALE_WRAP) || (info[2 + d] == Reader::SCALE_NOWRAP);
  if ((scaledxyz[1] != scaledxyz[0]) || (scaledxyz[2] != scaledxyz[0]))
    error->all(FLERR, Error::NOLASTLINE,
               "Coordinates in compute ws reference file {} must all be scaled or all be unscaled",
               reffile);

  double flo[3], fh[6], finv[6];
  file_box(box, info[0], flo, fh);
  triangular_inverse(fh, finv);
  if (!ref_stored) reference_box(flo, fh);

  Sites own, ghost;
  double **buf = nullptr;
  memory->create(buf, CHUNK, NFIELD, "ws:buf");

  bigint nleft = ntotal;
  while (nleft > 0) {
    const int n = static_cast<int>(std::min(nleft, static_cast<bigint>(CHUNK)));
    if (comm->me == 0) reader->read_atoms(n, NFIELD, buf);
    MPI_Bcast(&buf[0][0], n * NFIELD, MPI_DOUBLE, 0, world);
    nleft -= n;
    select_sites(buf, n, scaledxyz[0], flo, finv, own, ghost);
  }

  memory->destroy(buf);
  reader.reset();

  bigint nowned = own.tag.size();
  MPI_Allreduce(&nowned, &nsites, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  if (nsites != ntotal)
    error->all(FLERR, Error::NOLASTLINE,
               "Compute ws assigned {} of the {} sites in reference file {}", nsites, ntotal,
               reffile);
  if ((nsites == 0) && (comm->me == 0))
    error->warning(FLERR, "Compute ws reference contains no sites (empty file?)");

  order_sites(own, ghost);

  sub_domain(ref_split);
  ref_cutoff = comm->get_comm_cutoff();
  ref_stored = true;
  unaccounted_warned = 0;

  if (comm->me == 0)
    utils::logmesg(lmp, "Compute ws {}: {} reference sites read from {}\n", id, nsites, reffile);
}

/* ----------------------------------------------------------------------
   box of the reference at the first read of the file: the simulation box
   if the file box equals it, otherwise the file box (only with affine yes)
------------------------------------------------------------------------- */

void ComputeWS::reference_box(const double *flo, const double *fh)
{
  const double *prd = domain->prd;
  const double scale[6] = {prd[0], prd[1], prd[2], prd[1], prd[0], prd[0]};
  bool same = true;
  for (int d = 0; d < 3; d++)
    if (fabs(flo[d] - domain->boxlo[d]) > BOXTOL * prd[d]) same = false;
  for (int k = 0; k < 6; k++)
    if (fabs(fh[k] - domain->h[k]) > BOXTOL * scale[k]) same = false;

  if (same) {
    for (int d = 0; d < 3; d++) ref_boxlo[d] = domain->boxlo[d];
    for (int k = 0; k < 6; k++) ref_h[k] = domain->h[k];
  } else if (!affine_flag) {
    error->all(FLERR, Error::NOLASTLINE,
               "Box in compute ws reference file {} differs from the simulation box; use "
               "'affine yes' if the crystal was deformed homogeneously",
               reffile);
  } else {
    for (int d = 0; d < 3; d++) ref_boxlo[d] = flo[d];
    for (int k = 0; k < 6; k++) ref_h[k] = fh[k];
  }
  for (int k = 0; k < 6; k++) map_h[k] = ref_h[k];
}

/* ----------------------------------------------------------------------
   add the n sites of a chunk of the file that lie in this rank's
   sub-domain (own) or in its extension by the ghost cutoff, with all
   periodic images (ghost); the file box is flo, finv = its inverse
------------------------------------------------------------------------- */

void ComputeWS::select_sites(double **buf, int n, bool scaled, double *flo, double *finv,
                             Sites &own, Sites &ghost)
{
  double split[6];
  sub_domain(split);
  const int periodic[3] = {domain->xperiodic, domain->yperiodic, domain->zperiodic};
  double cutfrac[3];
  int nimage[3];
  for (int d = 0; d < 3; d++) {
    cutfrac[d] = domain->triclinic ? comm->cutghost[d] : comm->cutghost[d] / domain->prd[d];
    nimage[d] = periodic[d] ? 1 + static_cast<int>(cutfrac[d]) : 0;
  }
  std::vector<int> shift[3];

  for (int j = 0; j < n; j++) {
    const double *row = buf[j];
    const auto stag = static_cast<tagint>(row[0]);
    const int stype = static_cast<int>(row[1]);

    double lam[3] = {row[2], row[3], row[4]};
    if (!scaled) domain->x2lamda(lam, lam, flo, finv);

    bool owned = true;
    for (int d = 0; d < 3; d++) {
      if (periodic[d]) {
        lam[d] -= std::floor(lam[d]);
        if (lam[d] >= 1.0) lam[d] = 0.0;
      }
      const double lc = periodic[d] ? lam[d] : std::min(std::max(lam[d], 0.0), 1.0);
      if ((lc < split[2 * d]) || ((lc >= split[2 * d + 1]) && (split[2 * d + 1] < 1.0)))
        owned = false;

      shift[d].clear();
      for (int k = -nimage[d]; k <= nimage[d]; k++) {
        const double l = lam[d] + k;
        if ((l >= split[2 * d] - cutfrac[d]) && (l < split[2 * d + 1] + cutfrac[d]))
          shift[d].push_back(k);
      }
    }

    double xs[3];
    for (const int k0 : shift[0]) {
      for (const int k1 : shift[1]) {
        for (const int k2 : shift[2]) {
          if (owned && (k0 == 0) && (k1 == 0) && (k2 == 0)) continue;
          const double l[3] = {lam[0] + k0, lam[1] + k1, lam[2] + k2};
          triangular_apply(ref_h, ref_boxlo, l, xs);
          ghost.add(xs, stag, stype);
        }
      }
    }

    if (owned) {
      triangular_apply(ref_h, ref_boxlo, lam, xs);
      own.add(xs, stag, stype);
    }
  }
}

/* ----------------------------------------------------------------------
   store the sites read from a file, owned sites first, in the order of the
   local and ghost atoms with the same ID (ghost sites by the nearest image),
   as for a reference taken from the atoms; this only speeds up guess_site()
------------------------------------------------------------------------- */

void ComputeWS::order_sites(const Sites &own, const Sites &ghost)
{
  const int nown = static_cast<int>(own.tag.size());
  const int nghost = static_cast<int>(ghost.tag.size());
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  tagint *tag = atom->tag;
  double **x = atom->x;

  std::vector<int> order;
  order.reserve(nown + nghost);

  std::unordered_map<tagint, int> index;
  index.reserve(nown);
  for (int k = nown - 1; k >= 0; k--) index[own.tag[k]] = k;
  std::vector<bool> used(nown, false);
  for (int i = 0; i < nlocal; i++) {
    auto it = index.find(tag[i]);
    if ((it == index.end()) || used[it->second]) continue;
    used[it->second] = true;
    order.push_back(it->second);
  }
  for (int k = 0; k < nown; k++)
    if (!used[k]) order.push_back(k);

  std::vector<int> byid(nghost);
  std::iota(byid.begin(), byid.end(), 0);
  std::sort(byid.begin(), byid.end(), [&](int a, int b) {
    return ghost.tag[a] < ghost.tag[b];
  });
  std::vector<bool> gused(nghost, false);
  double m[6], c[3], minv[6], cinv[3];
  const int mapped = affine_flag ? box_map(m, c, minv, cinv) : 0;
  for (int i = nlocal; i < nall; i++) {
    double xm[3];
    const double *xi = x[i];
    if (mapped) {
      triangular_apply(m, c, x[i], xm);
      xi = xm;
    }
    auto it = std::lower_bound(byid.begin(), byid.end(), tag[i], [&](int a, tagint t) {
      return ghost.tag[a] < t;
    });
    int best = -1;
    double rsqbest = BIG;
    for (; (it != byid.end()) && (ghost.tag[*it] == tag[i]); ++it) {
      if (gused[*it]) continue;
      const double *xs = &ghost.x[3 * (*it)];
      const double dx = xi[0] - xs[0];
      const double dy = xi[1] - xs[1];
      const double dz = xi[2] - xs[2];
      const double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq < rsqbest) {
        rsqbest = rsq;
        best = *it;
      }
    }
    if (best >= 0) {
      gused[best] = true;
      order.push_back(nown + best);
    }
  }
  for (int k = 0; k < nghost; k++)
    if (!gused[k]) order.push_back(nown + k);

  grow_reference(nown + nghost);
  nref = 0;
  for (const int k : order) {
    const Sites &list = (k < nown) ? own : ghost;
    const int g = (k < nown) ? k : k - nown;
    for (int d = 0; d < 3; d++) xref[nref][d] = list.x[3 * g + d];
    tagref[nref] = list.tag[g];
    typeref[nref] = list.type[g];
    nref++;
  }
  nref_owned = nown;
}

/* ---------------------------------------------------------------------- */

void ComputeWS::Sites::add(const double *xs, tagint id, int itype)
{
  x.insert(x.end(), xs, xs + 3);
  tag.push_back(id);
  type.push_back(itype);
}

/* ----------------------------------------------------------------------
   build the cell list of the stored sites and compute the radius of the
   nearest-site test, on this rank and (for the translation) on all ranks
------------------------------------------------------------------------- */

void ComputeWS::build_container()
{
  nbins = 0;
  rcertsq = 0.0;
  if (nref > 0) build_bins();

  double rcertsq_me = (nref > 0) ? rcertsq : BIG;
  double rcertsq_all = BIG;
  MPI_Allreduce(&rcertsq_me, &rcertsq_all, 1, MPI_DOUBLE, MPI_MIN, world);
  rcert_all = (rcertsq_all < BIG) ? sqrt(rcertsq_all) : 0.0;
}

/* ----------------------------------------------------------------------
   cell list of the stored sites and radius of the nearest-site test
------------------------------------------------------------------------- */

void ComputeWS::build_bins()
{
  double lo[3], hi[3], len[3];
  for (int d = 0; d < 3; d++) lo[d] = hi[d] = xref[0][d];
  for (int s = 1; s < nref; s++) {
    for (int d = 0; d < 3; d++) {
      lo[d] = std::min(lo[d], xref[s][d]);
      hi[d] = std::max(hi[d], xref[s][d]);
    }
  }

  double volume = 1.0;
  int ndim = 0;
  for (int d = 0; d < 3; d++) {
    len[d] = hi[d] - lo[d];
    if (len[d] > 0.0) {
      volume *= len[d];
      ndim++;
    }
  }

  const double ntarget = std::max(1.0, nref / SITES_PER_BIN);
  auto count_bins = [&](double h) {
    double n = 1.0;
    for (int d = 0; d < 3; d++) n *= std::max(1.0, std::ceil(len[d] / h));
    return n;
  };
  double h = (ndim > 0) ? std::pow(volume / ntarget, 1.0 / ndim) : 1.0;
  if (!(h > 0.0)) h = 1.0;
  while (count_bins(h) > 2.0 * ntarget) h *= BIN_GROWTH;

  binsize = h;
  bininv = 1.0 / h;
  nbins = 1;
  for (int d = 0; d < 3; d++) {
    binlo[d] = lo[d];
    nbin[d] = std::max(1, static_cast<int>(std::ceil(len[d] * bininv)));
    nbins *= nbin[d];
  }

  if (nbins + 1 > maxbins) {
    maxbins = nbins + 1;
    memory->destroy(binstart);
    memory->create(binstart, maxbins, "ws:binstart");
  }

  std::fill_n(binstart, nbins + 1, 0);
  for (int s = 0; s < nref; s++) binstart[bin_index(xref[s]) + 1]++;
  for (int b = 0; b < nbins; b++) binstart[b + 1] += binstart[b];
  for (int s = 0; s < nref; s++) binsite[binstart[bin_index(xref[s])]++] = s;
  for (int b = nbins; b > 0; b--) binstart[b] = binstart[b - 1];
  binstart[0] = 0;

  for (int p = 0; p < nref; p++) {
    const int s = binsite[p];
    xbin[3 * p] = xref[s][0];
    xbin[3 * p + 1] = xref[s][1];
    xbin[3 * p + 2] = xref[s][2];
  }

  double rminsq = BIG;
  for (int p = 0; p < nref; p++) nearest_site(&xbin[3 * p], p, -1, rminsq);
  rcertsq = 0.25 * rminsq;
}

/* ----------------------------------------------------------------------
   translation of a new reference: zero, or with a reference file the
   translation that a restart file stored in the internal fix
------------------------------------------------------------------------- */

void ComputeWS::reset_drift()
{
  for (int d = 0; d < 3; d++) drift[d] = drift_used[d] = 0.0;
  drift_warned = jump_warned = converge_warned = margin_warned = 0;
  if (!drift_flag) return;

  double *store = fix_drift->vstore;
  if (reffile && (store[3] == 1.0)) {
    for (int d = 0; d < 3; d++) drift[d] = drift_used[d] = store[d];
    if (comm->me == 0)
      utils::logmesg(lmp,
                     "Compute ws {}: translation ({:.8g}, {:.8g}, {:.8g}) of the crystal "
                     "taken from the restart file\n",
                     id, drift[0], drift[1], drift[2]);
  }
  for (int d = 0; d < 3; d++) store[d] = drift[d];
  store[3] = 1.0;
}

/* ----------------------------------------------------------------------
   warn once if the ghost shell may be too thin for boundary sites
------------------------------------------------------------------------- */

void ComputeWS::check_ghost_cutoff()
{
  drift_margin = BIG;
  if (nsites == 0) return;

  if ((comm->nprocs == 1) && !domain->xperiodic && !domain->yperiodic && !domain->zperiodic) return;

  const double volume = domain->xprd * domain->yprd * domain->zprd;
  const double rws = std::cbrt(3.0 * volume / (4.0 * MY_PI * (double) nsites));
  const double needed = 2.0 * rws + 0.5 * neighbor->skin;
  const double cutcomm = comm->get_comm_cutoff();
  drift_margin = cutcomm - needed;

  if (!cutoff_warned && (cutcomm < needed)) {
    cutoff_warned = 1;
    if (comm->me == 0)
      error->warning(FLERR,
                     "Compute ws: communication cutoff {:.6g} is shorter than the estimated "
                     "{:.6g} needed for sites at sub-domain boundaries; see comm_modify cutoff",
                     cutcomm, needed);
  }
}

/* ----------------------------------------------------------------------
   fractional bounds (lo, hi in each dimension) of this rank's sub-domain,
   which a change of the box size or shape leaves unchanged
------------------------------------------------------------------------- */

void ComputeWS::sub_domain(double *s) const
{
  if (comm->layout != Comm::LAYOUT_TILED) {
    const int *myloc = comm->myloc;
    s[0] = comm->xsplit[myloc[0]];
    s[1] = comm->xsplit[myloc[0] + 1];
    s[2] = comm->ysplit[myloc[1]];
    s[3] = comm->ysplit[myloc[1] + 1];
    s[4] = comm->zsplit[myloc[2]];
    s[5] = comm->zsplit[myloc[2] + 1];
  } else {
    for (int d = 0; d < 3; d++) {
      s[2 * d] = comm->mysplit[d][0];
      s[2 * d + 1] = comm->mysplit[d][1];
    }
  }
}

/* ----------------------------------------------------------------------
   on all ranks: BOX_CHANGED if the box differs from the reference box,
   SUBDOMAIN_CHANGED if any rank's sub-domain changed since the reference
------------------------------------------------------------------------- */

int ComputeWS::geometry_changed()
{
  double split[6];
  sub_domain(split);

  int flag = 0;
  for (int d = 0; d < 3; d++)
    if (domain->boxlo[d] != ref_boxlo[d]) flag |= BOX_CHANGED;
  for (int k = 0; k < 6; k++) {
    if (domain->h[k] != ref_h[k]) flag |= BOX_CHANGED;
    if (split[k] != ref_split[k]) flag |= SUBDOMAIN_CHANGED;
  }

  int flagall = 0;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_BOR, world);
  return flagall;
}

/* ----------------------------------------------------------------------
   affine map from the current box into the reference box, x' = m x + c,
   and its inverse, x = minv x' + cinv; returns 0 if the boxes are equal.
   A box flip changes a tilt factor by a box length, after which the atoms
   of a rank no longer correspond to its stored sites: an error is raised
   when a tilt factor changed by more than half a box length since the
   previous mapping (map_h).
------------------------------------------------------------------------- */

int ComputeWS::box_map(double *m, double *c, double *minv, double *cinv)
{
  const double *h = domain->h;
  const double *lo = domain->boxlo;

  if ((fabs(h[5] - map_h[5]) > 0.5 * h[0]) || (fabs(h[4] - map_h[4]) > 0.5 * h[0]) ||
      (fabs(h[3] - map_h[3]) > 0.5 * h[1]))
    error->all(FLERR, Error::NOLASTLINE,
               "Compute ws with 'affine yes': a box tilt changed by more than half a box length "
               "since the previous invocation (box flip?); invoke the compute more often");
  for (int k = 0; k < 6; k++) map_h[k] = h[k];

  bool same = true;
  for (int d = 0; d < 3; d++)
    if (lo[d] != ref_boxlo[d]) same = false;
  for (int k = 0; k < 6; k++)
    if (h[k] != ref_h[k]) same = false;
  if (same) return 0;

  double inv[6];
  triangular_inverse(h, inv);
  MathExtra::multiply_shape_shape(ref_h, inv, m);
  triangular_inverse(ref_h, inv);
  MathExtra::multiply_shape_shape(h, inv, minv);

  const double zero[3] = {0.0, 0.0, 0.0};
  double t[3];
  triangular_apply(m, zero, lo, t);
  for (int d = 0; d < 3; d++) c[d] = ref_boxlo[d] - t[d];
  triangular_apply(minv, zero, ref_boxlo, t);
  for (int d = 0; d < 3; d++) cinv[d] = lo[d] - t[d];
  return 1;
}

/* ----------------------------------------------------------------------
   site with ID itag that is closer than r_min/2 to position xi, sought
   among the sites following the previous atom's site (snext) and the site
   following the one found last by find_site() (slast); -1 if there is none
------------------------------------------------------------------------- */

int ComputeWS::guess_site(const double *xi, tagint itag, int snext, int slast) const
{
  const int send = std::min(snext + HINT_WINDOW, nref);
  for (int s = snext; s < send; s++) {
    if (tagref[s] != itag) continue;
    const double dx = xi[0] - xref[s][0];
    const double dy = xi[1] - xref[s][1];
    const double dz = xi[2] - xref[s][2];
    if (dx * dx + dy * dy + dz * dz < rcertsq) return s;
  }

  if ((slast < nref) && (tagref[slast] == itag)) {
    const double dx = xi[0] - xref[slast][0];
    const double dy = xi[1] - xref[slast][1];
    const double dz = xi[2] - xref[slast][2];
    if (dx * dx + dy * dy + dz * dz < rcertsq) return slast;
  }

  return -1;
}

/* ----------------------------------------------------------------------
   index of the stored site nearest to position xi, -1 if no site is stored
------------------------------------------------------------------------- */

int ComputeWS::find_site(const double *xi) const
{
  if (nref == 0) return -1;

  int sbest = -1;
  double rsqbest = BIG;

  const int b = bin_index(xi);
  for (int p = binstart[b]; p < binstart[b + 1]; p++) {
    const double *xs = &xbin[3 * p];
    const double dx = xi[0] - xs[0];
    const double dy = xi[1] - xs[1];
    const double dz = xi[2] - xs[2];
    const double rsq = dx * dx + dy * dy + dz * dz;
    if (rsq < rcertsq) return binsite[p];
    if (rsq < rsqbest) {
      rsqbest = rsq;
      sbest = binsite[p];
    }
  }

  return nearest_site(xi, -1, sbest, rsqbest);
}

/* ----------------------------------------------------------------------
   exact search of the cell list for a stored site closer to xi than
   rsqbest, except the site at position skip of the cell list; returns the
   nearest such site, or sbest if there is none, and updates rsqbest
------------------------------------------------------------------------- */

int ComputeWS::nearest_site(const double *xi, int skip, int sbest, double &rsqbest) const
{
  int c[3];
  double gap = BIG;
  for (int d = 0; d < 3; d++) {
    c[d] = bin_coord(d, xi[d]);
    const double blo = binlo[d] + c[d] * binsize;
    gap = std::min(gap, std::min(xi[d] - blo, blo + binsize - xi[d]));
  }
  gap = std::max(gap, 0.0);

  const int kmax = std::max(nbin[0], std::max(nbin[1], nbin[2]));
  for (int k = 0; k < kmax; k++) {
    const int i2lo = std::max(c[2] - k, 0);
    const int i2hi = std::min(c[2] + k, nbin[2] - 1);
    const int i1lo = std::max(c[1] - k, 0);
    const int i1hi = std::min(c[1] + k, nbin[1] - 1);

    for (int i2 = i2lo; i2 <= i2hi; i2++) {
      const double dz = bin_distance(2, i2, xi[2]);
      const bool face2 = (i2 == c[2] - k) || (i2 == c[2] + k);

      for (int i1 = i1lo; i1 <= i1hi; i1++) {
        const double dy = bin_distance(1, i1, xi[1]);
        const bool face = face2 || (i1 == c[1] - k) || (i1 == c[1] + k);
        const int step = face ? 1 : 2 * k;

        for (int i0 = c[0] - k; i0 <= c[0] + k; i0 += step) {
          if ((i0 < 0) || (i0 >= nbin[0])) continue;
          const double dx = bin_distance(0, i0, xi[0]);
          if (dx * dx + dy * dy + dz * dz >= rsqbest) continue;

          const int b = i0 + nbin[0] * (i1 + nbin[1] * i2);
          for (int p = binstart[b]; p < binstart[b + 1]; p++) {
            if (p == skip) continue;
            const double *xs = &xbin[3 * p];
            const double rx = xi[0] - xs[0];
            const double ry = xi[1] - xs[1];
            const double rz = xi[2] - xs[2];
            const double rsq = rx * rx + ry * ry + rz * rz;
            if (rsq < rsqbest) {
              rsqbest = rsq;
              sbest = binsite[p];
            }
          }
        }
      }
    }

    const double bound = gap + k * binsize;
    if (bound * bound >= rsqbest) break;
  }

  return sbest;
}

/* ----------------------------------------------------------------------
   index of the bin that contains position xi, clamped to the cell list
------------------------------------------------------------------------- */

int ComputeWS::bin_index(const double *xi) const
{
  return bin_coord(0, xi[0]) + nbin[0] * (bin_coord(1, xi[1]) + nbin[1] * bin_coord(2, xi[2]));
}

/* ----------------------------------------------------------------------
   bin index of coordinate v in dimension d, clamped to the cell list
------------------------------------------------------------------------- */

int ComputeWS::bin_coord(int d, double v) const
{
  const double t = (v - binlo[d]) * bininv;
  if (!(t > 0.0)) return 0;
  if (t >= nbin[d] - 1) return nbin[d] - 1;
  return static_cast<int>(t);
}

/* ----------------------------------------------------------------------
   distance in dimension d from coordinate v to bin i
------------------------------------------------------------------------- */

double ComputeWS::bin_distance(int d, int i, double v) const
{
  const double blo = binlo[d] + i * binsize;
  if (v < blo) return blo - v;
  const double bhi = blo + binsize;
  if (v > bhi) return v - bhi;
  return 0.0;
}

/* ----------------------------------------------------------------------
   Wigner-Seitz analysis; produces all outputs at once
------------------------------------------------------------------------- */

void ComputeWS::analyze()
{
  if ((analyzed_step == update->ntimestep) && !update->setupflag) {
    if (update->whichflag == 0) {
      const int changed = (atom->nlocal != analyzed_nlocal) ? 1 : 0;
      int changedall = 0;
      MPI_Allreduce(&changed, &changedall, 1, MPI_INT, MPI_MAX, world);
      if (changedall)
        error->all(FLERR,
                   "Compute ws: atoms were added or removed since the last analysis; use "
                   "'run 0' before using the compute");
    }
    return;
  }

  if (!ref_stored) {
    if (reffile)
      read_reference();
    else
      store_reference();
    build_container();
    reset_drift();
  } else if (update->setupflag) {
    check_reference();
  }

  analyzed_step = update->ntimestep;
  analyzed_nlocal = atom->nlocal;
  peratom_step = -1;
  invoked_vector = invoked_peratom = invoked_local = analyzed_step;

  if (atom->nmax > nmax) {
    memory->destroy(peratom);
    memory->destroy(site_of_atom);
    nmax = atom->nmax;
    memory->create(peratom, nmax, size_peratom_cols, "ws:peratom");
    memory->create(site_of_atom, nmax, "ws:site_of_atom");
    array_atom = peratom;
  }

  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;

  double m[6], c[3], minv[6], cinv[3];
  const int mapped = affine_flag ? box_map(m, c, minv, cinv) : 0;

  // position at which the site of atom i is sought: mapped into the box of the
  // reference (affine yes) and moved back by the translation of the crystal (drift yes)

  double shift[3] = {0.0, 0.0, 0.0};
  if (drift_flag)
    for (int d = 0; d < 3; d++) shift[d] = drift[d];
  const bool moved = mapped || drift_flag;
  auto position = [&](int i, double *xm) -> const double * {
    if (!moved) return x[i];
    if (mapped) {
      triangular_apply(m, c, x[i], xm);
    } else {
      xm[0] = x[i][0];
      xm[1] = x[i][1];
      xm[2] = x[i][2];
    }
    xm[0] -= shift[0];
    xm[1] -= shift[1];
    xm[2] -= shift[2];
    return xm;
  };

  const bool allgroup = (igroup == 0);
  bigint ngroup = nlocal;
  if (!allgroup) {
    ngroup = 0;
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) ngroup++;
  }

  // weight of an atom in the translation: 1 closer than r_min/4 to its site, falling
  // smoothly to 0 at r_min/2 (such an atom certainly belongs to that site). The sums
  // are 64-bit integers, with fewer fixed-point bits for very large systems so that
  // they cannot overflow. Without the drift keyword they serve only the warning, and
  // only every DRIFT_SAMPLE-th atom (by ID) contributes.

  const bool measure = drift_flag || !drift_warned;
  const double rcsq = rcert_all * rcert_all;
  const double rcoresq = DRIFT_CORE * DRIFT_CORE * rcsq;
  int bits = DRIFT_BITS;
  while ((bits > 1) && (atom->natoms > (MAXBIGINT >> (bits + 2)))) bits--;
  const double weight_scale = std::ldexp(1.0, bits);
  const double sum_scale = (rcert_all > 0.0) ? std::ldexp(1.0, bits - std::ilogb(rcert_all)) : 0.0;
  const double tolerance = std::max(DRIFT_EPS, std::ldexp(4.0, -bits)) * rcert_all;

  bigint one[NCOUNTER];
  bigint all[NCOUNTER];

  // add the weighted displacement of an atom at xi from its site s if this rank owns s,
  // so that every atom is counted once (rounded toward zero in fixed point)

  auto add_weighted = [&](int s, const double *xi) {
    if ((s < 0) || (s >= nref_owned)) return;
    const double dx = xi[0] - xref[s][0];
    const double dy = xi[1] - xref[s][1];
    const double dz = xi[2] - xref[s][2];
    const double rsq = dx * dx + dy * dy + dz * dz;
    if (rsq >= rcsq) return;
    double w = 1.0;
    if (rsq > rcoresq) {
      const double v = (sqrt(rsq) / rcert_all - DRIFT_CORE) / (1.0 - DRIFT_CORE);
      w = 1.0 - v * v * (3.0 - 2.0 * v);
    }
    one[WEIGHT] += static_cast<bigint>(w * weight_scale);
    one[SUMX] += static_cast<bigint>(w * dx * sum_scale);
    one[SUMY] += static_cast<bigint>(w * dy * sum_scale);
    one[SUMZ] += static_cast<bigint>(w * dz * sum_scale);
  };

  double residual[3] = {0.0, 0.0, 0.0};
  double rlen = 0.0;

  for (int iter = 1;; iter++) {
    std::fill_n(one, NCOUNTER, 0);
    if (nref > 0) std::fill_n(occupancy, nref, 0);

    if (iter == 1) {

      // first assignment: guess from the order of the sites, else exact search

      int snext = 0;
      int slast = 0;
      for (int i = 0; i < nall; i++) {
        site_of_atom[i] = -1;
        if (!allgroup && !(mask[i] & groupbit)) continue;

        double xm[3];
        const double *xi = position(i, xm);

        int s = guess_site(xi, tag[i], snext, slast);
        if (s >= 0) {
          snext = s + 1;
        } else {
          s = find_site(xi);
          if (s < 0) continue;
          slast = s + 1;
        }

        site_of_atom[i] = s;
        occupancy[s]++;
        occupant[s] = i;
        if (measure && (drift_flag || (tag[i] % DRIFT_SAMPLE == 0))) add_weighted(s, xi);
      }

    } else {

      // assignment with the corrected translation: an atom closer than r_min/2 to
      // its previous site keeps it, all others are searched again

      for (int i = 0; i < nall; i++) {
        if (!allgroup && !(mask[i] & groupbit)) continue;

        double xm[3];
        const double *xi = position(i, xm);

        int s = site_of_atom[i];
        if (s >= 0) {
          const double dx = xi[0] - xref[s][0];
          const double dy = xi[1] - xref[s][1];
          const double dz = xi[2] - xref[s][2];
          if (dx * dx + dy * dy + dz * dz >= rcertsq) s = find_site(xi);
        } else {
          s = find_site(xi);
        }

        site_of_atom[i] = s;
        if (s < 0) continue;
        occupancy[s]++;
        occupant[s] = i;
        add_weighted(s, xi);
      }
    }

    bigint nassigned = 0;
    nvacancy = 0;
    for (int s = 0; s < nref_owned; s++) {
      const int occ = occupancy[s];
      nassigned += occ;

      if (occ == 0) {
        if (nvacancy == maxvacancy) {
          maxvacancy += DELTA;
          memory->grow(vacancy, maxvacancy, NLOCAL, "ws:vacancy");
        }
        double *row = vacancy[nvacancy++];
        row[0] = tagref[s];
        row[1] = typeref[s];
        const double xs[3] = {xref[s][0] + shift[0], xref[s][1] + shift[1], xref[s][2] + shift[2]};
        if (mapped) {
          triangular_apply(minv, cinv, xs, &row[2]);
        } else {
          row[2] = xs[0];
          row[3] = xs[1];
          row[4] = xs[2];
        }
      } else {
        one[INTERSTITIALS] += occ - 1;

        if (occ == 1) {
          const int i = occupant[s];
          if (tag[i] != tagref[s]) one[REPLACEMENTS]++;
          if (type[i] != typeref[s]) one[ANTISITES]++;
        }
      }
    }
    one[VACANCIES] = nvacancy;
    one[UNACCOUNTED] = ngroup - nassigned;

    MPI_Allreduce(one, all, NCOUNTER, MPI_LMP_BIGINT, MPI_SUM, world);

    // weighted mean displacement = translation of the crystal that the shift of this
    // assignment does not yet contain

    double rsq = 0.0;
    for (int d = 0; d < 3; d++) {
      residual[d] = 0.0;
      if (all[WEIGHT] > 0)
        residual[d] = (double) all[SUMX + d] * weight_scale / ((double) all[WEIGHT] * sum_scale);
      rsq += residual[d] * residual[d];
    }
    rlen = sqrt(rsq);

    if (!drift_flag) break;

    if ((iter == 1) && !jump_warned && (nsites > 0)) {
      if (all[WEIGHT] == 0) {
        jump_warned = 1;
        if (comm->me == 0)
          error->warning(FLERR,
                         "Compute ws: no atom is closer than half the shortest distance "
                         "between sites to a site at step {}, so the translation of the crystal "
                         "cannot be followed; invoke the compute more often",
                         update->ntimestep);
      } else if (rlen > DRIFT_JUMP * rcert_all) {
        jump_warned = 1;
        if (comm->me == 0)
          error->warning(
              FLERR,
              "Compute ws: the crystal moved by {:.6g} since the previous analysis "
              "at step {}; atoms are matched with the wrong sites if it moved by more than "
              "half the shortest distance between sites, so invoke the compute more often",
              rlen, update->ntimestep);
      }
    }
    if (rlen <= tolerance) break;
    if (iter == DRIFT_MAXITER) {
      if (!converge_warned) {
        converge_warned = 1;
        if (comm->me == 0)
          error->warning(FLERR,
                         "Compute ws: the translation of the crystal did not converge in {} "
                         "assignments at step {}",
                         DRIFT_MAXITER, update->ntimestep);
      }
      break;
    }
    for (int d = 0; d < 3; d++) shift[d] += residual[d];
  }

  array_local = vacancy;
  size_local_rows = nvacancy;

  for (int k = 0; k < NVECTOR; k++) vector[k] = (double) all[k];

  if (drift_flag) {
    for (int d = 0; d < 3; d++) {
      drift_used[d] = shift[d];
      drift[d] = shift[d] + residual[d];
      vector[NVECTOR + d] = drift_used[d];
      fix_drift->vstore[d] = drift[d];
    }
    fix_drift->vstore[3] = 1.0;

    const double dlen = sqrt(drift[0] * drift[0] + drift[1] * drift[1] + drift[2] * drift[2]);
    if (!margin_warned && (dlen > drift_margin)) {
      margin_warned = 1;
      if (comm->me == 0)
        error->warning(FLERR,
                       "Compute ws: the translation {:.6g} of the crystal exceeds the margin "
                       "{:.6g} of the communication cutoff; increase it with comm_modify cutoff",
                       dlen, drift_margin);
    }
  } else if (!drift_warned && (rlen > DRIFT_WARN * rcert_all)) {
    drift_warned = 1;
    if (comm->me == 0)
      error->warning(FLERR,
                     "Compute ws: the crystal is displaced from the reference sites by "
                     "{:.6g} at step {} ({:.0f}% of half the shortest distance between sites); "
                     "a drift of the crystal produces false Frenkel pairs, see the drift keyword",
                     rlen, update->ntimestep, 100.0 * rlen / rcert_all);
  }

  const bigint unaccounted = all[UNACCOUNTED];
  if ((unaccounted != 0) && !unaccounted_warned) {
    unaccounted_warned = 1;
    if (comm->me == 0)
      error->warning(FLERR,
                     "Compute ws: site occupancies differ from the number of group atoms by "
                     "{}; increase the communication cutoff or remove atoms far from all sites",
                     unaccounted);
  }
}

/* ----------------------------------------------------------------------
   public entry points: each stamps its own invocation and shares one analysis
------------------------------------------------------------------------- */

void ComputeWS::compute_vector()
{
  invoked_vector = update->ntimestep;
  analyze();
}

/* ---------------------------------------------------------------------- */

void ComputeWS::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  analyze();
  if (peratom_step == analyzed_step) return;
  peratom_step = analyzed_step;

  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    double *row = peratom[i];
    const int s = site_of_atom[i];
    if (s < 0) {
      std::fill_n(row, size_peratom_cols, 0.0);
      continue;
    }

    row[0] = occupancy[s];
    row[1] = tagref[s];
    row[2] = typeref[s];
  }
}

/* ---------------------------------------------------------------------- */

void ComputeWS::compute_local()
{
  invoked_local = update->ntimestep;
  analyze();
}

/* ---------------------------------------------------------------------- */

double ComputeWS::memory_usage()
{
  double bytes = (double) nmax * size_peratom_cols * sizeof(double);
  bytes += (double) nmax * sizeof(int);
  bytes += (double) maxref * 3 * sizeof(double);
  bytes += (double) maxref * sizeof(tagint);
  bytes += (double) maxref * 4 * sizeof(int);
  bytes += (double) maxref * 3 * sizeof(double);
  bytes += (double) maxvacancy * NLOCAL * sizeof(double);
  bytes += (double) maxbins * sizeof(int);
  return bytes;
}
