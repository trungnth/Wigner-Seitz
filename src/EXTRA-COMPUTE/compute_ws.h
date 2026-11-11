/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(ws,ComputeWS);
// clang-format on
#else

#ifndef LMP_COMPUTE_WS_H
#define LMP_COMPUTE_WS_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeWS : public Compute {
 public:
  ComputeWS(class LAMMPS *, int, char **);
  ~ComputeWS() override;

  void init() override;
  void setup() override;
  void compute_vector() override;
  void compute_peratom() override;
  void compute_local() override;
  double memory_usage() override;

 private:
  struct Sites {
    std::vector<double> x;
    std::vector<tagint> tag;
    std::vector<int> type;
    void add(const double *, tagint, int);
  };

  char *reffile;      // dump file with the reference sites (reference keyword), or nullptr
  int affine_flag;    // 1 = map positions into the box of the reference (affine yes)
  int drift_flag;     // 1 = follow a translation of the crystal (drift yes)

  double drift[3];         // translation of the crystal for the next analysis (reference frame)
  double drift_used[3];    // translation used by the last analysis (output)
  double drift_margin;     // largest translation the communication cutoff can follow
  double rcert_all;        // r_min/2 of all sites (smallest over all ranks)
  int drift_warned;        // 1 after the warning about a displaced crystal (drift no)
  int jump_warned;         // 1 after the warning about a large move between analyses
  int converge_warned;     // 1 after the warning about a translation that did not converge
  int margin_warned;       // 1 after the warning about the communication cutoff
  char *id_fix;            // internal fix that keeps the translation in restart files
  class FixStoreGlobal *fix_drift;

  int nref;               // stored sites of this rank (owned first, then ghost sites)
  int nref_owned;         // sites owned by this rank
  int maxref;             // allocated length of the site arrays
  double **xref;          // site positions
  tagint *tagref;         // site IDs
  int *typeref;           // site types
  bigint nsites;          // total number of sites
  bool ref_stored;        // true once the reference is stored
  double ref_boxlo[3];    // box of the reference: origin
  double ref_h[6];        // box of the reference: shape matrix (as domain->h)
  double ref_split[6];    // fractional sub-domain bounds when the sites were stored
  double map_h[6];        // box shape at the previous affine mapping (flip check)
  double ref_cutoff;      // communication cutoff when the sites were stored

  int *occupancy;    // number of atoms on each stored site
  int *occupant;     // local index of the last atom assigned to each site

  int nmax;             // allocated length of the per-atom arrays
  int *site_of_atom;    // stored site of each local atom, -1 if none
  double **peratom;     // per-atom output array

  int nvacancy;        // vacant owned sites of this rank
  int maxvacancy;      // allocated rows of the vacancy array
  double **vacancy;    // local output array

  bigint analyzed_step;      // timestep of the last analysis
  bigint peratom_step;       // timestep for which the per-atom array was filled
  int analyzed_nlocal;       // number of local atoms at the last analysis
  int cutoff_warned;         // 1 after the cutoff warning
  int unaccounted_warned;    // 1 after the warning about unassigned atoms

  double rcertsq;    // (r_min/2)^2: a site this close to an atom is its nearest site

  double binlo[3];    // cell list of the stored sites: origin
  double binsize;     // edge length of a bin
  double bininv;      // 1/binsize
  int nbin[3];        // bins per dimension
  int nbins;          // total bins
  int maxbins;        // allocated bins
  int *binstart;      // first entry of each bin in binsite and xbin
  int *binsite;       // sites in bin order
  double *xbin;       // site positions in bin order

  void analyze();
  void store_reference();
  void read_reference();
  void reference_box(const double *, const double *);
  void select_sites(double **, int, bool, double *, double *, Sites &, Sites &);
  void order_sites(const Sites &, const Sites &);
  void check_reference();
  void grow_reference(int);
  void build_container();
  void build_bins();
  void reset_drift();
  void check_ghost_cutoff();
  void sub_domain(double *) const;
  int geometry_changed();
  int box_map(double *, double *, double *, double *);
  int guess_site(const double *, tagint, int, int) const;
  int find_site(const double *) const;
  int nearest_site(const double *, int, int, double &) const;
  int bin_index(const double *) const;
  int bin_coord(int, double) const;
  double bin_distance(int, int, double) const;
};

}    // namespace LAMMPS_NS

#endif
#endif
