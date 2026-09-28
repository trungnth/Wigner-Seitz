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
ComputeStyle(ws,ComputeWS)
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
  void compute_peratom() override;
  void compute_local() override;
  void compute_vector() override;
  double memory_usage() override;

 private:
  int nmax;
  double **ws_array;
  int *assigned_site;

  int nref_max;
  int nref;
  double **ref_x;
  tagint *ref_tag;
  int *ref_type;
  bool ref_stored;

  int *site_occupancy;
  tagint *site_occ_tag;
  int *site_occ_type;

  double **v_array;
  int n_vacancies;
  double user_skin;

  void store_reference();
};

}    // namespace LAMMPS_NS

#endif
#endif