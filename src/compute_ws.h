#ifdef COMPUTE_CLASS

ComputeStyle(ws,ComputeWS)

#else

#ifndef LMP_COMPUTE_WS_H
#define LMP_COMPUTE_WS_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeWS : public Compute {
 public:
  ComputeWS(class LAMMPS *, int, char **);
  ~ComputeWS();
  void init() override;
  void compute_peratom() override;
  void compute_local() override;
  void compute_vector() override;
  double memory_usage() override;

 private:
  int nmax;
  double **ws_array;
  
  int nref_max;
  int nref;
  double **ref_x;
  tagint *ref_tag;
  int *ref_type;
  bool ref_stored;

  // Vacancy tracking arrays
  double **v_array;
  int n_vacancies;

  void store_reference();
};

}

#endif
#endif