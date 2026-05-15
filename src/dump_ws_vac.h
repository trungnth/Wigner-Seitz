#ifdef DUMP_CLASS

DumpStyle(ws/vac,DumpWSVac)

#else

#ifndef LMP_DUMP_WS_VAC_H
#define LMP_DUMP_WS_VAC_H

#include "dump_local.h"

namespace LAMMPS_NS {

class DumpWSVac : public DumpLocal {
 public:
  DumpWSVac(class LAMMPS *, int, char **);
  ~DumpWSVac() override {}
  void write_header(bigint) override;
};

}

#endif
#endif