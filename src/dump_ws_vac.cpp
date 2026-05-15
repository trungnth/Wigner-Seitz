#include "dump_ws_vac.h"
#include "domain.h"
#include "update.h"

using namespace LAMMPS_NS;

DumpWSVac::DumpWSVac(LAMMPS *lmp, int narg, char **arg) : DumpLocal(lmp, narg, arg) {}

void DumpWSVac::write_header(bigint ndump)
{
  if (multiproc) fprintf(fp,"ITEM: TIMESTEP\n");
  else fprintf(fp,"ITEM: TIMESTEP\n");
  fprintf(fp,BIGINT_FORMAT "\n",update->ntimestep);
  
  if (multiproc) fprintf(fp,"ITEM: NUMBER OF ATOMS\n");
  else fprintf(fp,"ITEM: NUMBER OF ATOMS\n");
  fprintf(fp,BIGINT_FORMAT "\n",ndump);
  
  if (domain->triclinic == 0) {
    fprintf(fp,"ITEM: BOX BOUNDS %s\n",boundstr);
    fprintf(fp,"%g %g\n",boxxlo,boxxhi);
    fprintf(fp,"%g %g\n",boxylo,boxyhi);
    fprintf(fp,"%g %g\n",boxzlo,boxzhi);
  } else {
    fprintf(fp,"ITEM: BOX BOUNDS xy xz yz %s\n",boundstr);
    fprintf(fp,"%g %g %g\n",boxxlo,boxxhi,boxxy);
    fprintf(fp,"%g %g %g\n",boxylo,boxyhi,boxxz);
    fprintf(fp,"%g %g %g\n",boxzlo,boxzhi,boxyz);
  }
  
  fprintf(fp,"ITEM: ATOMS id type x y z\n");
}