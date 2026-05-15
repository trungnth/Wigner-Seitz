#include "compute_ws.h"
#include "atom.h"
#include "update.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "voro++.hh"
#include <map>
#include <cmath>

using namespace LAMMPS_NS;

ComputeWS::ComputeWS(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg) {
  if (narg != 3) error->all(FLERR,"Illegal compute ws command");

  peratom_flag = 1;
  size_peratom_cols = 4;
  nmax = 0;
  ws_array = nullptr;
  
  local_flag = 1;
  size_local_cols = 5; // [ID, Type, X, Y, Z]
  v_array = nullptr;
  n_vacancies = 0;

  vector_flag = 1;
  size_vector = 4; // [Interstitials, Vacancies, Replacements, Antisites]
  extvector = 1;
  memory->create(vector, size_vector, "ws:vector");

  nref_max = 0;
  nref = 0;
  ref_x = nullptr;
  ref_tag = nullptr;
  ref_type = nullptr;
  ref_stored = false;
}

ComputeWS::~ComputeWS() {
  memory->destroy(ws_array);
  memory->destroy(ref_x);
  memory->destroy(ref_tag);
  memory->destroy(ref_type);
  memory->destroy(v_array);
  memory->destroy(vector);
}

void ComputeWS::init() {
  if (domain->dimension != 3)
    error->all(FLERR,"Compute ws requires 3D simulation");
}

void ComputeWS::store_reference() {
  int nall = atom->nlocal + atom->nghost; 
  
  if (nall > nref_max) {
    nref_max = atom->nmax;
    memory->destroy(ref_x);
    memory->destroy(ref_tag);
    memory->destroy(ref_type);
    memory->create(ref_x, nref_max, 3, "ws:ref_x");
    memory->create(ref_tag, nref_max, "ws:ref_tag");
    memory->create(ref_type, nref_max, "ws:ref_type");
  }

  nref = nall;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *type = atom->type;

  for (int i = 0; i < nall; i++) {
    ref_x[i][0] = x[i][0];
    ref_x[i][1] = x[i][1];
    ref_x[i][2] = x[i][2];
    ref_tag[i] = tag[i];
    ref_type[i] = type[i];
  }
  ref_stored = true;
}

void ComputeWS::compute_peratom() {
  invoked_peratom = update->ntimestep;

  if (!ref_stored) store_reference();

  int nlocal = atom->nlocal;
  if (nlocal > nmax) {
    nmax = atom->nmax;
    memory->destroy(ws_array);
    memory->create(ws_array, nmax, 4, "ws:ws_array");
    array_atom = ws_array;
  }

  double *sublo = domain->sublo;
  double *subhi = domain->subhi;
  double skin = 10.0; 

  double dx = subhi[0] - sublo[0] + 2.0 * skin;
  double dy = subhi[1] - sublo[1] + 2.0 * skin;
  double dz = subhi[2] - sublo[2] + 2.0 * skin;
  double vol = dx * dy * dz;

  // Voro++ runs optimally with about 3-8 atoms per bin according to library documentation
  double optimal_bins = nref / 5.0; 
  if (optimal_bins < 1.0) optimal_bins = 1.0;
  
  double scale = pow(optimal_bins / vol, 1.0/3.0);
  int nx = round(dx * scale);
  int ny = round(dy * scale);
  int nz = round(dz * scale);
  
  if (nx < 1) nx = 1;
  if (ny < 1) ny = 1;
  if (nz < 1) nz = 1;

  voro::container con(sublo[0] - skin, subhi[0] + skin, 
                      sublo[1] - skin, subhi[1] + skin, 
                      sublo[2] - skin, subhi[2] + skin,
                      nx, ny, nz, false, false, false, 8);

  for (int i = 0; i < nref; i++) {
    con.put(i, ref_x[i][0], ref_x[i][1], ref_x[i][2]);
  }

  double **x = atom->x;
  int nall = atom->nlocal + atom->nghost;
  int *assigned_site = new int[nall];
  std::map<int, int> site_occupancy;

  // Pass 1: Map current local AND ghost atoms
  for (int i = 0; i < nall; i++) {
    double rx, ry, rz;
    int site_id;
    if (con.find_voronoi_cell(x[i][0], x[i][1], x[i][2], rx, ry, rz, site_id)) {
      assigned_site[i] = site_id;
      site_occupancy[site_id]++;
    } else {
      assigned_site[i] = -1;
    }
  }

  // Pass 2: Populate LAMMPS per-atom output array, Count Sputtered, Replacements, and Antisites
  int loc_rep = 0;
  int loc_anti = 0; 
  int loc_sputtered = 0; 
  for (int i = 0; i < nlocal; i++) {
    int s_id = assigned_site[i];
    if (s_id >= 0 && s_id < nref) {
      ws_array[i][0] = site_occupancy[s_id]; 
      ws_array[i][1] = s_id;                 
      ws_array[i][2] = ref_tag[s_id];        
      ws_array[i][3] = ref_type[s_id];       
      
      // Structural mixing check (Replacements)
      if (ws_array[i][2] != atom->tag[i]) loc_rep++;
      
      // Chemical mixing check (Antisite Defects)
      if (ws_array[i][3] != atom->type[i]) loc_anti++;
      
    } else {
      ws_array[i][0] = 0;  
      ws_array[i][1] = -1; 
      ws_array[i][2] = 0;  
      ws_array[i][3] = 0;
      loc_sputtered++; 
    }
  }

  // Pass 3: Identify Vacancies & Interstitials (in-lattice)
  int loc_vac = 0;
  int loc_int = 0;
  for (int i = 0; i < nref; i++) {
    if (ref_x[i][0] >= sublo[0] && ref_x[i][0] < subhi[0] &&
        ref_x[i][1] >= sublo[1] && ref_x[i][1] < subhi[1] &&
        ref_x[i][2] >= sublo[2] && ref_x[i][2] < subhi[2]) {
      
      // Vacancy check
      if (site_occupancy.find(i) == site_occupancy.end() || site_occupancy[i] == 0) {
        loc_vac++;
      } 
      // Interstitial check (multiple atoms in one W-S cell)
      else if (site_occupancy[i] > 1) {
        loc_int += (site_occupancy[i] - 1);
      }
    }
  }

  // Total Interstitials = Lattice Interstitials + Sputtered/Ejected Atoms
  int total_loc_int = loc_int + loc_sputtered;

  // Pass 4: Populate Local Array (Dummy Atoms for Vacancies)
  n_vacancies = loc_vac;
  memory->destroy(v_array);
  memory->create(v_array, n_vacancies, 5, "ws:v_array"); 

  int v_idx = 0;
  for (int i = 0; i < nref; i++) {
    if (ref_x[i][0] >= sublo[0] && ref_x[i][0] < subhi[0] &&
        ref_x[i][1] >= sublo[1] && ref_x[i][1] < subhi[1] &&
        ref_x[i][2] >= sublo[2] && ref_x[i][2] < subhi[2]) {
      if (site_occupancy.find(i) == site_occupancy.end() || site_occupancy[i] == 0) {
        v_array[v_idx][0] = ref_tag[i];  
        v_array[v_idx][1] = ref_type[i]; 
        v_array[v_idx][2] = ref_x[i][0]; 
        v_array[v_idx][3] = ref_x[i][1]; 
        v_array[v_idx][4] = ref_x[i][2]; 
        v_idx++;
      }
    }
  }

  array_local = v_array;
  size_local_rows = n_vacancies;

  // MPI Reduction for Global Vector [Interstitials, Vacancies, Replacements, Antisites]
  double loc_stats[4] = {(double)total_loc_int, (double)loc_vac, (double)loc_rep, (double)loc_anti};
  MPI_Allreduce(loc_stats, vector, 4, MPI_DOUBLE, MPI_SUM, world);

  delete[] assigned_site;
}

void ComputeWS::compute_local() {
  invoked_local = update->ntimestep;
  if (invoked_peratom != update->ntimestep) compute_peratom();
}

void ComputeWS::compute_vector() {
  invoked_vector = update->ntimestep;
  if (invoked_peratom != update->ntimestep) compute_peratom();
}

double ComputeWS::memory_usage() {
  double bytes = nmax * 4 * sizeof(double);
  bytes += nref_max * 3 * sizeof(double);
  bytes += nref_max * sizeof(tagint);
  bytes += nref_max * sizeof(int);
  bytes += n_vacancies * 5 * sizeof(double); 
  return bytes;
}
