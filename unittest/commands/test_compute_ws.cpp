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

// Unit tests for compute ws (Wigner-Seitz defect analysis, EXTRA-COMPUTE package)
//
// Test system: L1_2-ordered fcc crystal (type 1 on cube corners, type 2 on
// face centers) with a known set of defects. Type-resolved and sublattice
// quantities are rebuilt from the per-atom and local outputs, as users do
// in post-processing. Every check uses globally
// reduced values or is restricted to the rank that owns an atom, so the
// same binary is valid for any number of MPI ranks:
//
//   ./test_compute_ws                   serial
//   mpirun -np 4 ./test_compute_ws      decomposition independence

#include "lammps.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "group.h"
#include "info.h"
#include "input.h"
#include "modify.h"
#include "utils.h"

#include "../testing/core.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>

// whether to print verbose output (i.e. not capturing LAMMPS screen output).
bool verbose = false;

using LAMMPS_NS::Compute;
using LAMMPS_NS::tagint;
using ::testing::HasSubstr;
using ::testing::Not;

// Like TEST_FAILURE from ../testing/core.h, but safe with several MPI ranks:
// every rank must get the error, and only rank 0, which prints the error
// message, checks its text. Non-fatal checks keep all ranks executing the
// same commands; a fatal check on one rank only would leave the other ranks
// waiting in a collective call (Error::all() starts with an MPI barrier).
#define TEST_FAILURE_MPI(errmsg, ...)                            \
    {                                                            \
        bool thrown = false;                                     \
        ::testing::internal::CaptureStdout();                    \
        try {                                                    \
            __VA_ARGS__;                                         \
        } catch (std::exception &) {                             \
            thrown = true;                                       \
        }                                                        \
        auto mesg = ::testing::internal::GetCapturedStdout();    \
        if (verbose) std::cout << mesg;                          \
        EXPECT_TRUE(thrown) << "no error raised";                \
        if (lmp->comm->me == 0) {                                \
            EXPECT_THAT(mesg, ::testing::ContainsRegex(errmsg)); \
        }                                                        \
    }

namespace {

constexpr int NCELL   = 8;                         // fcc unit cells per box edge
constexpr int NSITES  = 4 * NCELL * NCELL * NCELL; // 2048 sites
constexpr int NSITES1 = NCELL * NCELL * NCELL;     // corner sites, type 1
constexpr int NSITES2 = NSITES - NSITES1;          // face-center sites, type 2

} // namespace

class ComputeWSTest : public LAMMPSTest {
protected:
    void SetUp() override
    {
        testbinary = "ComputeWSTest";
        LAMMPSTest::SetUp();
        if (!info->has_style("compute", "ws")) GTEST_SKIP() << "compute ws is not available";
    }

    // files written by the tests; removed also when a test failed before its own cleanup
    void TearDown() override
    {
        if (lmp) {
            MPI_Barrier(lmp->world);
            if (lmp->comm->me == 0) {
                for (const char *name : {"ws_site.dump", "ws_bad.dump", "ws_ref.dump",
                                         "ws_frenkel.dump", "ws_test.restart"})
                    std::remove(name);
            }
        }
        LAMMPSTest::TearDown();
    }

    // L1_2-ordered fcc crystal in LJ units
    void create_crystal(bool triclinic)
    {
        BEGIN_HIDE_OUTPUT();
        command("units lj");
        command("atom_style atomic");
        command("atom_modify map array");
        command("boundary p p p");
        command("lattice fcc 0.8442");
        command("region box block 0 8 0 8 0 8");
        command("create_box 2 box");
        command("create_atoms 1 box basis 1 1 basis 2 2 basis 3 2 basis 4 2");
        if (triclinic) command("change_box all triclinic");
        command("mass * 1.0");
        command("pair_style lj/cut 2.5");
        command("pair_coeff * * 1.0 1.0");
        command("neighbor 0.3 bin");
        END_HIDE_OUTPUT();
    }

    // define the compute and make thermo output invoke it
    void define_compute(const std::string &options)
    {
        BEGIN_HIDE_OUTPUT();
        command("compute ws all ws " + options);
        command("thermo_style custom step c_ws[1] c_ws[2] c_ws[3] c_ws[4]");
        END_HIDE_OUTPUT();
    }

    // Known defects, chosen by position so the result does not depend on how
    // atom IDs were assigned to ranks:
    //   vacancies     corner site (2,2,2) [type 1] and face site (2.5,2.5,2) [type 2]
    //   antisite      corner site (3,3,3) set to type 2
    //   replacement   atom on corner site (6,6,6) deleted and re-created (new ID)
    //   interstitial  type-2 atom placed in the cell of face site (5.5,5.5,5)
    // followed by small random displacements of all atoms (thermal noise)
    void create_defects()
    {
        BEGIN_HIDE_OUTPUT();
        command("region v1 sphere 2.0 2.0 2.0 0.1 units lattice");
        command("region v2 sphere 2.5 2.5 2.0 0.1 units lattice");
        command("region as sphere 3.0 3.0 3.0 0.1 units lattice");
        command("region rp sphere 6.0 6.0 6.0 0.1 units lattice");
        command("delete_atoms region v1 compress no");
        command("delete_atoms region v2 compress no");
        command("set region as type 2");
        command("delete_atoms region rp compress no");
        command("create_atoms 1 single 6.0 6.0 6.0 units lattice");
        command("create_atoms 2 single 5.5 5.15 5.0 units lattice");
        command("displace_atoms all random 0.03 0.03 0.03 4711 units lattice");
        END_HIDE_OUTPUT();
    }

    Compute *run0()
    {
        BEGIN_HIDE_OUTPUT();
        command("run 0 post no");
        END_HIDE_OUTPUT();
        return lmp->modify->get_compute_by_id("ws");
    }

    // largest atom ID on any rank: the most recently created atom
    tagint max_tag()
    {
        tagint mine = 0;
        for (int i = 0; i < lmp->atom->nlocal; i++)
            mine = std::max(mine, lmp->atom->tag[i]);
        tagint all = 0;
        MPI_Allreduce(&mine, &all, 1, MPI_LMP_TAGINT, MPI_MAX, lmp->world);
        return all;
    }

    // per-atom row of the atom with ID "tag", only on the rank that owns it
    bool owned_row(Compute *c, tagint tag, std::vector<double> &row)
    {
        const int i = lmp->atom->map(tag);
        if ((i < 0) || (i >= lmp->atom->nlocal)) return false;
        row.assign(c->array_atom[i], c->array_atom[i] + c->size_peratom_cols);
        return true;
    }

    // vacancies per sublattice from the local array: v[j-1] = vacant sites of type j
    std::vector<int> vacancies_by_site_type(Compute *c)
    {
        const int ntypes = lmp->atom->ntypes;
        std::vector<int> one(ntypes, 0), all(ntypes, 0);
        c->compute_local();
        for (int r = 0; r < c->size_local_rows; r++)
            one[(int)c->array_local[r][1] - 1]++;
        MPI_Allreduce(one.data(), all.data(), ntypes, MPI_INT, MPI_SUM, lmp->world);
        return all;
    }

    // occupancy matrix from the per-atom array: m[(i-1)*ntypes + j-1] = number of
    // type-i atoms on sites of type j; "classical" restricts it to singly occupied sites
    std::vector<int> occupancy_matrix(Compute *c, bool classical = false)
    {
        const int ntypes = lmp->atom->ntypes;
        std::vector<int> one(ntypes * ntypes, 0), all(ntypes * ntypes, 0);
        c->compute_peratom();
        for (int i = 0; i < lmp->atom->nlocal; i++) {
            const double occ = c->array_atom[i][0];
            if ((occ < 1.0) || (classical && (occ != 1.0))) continue;
            one[(lmp->atom->type[i] - 1) * ntypes + (int)c->array_atom[i][2] - 1]++;
        }
        MPI_Allreduce(one.data(), all.data(), ntypes * ntypes, MPI_INT, MPI_SUM, lmp->world);
        return all;
    }

    // shared check of the known-defect configuration
    void check_known_defects(bool triclinic)
    {
        create_crystal(triclinic);
        define_compute("");
        run0(); // stores the reference: perfect crystal
        create_defects();

        BEGIN_CAPTURE_OUTPUT();
        command("run 0 post no");
        auto output = END_CAPTURE_OUTPUT();
        Compute *c  = lmp->modify->get_compute_by_id("ws");
        ASSERT_NE(c, nullptr);

        // the internal occupancy-sum check must stay silent
        EXPECT_THAT(output, Not(HasSubstr("site occupancies differ")));

        // global vector: exactly 4 intensive counts
        ASSERT_EQ(c->size_vector, 4);
        EXPECT_EQ(c->extvector, 0);
        EXPECT_DOUBLE_EQ(c->vector[0], 1.0); // interstitials
        EXPECT_DOUBLE_EQ(c->vector[1], 2.0); // vacancies
        EXPECT_DOUBLE_EQ(c->vector[2], 1.0); // replacements
        EXPECT_DOUBLE_EQ(c->vector[3], 1.0); // antisites
        // every atom counted exactly once: N_int - N_vac = N_group - N_sites
        EXPECT_DOUBLE_EQ(c->vector[0] - c->vector[1], (double)lmp->atom->natoms - NSITES);

        // no global array in either mode
        EXPECT_EQ(c->array_flag, 0);

        // local array: one vacancy on each sublattice
        EXPECT_EQ(c->size_local_cols, 5);
        const auto vac = vacancies_by_site_type(c);
        EXPECT_EQ(vac[0], 1);
        EXPECT_EQ(vac[1], 1);
        EXPECT_DOUBLE_EQ(vac[0] + vac[1], c->vector[1]);

        // occupancy matrix rebuilt from the per-atom array (row: atom type, column: site type)
        const auto m = occupancy_matrix(c);
        EXPECT_EQ(m[0], NSITES1 - 2); // type 1 on type-1 sites: minus vacancy and antisite
        EXPECT_EQ(m[1], 0);           // type 1 on type-2 sites
        EXPECT_EQ(m[2], 1);           // type 2 on type-1 sites: the antisite
        EXPECT_EQ(m[3], NSITES2);     // type 2 on type-2 sites: minus vacancy, plus interstitial

        // classical antisites from the per-atom array agree with vector element 4
        const auto a = occupancy_matrix(c, true);
        EXPECT_DOUBLE_EQ(a[1] + a[2], c->vector[3]);

        // per-atom array of the two created atoms (created last, consecutive IDs)
        ASSERT_EQ(c->size_peratom_cols, 3);
        c->compute_peratom();
        const tagint interstitial = max_tag();
        const tagint replacement  = interstitial - 1;
        std::vector<double> row;
        int found = 0;

        if (owned_row(c, interstitial, row)) {
            ++found;
            EXPECT_DOUBLE_EQ(row[0], 2.0);           // shares the cell with the original atom
            EXPECT_NE((tagint)row[1], interstitial); // site belongs to another atom
            EXPECT_DOUBLE_EQ(row[2], 2.0);           // face-center (type 2) site
        }
        if (owned_row(c, replacement, row)) {
            ++found;
            EXPECT_DOUBLE_EQ(row[0], 1.0);
            EXPECT_NE((tagint)row[1], replacement); // occupies the deleted atom's site
            EXPECT_DOUBLE_EQ(row[2], 1.0);
        }
        int allfound = 0;
        MPI_Allreduce(&found, &allfound, 1, MPI_INT, MPI_SUM, lmp->world);
        EXPECT_EQ(allfound, 2);
    }

    // atoms of the group with bitmask bit (IDs, types, positions) gathered
    // on every rank
    struct Sites {
        std::vector<tagint> tag;
        std::vector<int> type;
        std::vector<double> x;
    };

    Sites gather_sites(int bit = 1)
    {
        auto *atom       = lmp->atom;
        const int nprocs = lmp->comm->nprocs;
        std::vector<tagint> tlocal;
        std::vector<int> ylocal;
        std::vector<double> xlocal;
        for (int i = 0; i < atom->nlocal; i++) {
            if (!(atom->mask[i] & bit)) continue;
            tlocal.push_back(atom->tag[i]);
            ylocal.push_back(atom->type[i]);
            for (int d = 0; d < 3; d++)
                xlocal.push_back(atom->x[i][d]);
        }
        const int nlocal = tlocal.size();
        tlocal.reserve(1); // non-null buffers also on ranks without atoms
        ylocal.reserve(1);
        xlocal.reserve(3);
        std::vector<int> counts(nprocs), displs(nprocs), counts3(nprocs), displs3(nprocs);
        MPI_Allgather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, lmp->world);
        int ntotal = 0;
        for (int p = 0; p < nprocs; p++) {
            displs[p]  = ntotal;
            counts3[p] = 3 * counts[p];
            displs3[p] = 3 * ntotal;
            ntotal += counts[p];
        }
        Sites s;
        s.tag.resize(ntotal);
        s.type.resize(ntotal);
        s.x.resize(3 * ntotal);
        MPI_Allgatherv(tlocal.data(), nlocal, MPI_LMP_TAGINT, s.tag.data(), counts.data(),
                       displs.data(), MPI_LMP_TAGINT, lmp->world);
        MPI_Allgatherv(ylocal.data(), nlocal, MPI_INT, s.type.data(), counts.data(), displs.data(),
                       MPI_INT, lmp->world);
        MPI_Allgatherv(xlocal.data(), 3 * nlocal, MPI_DOUBLE, s.x.data(), counts3.data(),
                       displs3.data(), MPI_DOUBLE, lmp->world);
        return s;
    }

    // index of the reference site nearest to x, searched over all sites and
    // their 27 nearest periodic images (periodic dimensions only)
    int brute_force_site(const Sites &ref, const double *x)
    {
        const double *h    = lmp->domain->h;
        const double *hinv = lmp->domain->h_inv;
        const int per[3] = {lmp->domain->xperiodic, lmp->domain->yperiodic, lmp->domain->zperiodic};
        const int nsite  = ref.tag.size();
        double best      = 1.0e300;
        int jbest        = -1;
        for (int j = 0; j < nsite; j++) {
            double d[3] = {x[0] - ref.x[3 * j], x[1] - ref.x[3 * j + 1], x[2] - ref.x[3 * j + 2]};
            double l[3] = {hinv[0] * d[0] + hinv[5] * d[1] + hinv[4] * d[2],
                           hinv[1] * d[1] + hinv[3] * d[2], hinv[2] * d[2]};
            for (int k = 0; k < 3; k++)
                if (per[k]) l[k] -= std::round(l[k]);
            for (int na = -per[0]; na <= per[0]; na++) {
                for (int nb = -per[1]; nb <= per[1]; nb++) {
                    for (int nc = -per[2]; nc <= per[2]; nc++) {
                        const double la = l[0] + na, lb = l[1] + nb, lc = l[2] + nc;
                        const double rx  = h[0] * la + h[5] * lb + h[4] * lc;
                        const double ry  = h[1] * lb + h[3] * lc;
                        const double rz  = h[2] * lc;
                        const double rsq = rx * rx + ry * ry + rz * rz;
                        if (rsq < best) {
                            best  = rsq;
                            jbest = j;
                        }
                    }
                }
            }
        }
        return jbest;
    }

    // compare all outputs of the compute with a brute-force assignment of
    // every group atom to its nearest reference site; atoms outside the
    // group must have all-zero per-atom values
    void check_brute_force(Compute *c, const Sites &ref, int bit = 1)
    {
        auto *atom       = lmp->atom;
        const int nlocal = atom->nlocal;
        const int nsite  = ref.tag.size();

        std::vector<int> site(nlocal, -1);
        std::vector<int> occ(nsite, 0), typesum(nsite, 0);
        std::vector<tagint> tagsum(nsite, 0);
        for (int i = 0; i < nlocal; i++) {
            if (!(atom->mask[i] & bit)) continue;
            site[i] = brute_force_site(ref, atom->x[i]);
            occ[site[i]]++;
            tagsum[site[i]] += atom->tag[i];
            typesum[site[i]] += atom->type[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, occ.data(), nsite, MPI_INT, MPI_SUM, lmp->world);
        MPI_Allreduce(MPI_IN_PLACE, tagsum.data(), nsite, MPI_LMP_TAGINT, MPI_SUM, lmp->world);
        MPI_Allreduce(MPI_IN_PLACE, typesum.data(), nsite, MPI_INT, MPI_SUM, lmp->world);

        double expect[4] = {0.0, 0.0, 0.0, 0.0};
        double vacsum    = 0.0;
        for (int j = 0; j < nsite; j++) {
            if (occ[j] == 0) {
                expect[1] += 1.0;
                vacsum += ref.tag[j];
            } else {
                expect[0] += occ[j] - 1;
                if ((occ[j] == 1) && (tagsum[j] != ref.tag[j])) expect[2] += 1.0;
                if ((occ[j] == 1) && (typesum[j] != ref.type[j])) expect[3] += 1.0;
            }
        }
        for (int k = 0; k < 4; k++)
            EXPECT_DOUBLE_EQ(c->vector[k], expect[k]) << "vector element " << k + 1;

        c->compute_peratom();
        int nbad = 0;
        for (int i = 0; i < nlocal; i++) {
            const double *row = c->array_atom[i];
            const int j       = site[i];
            if (j < 0) {
                if ((row[0] != 0.0) || (row[1] != 0.0) || (row[2] != 0.0)) nbad++;
            } else if ((row[0] != occ[j]) || (row[1] != ref.tag[j]) || (row[2] != ref.type[j])) {
                nbad++;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &nbad, 1, MPI_INT, MPI_SUM, lmp->world);
        EXPECT_EQ(nbad, 0) << "atoms with per-atom values different from the brute-force search";

        c->compute_local();
        double rows[2] = {(double)c->size_local_rows, 0.0};
        for (int r = 0; r < c->size_local_rows; r++)
            rows[1] += c->array_local[r][0];
        MPI_Allreduce(MPI_IN_PLACE, rows, 2, MPI_DOUBLE, MPI_SUM, lmp->world);
        EXPECT_DOUBLE_EQ(rows[0], expect[1]);
        EXPECT_DOUBLE_EQ(rows[1], vacsum);
    }

    // global sums of the per-atom and local outputs: occupancy, atom ID times
    // site ID, site type, number of vacancies, sum of their IDs, and the sums
    // of their fractional coordinates in the current box (wrapped into [0,1))
    std::vector<double> checksums(Compute *c)
    {
        auto *atom   = lmp->atom;
        auto *domain = lmp->domain;
        std::vector<double> s(8, 0.0);
        c->compute_peratom();
        for (int i = 0; i < atom->nlocal; i++) {
            s[0] += c->array_atom[i][0];
            s[1] += (double)atom->tag[i] * c->array_atom[i][1];
            s[2] += c->array_atom[i][2];
        }
        c->compute_local();
        const double *hinv = domain->h_inv;
        for (int r = 0; r < c->size_local_rows; r++) {
            const double *row = c->array_local[r];
            const double d[3] = {row[2] - domain->boxlo[0], row[3] - domain->boxlo[1],
                                 row[4] - domain->boxlo[2]};
            double l[3]       = {hinv[0] * d[0] + hinv[5] * d[1] + hinv[4] * d[2],
                                 hinv[1] * d[1] + hinv[3] * d[2], hinv[2] * d[2]};
            s[3] += 1.0;
            s[4] += row[0];
            for (int k = 0; k < 3; k++)
                s[5 + k] += l[k] - std::floor(l[k]);
        }
        MPI_Allreduce(MPI_IN_PLACE, s.data(), 8, MPI_DOUBLE, MPI_SUM, lmp->world);
        return s;
    }

    void expect_same_outputs(Compute *c, const std::vector<double> &vec,
                             const std::vector<double> &sums)
    {
        for (int k = 0; k < 4; k++)
            EXPECT_DOUBLE_EQ(c->vector[k], vec[k]) << "vector element " << k + 1;
        const auto now = checksums(c);
        for (int k = 0; k < 5; k++)
            EXPECT_DOUBLE_EQ(now[k], sums[k]) << "checksum " << k;
        for (int k = 5; k < 8; k++)
            EXPECT_NEAR(now[k], sums[k], 1.0e-9) << "checksum " << k;
    }

    // processor grid dimensions with more than one rank, as balance arguments
    // that move the cuts toward the lower box boundary, the middle cut by a
    // quarter of the box length (more than the ghost cutoff)
    std::string balance_cuts()
    {
        std::string args;
        const char *dim[3] = {"x", "y", "z"};
        for (int d = 0; d < 3; d++) {
            const int p = lmp->comm->procgrid[d];
            if (p < 2) continue;
            args += std::string(" ") + dim[d];
            for (int i = 1; i < p; i++)
                args += " " + std::to_string(0.5 * i / p);
        }
        return args;
    }

    // reference file of the current atoms with all digits of the positions
    void write_reference(const std::string &columns = "x y z")
    {
        BEGIN_HIDE_OUTPUT();
        command("write_dump all custom ws_site.dump id type " + columns +
                " modify format float %20.15g");
        END_HIDE_OUTPUT();
    }

    void remove_file(const std::string &name)
    {
        MPI_Barrier(lmp->world);
        if (lmp->comm->me == 0) std::remove(name.c_str());
    }

    // replace the text of a file on rank 0 by edit(text)
    template <typename F> void edit_file(const std::string &name, F edit)
    {
        MPI_Barrier(lmp->world);
        if (lmp->comm->me == 0) {
            std::ifstream in(name);
            std::stringstream text;
            text << in.rdbuf();
            in.close();
            std::ofstream out(name);
            out << edit(text.str());
        }
        MPI_Barrier(lmp->world);
    }

    // position of the atom with ID tag, on every rank
    std::vector<double> atom_position(tagint tag)
    {
        std::vector<double> xt(3, 0.0);
        const int i = lmp->atom->map(tag);
        if ((i >= 0) && (i < lmp->atom->nlocal))
            for (int d = 0; d < 3; d++)
                xt[d] = lmp->atom->x[i][d];
        MPI_Allreduce(MPI_IN_PLACE, xt.data(), 3, MPI_DOUBLE, MPI_SUM, lmp->world);
        return xt;
    }

    // thermal noise on all atoms, and every 7th atom displaced by up to 0.7
    // lattice spacings, so that many atoms leave the cell of their own site
    // or cross a periodic boundary
    Compute *displace_and_run(const Sites &ref, int bit = 1)
    {
        BEGIN_HIDE_OUTPUT();
        command("variable far atom \"id % 7 == 0\"");
        command("group far variable far");
        command("displace_atoms all random 0.05 0.05 0.05 8123 units lattice");
        command("displace_atoms far random 0.7 0.7 0.7 4711 units lattice");
        END_HIDE_OUTPUT();

        BEGIN_CAPTURE_OUTPUT();
        command("run 0 post no");
        auto output = END_CAPTURE_OUTPUT();
        EXPECT_THAT(output, Not(HasSubstr("site occupancies differ")));

        Compute *c = lmp->modify->get_compute_by_id("ws");
        if (c) check_brute_force(c, ref, bit);
        return c;
    }
};

// ---------------------------------------------------------------------------
// physics
// ---------------------------------------------------------------------------

TEST_F(ComputeWSTest, PerfectCrystalWithNoise)
{
    create_crystal(false);
    define_compute("");
    run0();
    BEGIN_HIDE_OUTPUT();
    command("displace_atoms all random 0.05 0.05 0.05 8123 units lattice");
    END_HIDE_OUTPUT();
    Compute *c = run0();
    ASSERT_NE(c, nullptr);

    for (int k = 0; k < 4; k++)
        EXPECT_DOUBLE_EQ(c->vector[k], 0.0) << "vector element " << k + 1;
    EXPECT_EQ(c->size_vector, 4);
    EXPECT_EQ(c->size_peratom_cols, 3);

    const auto m = occupancy_matrix(c);
    EXPECT_EQ(m[0], NSITES1);
    EXPECT_EQ(m[1] + m[2], 0);
    EXPECT_EQ(m[3], NSITES2);
    const auto vac = vacancies_by_site_type(c);
    EXPECT_EQ(vac[0] + vac[1], 0);
}

TEST_F(ComputeWSTest, KnownDefectsOrthogonal)
{
    check_known_defects(false);
}

TEST_F(ComputeWSTest, KnownDefectsTriclinic)
{
    check_known_defects(true);
}

// the documented filter for dumping defective atoms selects exactly the atoms
// of the known defects: 2 atoms sharing the interstitial site, the antisite,
// and the replacement atom
TEST_F(ComputeWSTest, DefectFilter)
{
    create_crystal(false);
    BEGIN_HIDE_OUTPUT();
    command("compute ws all ws");
    command("variable def atom \"c_ws[1] >= 2 || type != c_ws[3] || id != c_ws[2]\"");
    command("compute ndef all reduce sum v_def");
    command("thermo_style custom step c_ws[1] c_ws[2] c_ndef");
    END_HIDE_OUTPUT();
    run0();
    Compute *n = lmp->modify->get_compute_by_id("ndef");
    ASSERT_NE(n, nullptr);
    EXPECT_DOUBLE_EQ(n->scalar, 0.0); // perfect crystal: nothing selected

    create_defects();
    run0();
    EXPECT_DOUBLE_EQ(n->scalar, 4.0);
}

// the compute is invoked by a dynamic group (at setup before the compute
// itself is set up, and after the forces of every step) and by the thermo
// output of the same timestep; a minimization (where the group is set only
// before the compute is set up) at the same timestep after the atoms
// changed must not reuse the analysis of the previous run
TEST_F(ComputeWSTest, DynamicGroupSameTimestep)
{
    create_crystal(false);
    BEGIN_HIDE_OUTPUT();
    command("compute ws all ws");
    command("variable def atom \"c_ws[1] >= 2 || type != c_ws[3] || id != c_ws[2]\"");
    command("group def dynamic all var def every 1");
    command("thermo_style custom step c_ws[1] c_ws[2] c_ws[3] c_ws[4]");
    command("thermo 1");
    END_HIDE_OUTPUT();
    Compute *c = run0();
    ASSERT_NE(c, nullptr);
    const int idef = lmp->group->find("def");
    EXPECT_EQ(lmp->group->count(idef), 0);

    create_defects();
    BEGIN_HIDE_OUTPUT();
    command("minimize 0.0 0.0 0 0");
    END_HIDE_OUTPUT();
    EXPECT_EQ(lmp->group->count(idef), 4);
    EXPECT_DOUBLE_EQ(c->vector[0], 1.0);
    EXPECT_DOUBLE_EQ(c->vector[1], 2.0);

    BEGIN_HIDE_OUTPUT();
    command("run 2 post no");
    END_HIDE_OUTPUT();
    EXPECT_EQ(lmp->group->count(idef), 4);
    EXPECT_DOUBLE_EQ(c->vector[2], 1.0);
    EXPECT_DOUBLE_EQ(c->vector[3], 1.0);
}

TEST_F(ComputeWSTest, FrenkelPairAcrossPeriodicBoundary)
{
    create_crystal(false);
    define_compute("");
    run0();

    // move the corner atom at the origin into the cell of the face-center
    // site (0.5,-0.5,0), which lies across the periodic y boundary
    BEGIN_HIDE_OUTPUT();
    command("region fp sphere 0.0 0.0 0.0 0.1 units lattice");
    command("group fp region fp");
    command("displace_atoms fp move 0.5 -0.35 0.0 units lattice");
    END_HIDE_OUTPUT();
    Compute *c = run0();
    ASSERT_NE(c, nullptr);
    EXPECT_DOUBLE_EQ(c->vector[0], 1.0);
    EXPECT_DOUBLE_EQ(c->vector[1], 1.0);
}

// a new reference is taken by defining the compute again with the same ID;
// thermo output finds the new compute without changes
TEST_F(ComputeWSTest, RedefineForNewReference)
{
    create_crystal(false);
    define_compute("");
    run0();
    create_defects();
    Compute *c = run0();
    ASSERT_NE(c, nullptr);
    EXPECT_DOUBLE_EQ(c->vector[1], 2.0);

    BEGIN_HIDE_OUTPUT();
    command("uncompute ws");
    command("compute ws all ws");
    END_HIDE_OUTPUT();
    c = run0();
    ASSERT_NE(c, nullptr);
    for (int k = 0; k < 4; k++)
        EXPECT_DOUBLE_EQ(c->vector[k], 0.0) << "vector element " << k + 1;

    BEGIN_HIDE_OUTPUT();
    command("region fp sphere 4.0 4.0 4.0 0.1 units lattice");
    command("group fp region fp");
    command("displace_atoms fp move 0.5 0.15 0.0 units lattice");
    END_HIDE_OUTPUT();
    c = run0();
    ASSERT_NE(c, nullptr);
    EXPECT_DOUBLE_EQ(c->vector[0], 1.0);
    EXPECT_DOUBLE_EQ(c->vector[1], 1.0);
}

// ---------------------------------------------------------------------------
// affine mapping into the reference box
// ---------------------------------------------------------------------------

// homogeneous strain between runs: all outputs as before, vacancy positions
// at the same fractional coordinates of the new box
TEST_F(ComputeWSTest, AffineChangeBoxBetweenRuns)
{
    create_crystal(false);
    define_compute("affine yes");
    run0();
    create_defects();
    Compute *c = run0();
    ASSERT_NE(c, nullptr);
    EXPECT_DOUBLE_EQ(c->vector[1], 2.0);
    const std::vector<double> vec(c->vector, c->vector + 4);
    const auto sums = checksums(c);

    BEGIN_HIDE_OUTPUT();
    command("change_box all x scale 1.02 y scale 0.98 z scale 1.01 remap");
    END_HIDE_OUTPUT();
    c = run0();
    ASSERT_NE(c, nullptr);
    expect_same_outputs(c, vec, sums);
}

TEST_F(ComputeWSTest, AffineShearTriclinic)
{
    create_crystal(true);
    define_compute("affine yes");
    run0();
    create_defects();
    Compute *c = run0();
    ASSERT_NE(c, nullptr);
    const std::vector<double> vec(c->vector, c->vector + 4);
    const auto sums = checksums(c);

    BEGIN_HIDE_OUTPUT();
    command("change_box all xy final 1.0 xz final -0.5 yz final 0.6 x scale 1.01 remap units box");
    END_HIDE_OUTPUT();
    c = run0();
    ASSERT_NE(c, nullptr);
    expect_same_outputs(c, vec, sums);
}

// box deformed during the run (atoms remapped with the box, no time integration)
TEST_F(ComputeWSTest, AffineDeformDuringRun)
{
    create_crystal(false);
    define_compute("affine yes");
    run0();
    create_defects();
    Compute *c = run0();
    ASSERT_NE(c, nullptr);
    const std::vector<double> vec(c->vector, c->vector + 4);
    const auto sums = checksums(c);

    BEGIN_HIDE_OUTPUT();
    command("fix 1 all deform 1 x scale 1.03 y scale 0.98 z scale 1.01 remap x");
    command("run 20 post no");
    END_HIDE_OUTPUT();
    expect_same_outputs(c, vec, sums);
}

// shear by 40% of the box length during the run (thermo output every 5 steps)
TEST_F(ComputeWSTest, AffineLargeShear)
{
    create_crystal(true);
    define_compute("affine yes");
    run0();
    create_defects();
    Compute *c = run0();
    ASSERT_NE(c, nullptr);
    const std::vector<double> vec(c->vector, c->vector + 4);
    const auto sums = checksums(c);

    const double xyfinal = 0.4 * lmp->domain->xprd;
    BEGIN_HIDE_OUTPUT();
    command("fix 1 all deform 1 xy final " + std::to_string(xyfinal) +
            " remap x flip no units box");
    command("thermo 5");
    command("run 100 post no");
    END_HIDE_OUTPUT();
    EXPECT_NEAR(lmp->domain->xy, xyfinal, 1.0e-6);
    expect_same_outputs(c, vec, sums);
}

// shear by 70% of the box length flips the triclinic box during the run
TEST_F(ComputeWSTest, AffineBoxFlipRejected)
{
    create_crystal(true);
    define_compute("affine yes");
    run0();
    const double xyfinal = 0.7 * lmp->domain->xprd;
    BEGIN_HIDE_OUTPUT();
    command("fix 1 all deform 1 xy final " + std::to_string(xyfinal) +
            " remap x flip yes units box");
    command("thermo 5");
    END_HIDE_OUTPUT();
    TEST_FAILURE_MPI(".*ERROR: Compute ws with 'affine yes': a box tilt changed by "
                     "more than half a box length.*",
                     command("run 100 post no"););
}

// tilted reference box away from the origin, then sheared, scaled and
// shifted: the vacancy is reported at the position of a tracer atom (not in
// the compute group) that took the place of the deleted atom
TEST_F(ComputeWSTest, AffineOffsetTiltedBox)
{
    BEGIN_HIDE_OUTPUT();
    command("units lj");
    command("atom_modify map array");
    command("lattice fcc 0.8442");
    command("region box block 0 8 0 8 0 8");
    command("create_box 3 box");
    command("create_atoms 1 box basis 1 1 basis 2 2 basis 3 2 basis 4 2");
    command("change_box all triclinic");
    command("change_box all xy final 1.3 xz final -0.9 yz final 0.7 x delta 20.0 20.0 "
            "y delta -15.0 -15.0 z delta 30.0 30.0 remap units box");
    command("mass * 1.0");
    command("pair_style zero 2.5");
    command("pair_coeff * *");
    command("group crystal type 1 2");
    command("compute ws crystal ws affine yes");
    command("thermo_style custom step c_ws[1] c_ws[2] c_ws[3] c_ws[4]");
    END_HIDE_OUTPUT();
    run0();

    // the atom closest to the box center
    auto *domain = lmp->domain;
    double center[3];
    double half[3] = {0.5, 0.5, 0.5};
    domain->lamda2x(half, center);
    struct {
        double rsq;
        int tag;
    } mine = {1.0e300, 0}, best;
    for (int i = 0; i < lmp->atom->nlocal; i++) {
        const double *xi = lmp->atom->x[i];
        const double rsq = (xi[0] - center[0]) * (xi[0] - center[0]) +
                           (xi[1] - center[1]) * (xi[1] - center[1]) +
                           (xi[2] - center[2]) * (xi[2] - center[2]);
        if (rsq < mine.rsq) mine = {rsq, (int)lmp->atom->tag[i]};
    }
    MPI_Allreduce(&mine, &best, 1, MPI_DOUBLE_INT, MPI_MINLOC, lmp->world);
    const auto x0 = atom_position(best.tag);
    BEGIN_HIDE_OUTPUT();
    command("group del id " + std::to_string(best.tag));
    command("delete_atoms group del compress no");
    command(fmt::format("create_atoms 3 single {:.15g} {:.15g} {:.15g} units box", x0[0], x0[1],
                        x0[2]));
    command("change_box all xy final 1.6 yz final 0.4 x scale 1.05 y scale 0.97 "
            "x delta 0.5 0.5 z delta -0.7 -0.7 remap units box");
    END_HIDE_OUTPUT();
    Compute *c = run0();
    ASSERT_NE(c, nullptr);
    EXPECT_DOUBLE_EQ(c->vector[0], 0.0);
    EXPECT_DOUBLE_EQ(c->vector[1], 1.0);

    const auto xt = atom_position(max_tag());
    std::vector<double> xv(4, 0.0);
    c->compute_local();
    for (int r = 0; r < c->size_local_rows; r++) {
        xv[0] += c->array_local[r][0];
        for (int d = 0; d < 3; d++)
            xv[1 + d] += c->array_local[r][2 + d];
    }
    MPI_Allreduce(MPI_IN_PLACE, xv.data(), 4, MPI_DOUBLE, MPI_SUM, lmp->world);
    EXPECT_DOUBLE_EQ(xv[0], (double)best.tag);
    for (int d = 0; d < 3; d++)
        EXPECT_NEAR(xv[1 + d], xt[d], 1.0e-9) << "coordinate " << d;
}

TEST_F(ComputeWSTest, AffineTiltWarning)
{
    create_crystal(true);
    define_compute("affine yes");
    BEGIN_HIDE_OUTPUT();
    command("fix 1 all deform 1 xy delta 0.1 remap x");
    END_HIDE_OUTPUT();
    BEGIN_CAPTURE_OUTPUT();
    command("run 0 post no");
    auto output = END_CAPTURE_OUTPUT();
    if (lmp->comm->me == 0) {
        EXPECT_THAT(output, HasSubstr("does not support box flips"));
    }
}

// ---------------------------------------------------------------------------
// reference sites from a dump file
// ---------------------------------------------------------------------------

// sites from the file of a homogeneously strained crystal, mapped into the
// current box: the known defects
TEST_F(ComputeWSTest, ReferenceFileAffine)
{
    create_crystal(false);
    write_reference();
    create_defects();
    BEGIN_HIDE_OUTPUT();
    command("change_box all x scale 1.02 y scale 1.01 z scale 0.99 remap");
    END_HIDE_OUTPUT();
    define_compute("reference ws_site.dump affine yes");

    BEGIN_CAPTURE_OUTPUT();
    command("run 0 post no");
    auto output = END_CAPTURE_OUTPUT();
    Compute *c  = lmp->modify->get_compute_by_id("ws");
    ASSERT_NE(c, nullptr);
    if (lmp->comm->me == 0) {
        EXPECT_THAT(output,
                    HasSubstr(std::to_string(NSITES) + " reference sites read from ws_site.dump"));
    }
    EXPECT_THAT(output, Not(HasSubstr("site occupancies differ")));
    EXPECT_DOUBLE_EQ(c->vector[0], 1.0);
    EXPECT_DOUBLE_EQ(c->vector[1], 2.0);
    EXPECT_DOUBLE_EQ(c->vector[2], 1.0);
    EXPECT_DOUBLE_EQ(c->vector[3], 1.0);
    remove_file("ws_site.dump");
}

TEST_F(ComputeWSTest, ReferenceFileBoxMismatch)
{
    create_crystal(false);
    BEGIN_HIDE_OUTPUT();
    command("write_dump all custom ws_site.dump id type x y z");
    command("change_box all x scale 1.02 remap");
    END_HIDE_OUTPUT();
    define_compute("reference ws_site.dump");
    TEST_FAILURE_MPI(".*ERROR: Box in compute ws reference file ws_site.dump differs.*",
                     command("run 0 post no"););
    remove_file("ws_site.dump");
}

// the reference survives a restart: the same counts after read_restart,
// whereas a reference taken from the atoms is the damaged crystal
TEST_F(ComputeWSTest, ReferenceFileAfterRestart)
{
    create_crystal(false);
    write_reference();
    define_compute("reference ws_site.dump");
    run0();
    create_defects();
    Compute *c = run0();
    ASSERT_NE(c, nullptr);
    const std::vector<double> vec(c->vector, c->vector + 4);
    EXPECT_DOUBLE_EQ(vec[1], 2.0);

    BEGIN_HIDE_OUTPUT();
    command("write_restart ws_test.restart");
    command("clear");
    command("read_restart ws_test.restart");
    command("compute ws all ws reference ws_site.dump");
    command("compute ws2 all ws");
    command("thermo_style custom step c_ws[1] c_ws[2] c_ws[3] c_ws[4] c_ws2[1] c_ws2[2]");
    END_HIDE_OUTPUT();
    c = run0();
    ASSERT_NE(c, nullptr);
    for (int k = 0; k < 4; k++)
        EXPECT_DOUBLE_EQ(c->vector[k], vec[k]) << "vector element " << k + 1;
    Compute *c2 = lmp->modify->get_compute_by_id("ws2");
    ASSERT_NE(c2, nullptr);
    EXPECT_DOUBLE_EQ(c2->vector[0], 0.0);
    EXPECT_DOUBLE_EQ(c2->vector[1], 0.0);
    remove_file("ws_site.dump");
    remove_file("ws_test.restart");
}

// a larger communication cutoff after the reference was read: the file is
// read again with the larger ghost shell
TEST_F(ComputeWSTest, ReferenceFileCutoffIncrease)
{
    create_crystal(false);
    BEGIN_HIDE_OUTPUT();
    command("pair_style zero 0.3");
    command("pair_coeff * *");
    command("neighbor 0.0 bin");
    END_HIDE_OUTPUT();
    write_reference();
    define_compute("reference ws_site.dump");
    run0();
    BEGIN_HIDE_OUTPUT();
    command("comm_modify cutoff 3.0");
    command("displace_atoms all random 0.05 0.05 0.05 8123 units lattice");
    END_HIDE_OUTPUT();
    BEGIN_CAPTURE_OUTPUT();
    command("run 0 post no");
    auto output = END_CAPTURE_OUTPUT();
    Compute *c  = lmp->modify->get_compute_by_id("ws");
    ASSERT_NE(c, nullptr);
    EXPECT_THAT(output, Not(HasSubstr("site occupancies differ")));
    for (int k = 0; k < 4; k++)
        EXPECT_DOUBLE_EQ(c->vector[k], 0.0) << "vector element " << k + 1;
    remove_file("ws_site.dump");
}

// the same for a reference taken from the atoms: a warning
TEST_F(ComputeWSTest, CutoffIncreaseWarning)
{
    create_crystal(false);
    BEGIN_HIDE_OUTPUT();
    command("pair_style zero 0.3");
    command("pair_coeff * *");
    command("neighbor 0.0 bin");
    END_HIDE_OUTPUT();
    define_compute("");
    run0();
    BEGIN_HIDE_OUTPUT();
    command("comm_modify cutoff 3.0");
    END_HIDE_OUTPUT();
    BEGIN_CAPTURE_OUTPUT();
    command("run 0 post no");
    auto output = END_CAPTURE_OUTPUT();
    if (lmp->comm->me == 0) {
        EXPECT_THAT(output, HasSubstr("communication cutoff increased after the reference"));
    }
}

// unwrapped z with wrapped x and y
TEST_F(ComputeWSTest, ReferenceFileMixedWrapping)
{
    create_crystal(false);
    write_reference("x y zu");
    define_compute("reference ws_site.dump");
    Compute *c = run0();
    ASSERT_NE(c, nullptr);
    for (int k = 0; k < 4; k++)
        EXPECT_DOUBLE_EQ(c->vector[k], 0.0) << "vector element " << k + 1;
    remove_file("ws_site.dump");
}

// a site outside a non-periodic box is owned by the last rank in that
// dimension and is vacant
TEST_F(ComputeWSTest, ReferenceFileSiteOutsideNonPeriodicBox)
{
    BEGIN_HIDE_OUTPUT();
    command("units lj");
    command("atom_modify map array");
    command("boundary p p f");
    if (lmp->comm->nprocs > 1) command("processors 1 1 *");
    command("lattice fcc 0.8442");
    command("region box block 0 4 0 4 0 4");
    command("create_box 1 box");
    command("create_atoms 1 box");
    command("mass 1 1.0");
    command("pair_style zero 2.5");
    command("pair_coeff * *");
    END_HIDE_OUTPUT();
    write_reference();
    const double zout = lmp->domain->boxhi[2] + 0.5;
    edit_file("ws_site.dump", [&](const std::string &text) {
        std::string t         = text;
        const std::string key = "ITEM: NUMBER OF ATOMS\n";
        auto p                = t.find(key) + key.size();
        auto q                = t.find('\n', p);
        const long n          = std::stol(t.substr(p, q - p));
        t.replace(p, q - p, std::to_string(n + 1));
        return t + fmt::format("99999 1 0.5 0.5 {:.15g}\n", zout);
    });
    BEGIN_HIDE_OUTPUT();
    command("compute ws all ws reference ws_site.dump");
    command("thermo_style custom step c_ws[1] c_ws[2]");
    END_HIDE_OUTPUT();
    Compute *c = run0();
    ASSERT_NE(c, nullptr);
    EXPECT_DOUBLE_EQ(c->vector[0], 0.0);
    EXPECT_DOUBLE_EQ(c->vector[1], 1.0);
    remove_file("ws_site.dump");
}

TEST_F(ComputeWSTest, ReferenceFileNoColumnLabels)
{
    create_crystal(false);
    BEGIN_HIDE_OUTPUT();
    command("write_dump all custom ws_bad.dump id type x y z");
    END_HIDE_OUTPUT();
    edit_file("ws_bad.dump", [](const std::string &text) {
        std::string t         = text;
        const std::string key = "ITEM: ATOMS id type x y z";
        t.replace(t.find(key), key.size(), "ITEM: ATOMS");
        return t;
    });
    define_compute("reference ws_bad.dump");
    TEST_FAILURE_MPI(".*ERROR: Compute ws reference file ws_bad.dump must contain the "
                     "columns id, type, x, y, z.*",
                     command("run 0 post no"););
    remove_file("ws_bad.dump");
}

// a dynamic group invokes the compute during setup before the compute is set
// up; after balance the file must already be read again for that invocation,
// also in a minimization (where the group is set only once)
TEST_F(ComputeWSTest, ReferenceFileBalanceDynamicGroupMinimize)
{
    create_crystal(false);
    const auto cuts = balance_cuts();
    if (cuts.empty()) GTEST_SKIP() << "needs more than one MPI rank";
    write_reference();
    BEGIN_HIDE_OUTPUT();
    command("compute ws all ws reference ws_site.dump");
    command("variable def atom \"c_ws[1] >= 2 || type != c_ws[3] || id != c_ws[2]\"");
    command("group def dynamic all var def every 1");
    command("thermo_style custom step c_ws[1] c_ws[2] c_ws[3] c_ws[4]");
    END_HIDE_OUTPUT();
    run0();
    const int idef = lmp->group->find("def");
    EXPECT_EQ(lmp->group->count(idef), 0);

    BEGIN_HIDE_OUTPUT();
    command("balance 0.9" + cuts);
    END_HIDE_OUTPUT();
    BEGIN_CAPTURE_OUTPUT();
    command("minimize 0.0 0.0 0 0");
    auto output = END_CAPTURE_OUTPUT();
    EXPECT_THAT(output, Not(HasSubstr("site occupancies differ")));
    EXPECT_EQ(lmp->group->count(idef), 0);
    remove_file("ws_site.dump");
}

// sub-domains changed between runs: the file is read again, outputs as before
TEST_F(ComputeWSTest, ReferenceFileBalanceBetweenRuns)
{
    create_crystal(false);
    const auto cuts = balance_cuts();
    if (cuts.empty()) GTEST_SKIP() << "needs more than one MPI rank";
    write_reference();
    define_compute("reference ws_site.dump");
    run0();
    create_defects();
    Compute *c = run0();
    ASSERT_NE(c, nullptr);
    const std::vector<double> vec(c->vector, c->vector + 4);
    const auto sums = checksums(c);

    BEGIN_HIDE_OUTPUT();
    command("balance 0.9" + cuts);
    END_HIDE_OUTPUT();
    c = run0();
    ASSERT_NE(c, nullptr);
    expect_same_outputs(c, vec, sums);
    remove_file("ws_site.dump");
}

TEST_F(ComputeWSTest, Rerun)
{
    create_crystal(false);
    BEGIN_HIDE_OUTPUT();
    command("write_dump all custom ws_ref.dump id type x y z");
    command("region fp sphere 4.0 4.0 4.0 0.1 units lattice");
    command("group fp region fp");
    command("displace_atoms fp move 0.5 0.15 0.0 units lattice");
    command("reset_timestep 1");
    command("write_dump all custom ws_frenkel.dump id type x y z");
    END_HIDE_OUTPUT();
    define_compute("");

    // the first snapshot (perfect crystal) becomes the reference
    BEGIN_HIDE_OUTPUT();
    command("thermo 1");
    command("rerun ws_ref.dump ws_frenkel.dump dump x y z");
    END_HIDE_OUTPUT();
    Compute *c = lmp->modify->get_compute_by_id("ws");
    ASSERT_NE(c, nullptr);
    EXPECT_DOUBLE_EQ(c->vector[0], 1.0);
    EXPECT_DOUBLE_EQ(c->vector[1], 1.0);

    MPI_Barrier(lmp->world);
    if (lmp->comm->me == 0) {
        std::remove("ws_ref.dump");
        std::remove("ws_frenkel.dump");
    }
}

// ---------------------------------------------------------------------------
// diagnostics and errors
// ---------------------------------------------------------------------------

TEST_F(ComputeWSTest, ThinGhostShellWarning)
{
    create_crystal(false);
    BEGIN_HIDE_OUTPUT();
    command("pair_style zero 0.2");
    command("pair_coeff * *");
    command("neighbor 0.0 bin");
    END_HIDE_OUTPUT();
    define_compute("");

    BEGIN_CAPTURE_OUTPUT();
    command("run 0 post no");
    auto output = END_CAPTURE_OUTPUT();
    if (lmp->comm->me == 0) {
        EXPECT_THAT(output, HasSubstr("communication cutoff"));
    }
}

TEST_F(ComputeWSTest, GeometryChangeBetweenRuns)
{
    create_crystal(false);
    define_compute("");
    run0();
    BEGIN_HIDE_OUTPUT();
    command("change_box all x scale 1.01 remap");
    END_HIDE_OUTPUT();
    TEST_FAILURE_MPI(".*ERROR: Box changed since compute ws stored its reference.*",
                     command("run 0 post no"););
}

TEST_F(ComputeWSTest, SubdomainChangeBetweenRuns)
{
    create_crystal(false);
    const auto cuts = balance_cuts();
    if (cuts.empty()) GTEST_SKIP() << "needs more than one MPI rank";
    define_compute("affine yes");
    run0();
    BEGIN_HIDE_OUTPUT();
    command("balance 0.9" + cuts);
    END_HIDE_OUTPUT();
    TEST_FAILURE_MPI(".*ERROR: Processor sub-domains changed since compute ws.*",
                     command("run 0 post no"););
}

TEST_F(ComputeWSTest, StaticBoxRequired)
{
    create_crystal(false);
    define_compute("");
    BEGIN_HIDE_OUTPUT();
    command("fix 1 all deform 1 x scale 1.01");
    END_HIDE_OUTPUT();
    TEST_FAILURE_MPI(".*ERROR: Compute ws requires a static simulation box.*",
                     command("run 0 post no"););
}

TEST_F(ComputeWSTest, FixBalanceRejected)
{
    create_crystal(false);
    define_compute("affine yes");
    BEGIN_HIDE_OUTPUT();
    command("fix 2 all balance 10 1.1 shift x 5 1.1");
    END_HIDE_OUTPUT();
    TEST_FAILURE_MPI(".*ERROR: Compute ws requires static processor sub-domains.*",
                     command("run 0 post no"););
}

TEST_F(ComputeWSTest, ShrinkWrapRejected)
{
    BEGIN_HIDE_OUTPUT();
    command("units lj");
    command("boundary p p m");
    command("lattice fcc 0.8442");
    command("region box block 0 4 0 4 0 4");
    command("create_box 1 box");
    command("create_atoms 1 box");
    command("mass 1 1.0");
    command("pair_style zero 2.5");
    command("pair_coeff * *");
    command("compute ws all ws affine yes");
    END_HIDE_OUTPUT();
    TEST_FAILURE_MPI(".*ERROR: Compute ws does not support shrink-wrapped boundaries.*",
                     command("run 0 post no"););
}

TEST_F(ComputeWSTest, AffineRemapVWarning)
{
    create_crystal(false);
    define_compute("affine yes");
    BEGIN_HIDE_OUTPUT();
    command("fix 1 all deform 1 x scale 1.01 remap v");
    END_HIDE_OUTPUT();
    BEGIN_CAPTURE_OUTPUT();
    command("run 0 post no");
    auto output = END_CAPTURE_OUTPUT();
    if (lmp->comm->me == 0) {
        EXPECT_THAT(output, HasSubstr("remap v"));
    }
}

TEST_F(ComputeWSTest, ReferenceFileMissingColumn)
{
    create_crystal(false);
    BEGIN_HIDE_OUTPUT();
    command("write_dump all custom ws_bad.dump id x y z");
    END_HIDE_OUTPUT();
    define_compute("reference ws_bad.dump");
    TEST_FAILURE_MPI(".*ERROR: Compute ws reference file ws_bad.dump must contain the "
                     "columns id, type, x, y, z.*",
                     command("run 0 post no"););
    remove_file("ws_bad.dump");
}

TEST_F(ComputeWSTest, AtomsChangedBetweenRuns)
{
    create_crystal(false);
    define_compute("");
    run0();
    BEGIN_HIDE_OUTPUT();
    command("create_atoms 1 single 4.25 4.25 4.25 units lattice");
    END_HIDE_OUTPUT();
    TEST_FAILURE_MPI(".*ERROR: Compute ws: atoms were added or removed since the last analysis.*",
                     command("write_dump all custom ws_ref.dump id c_ws[1]"););
    remove_file("ws_ref.dump");

    BEGIN_HIDE_OUTPUT();
    command("run 0 post no");
    END_HIDE_OUTPUT();
    auto *c = lmp->modify->get_compute_by_id("ws");
    EXPECT_EQ(c->vector[0], 1.0);
    EXPECT_EQ(c->vector[1], 0.0);
}

TEST_F(ComputeWSTest, DynamicGroupRejected)
{
    create_crystal(false);
    BEGIN_HIDE_OUTPUT();
    command("group dyn dynamic all region box");
    command("compute ws dyn ws");
    END_HIDE_OUTPUT();
    TEST_FAILURE_MPI(".*dynamic group.*", command("run 0 post no"););
}

TEST_F(ComputeWSTest, Requires3d)
{
    BEGIN_HIDE_OUTPUT();
    command("dimension 2");
    command("lattice sq 0.8");
    command("region box block 0 4 0 4 -0.5 0.5");
    command("create_box 1 box");
    command("create_atoms 1 box");
    command("mass 1 1.0");
    command("pair_style zero 1.0");
    command("pair_coeff * *");
    command("compute ws all ws");
    END_HIDE_OUTPUT();
    TEST_FAILURE_MPI(".*ERROR: Compute ws requires a 3d system.*", command("run 0 post no"););
}

TEST_F(ComputeWSTest, RequiresAtomIDs)
{
    BEGIN_HIDE_OUTPUT();
    command("atom_modify id no");
    command("region box block 0 2 0 2 0 2");
    command("create_box 1 box");
    END_HIDE_OUTPUT();
    TEST_FAILURE_MPI(".*ERROR: Compute ws requires atom IDs.*", command("compute ws all ws"););
}

TEST_F(ComputeWSTest, KeywordErrors)
{
    create_crystal(false);
    TEST_FAILURE_MPI(".*ERROR: Illegal compute ws reference command: missing argument.*",
                     command("compute ws all ws reference"););
    TEST_FAILURE_MPI(".*ERROR: Illegal compute ws affine command: missing argument.*",
                     command("compute ws all ws affine"););
    TEST_FAILURE_MPI(".*ERROR: Unknown compute ws keyword: bogus.*",
                     command("compute ws all ws bogus yes"););
    TEST_FAILURE_MPI(".*ERROR: Expected boolean parameter.*",
                     command("compute ws all ws affine maybe"););
}

// ---------------------------------------------------------------------------
// nearest-site search against a brute-force search
// ---------------------------------------------------------------------------

TEST_F(ComputeWSTest, BruteForceOrthogonal)
{
    create_crystal(false);
    define_compute("");
    run0();
    const auto ref = gather_sites();
    ASSERT_NE(displace_and_run(ref), nullptr);
}

TEST_F(ComputeWSTest, BruteForceTriclinic)
{
    create_crystal(true);
    BEGIN_HIDE_OUTPUT();
    command("change_box all xy final 1.3 xz final -0.9 yz final 0.7 remap units box");
    END_HIDE_OUTPUT();
    define_compute("");
    run0();
    const auto ref = gather_sites();
    ASSERT_NE(displace_and_run(ref), nullptr);
}

// renumbered atom IDs: the own-site test then looks at wrong sites, which
// must not change the assignment
TEST_F(ComputeWSTest, BruteForceRenumberedIDs)
{
    create_crystal(false);
    define_compute("");
    run0();
    const auto ref = gather_sites();
    BEGIN_HIDE_OUTPUT();
    command("region v1 sphere 2.0 2.0 2.0 0.1 units lattice");
    command("delete_atoms region v1 compress no");
    command("reset_atoms id");
    END_HIDE_OUTPUT();
    Compute *c = displace_and_run(ref);
    ASSERT_NE(c, nullptr);
    EXPECT_GT(c->vector[2], 0.0);
}

// one sublattice: the type-2 atoms are both sites and occupants, and the
// type-1 atoms must have all-zero per-atom values
TEST_F(ComputeWSTest, BruteForceGroup)
{
    create_crystal(false);
    BEGIN_HIDE_OUTPUT();
    command("group t2 type 2");
    command("compute ws t2 ws");
    command("thermo_style custom step c_ws[1] c_ws[2] c_ws[3] c_ws[4]");
    END_HIDE_OUTPUT();
    run0();
    const int bit  = lmp->group->get_bitmask_by_id(FLERR, "t2", "test");
    const auto ref = gather_sites(bit);
    ASSERT_EQ((int)ref.tag.size(), NSITES2);
    ASSERT_NE(displace_and_run(ref, bit), nullptr);
}

// two reference sites 0.01 lattice spacings apart: the own-site test almost
// never applies, so nearly every atom is assigned by the cell-list search
TEST_F(ComputeWSTest, BruteForceCloseReferencePair)
{
    create_crystal(false);
    BEGIN_HIDE_OUTPUT();
    command("create_atoms 2 single 4.01 4.0 4.0 units lattice");
    END_HIDE_OUTPUT();
    define_compute("");
    run0();
    const auto ref = gather_sites();
    ASSERT_NE(displace_and_run(ref), nullptr);
}

// reference sites read from a dump file, in an orthogonal box, and with
// scaled coordinates in a triclinic box
TEST_F(ComputeWSTest, BruteForceReferenceFile)
{
    create_crystal(false);
    write_reference();
    define_compute("reference ws_site.dump");
    run0();
    const auto ref = gather_sites();
    ASSERT_NE(displace_and_run(ref), nullptr);
    remove_file("ws_site.dump");
}

TEST_F(ComputeWSTest, BruteForceReferenceFileTriclinicUnscaled)
{
    create_crystal(true);
    BEGIN_HIDE_OUTPUT();
    command("change_box all xy final 1.3 xz final -0.9 yz final 0.7 remap units box");
    END_HIDE_OUTPUT();
    write_reference();
    define_compute("reference ws_site.dump");
    run0();
    const auto ref = gather_sites();
    ASSERT_NE(displace_and_run(ref), nullptr);
    remove_file("ws_site.dump");
}

// crystal slab with vacuum in a non-periodic z dimension, all ranks along z
TEST_F(ComputeWSTest, BruteForceReferenceFileNonPeriodic)
{
    BEGIN_HIDE_OUTPUT();
    command("units lj");
    command("atom_modify map array");
    command("boundary p p f");
    if (lmp->comm->nprocs > 1) command("processors 1 1 *");
    command("lattice fcc 0.8442");
    command("region box block 0 8 0 8 -1 9");
    command("create_box 2 box");
    command("region slab block 0 8 0 8 0 7.9");
    command("create_atoms 1 region slab basis 1 1 basis 2 2 basis 3 2 basis 4 2");
    command("mass * 1.0");
    command("pair_style lj/cut 2.5");
    command("pair_coeff * * 1.0 1.0");
    command("neighbor 0.3 bin");
    END_HIDE_OUTPUT();
    write_reference();
    define_compute("reference ws_site.dump");
    run0();
    const auto ref = gather_sites();
    ASSERT_NE(displace_and_run(ref), nullptr);
    remove_file("ws_site.dump");
}

// recursive coordinate bisection (comm_style tiled), reference from the atoms
// and from a file
TEST_F(ComputeWSTest, BruteForceTiled)
{
    create_crystal(false);
    BEGIN_HIDE_OUTPUT();
    command("comm_style tiled");
    command("balance 0.9 rcb");
    END_HIDE_OUTPUT();
    define_compute("");
    run0();
    const auto ref = gather_sites();
    ASSERT_NE(displace_and_run(ref), nullptr);
}

TEST_F(ComputeWSTest, BruteForceReferenceFileTiled)
{
    create_crystal(false);
    write_reference();
    BEGIN_HIDE_OUTPUT();
    command("comm_style tiled");
    command("balance 0.9 rcb");
    END_HIDE_OUTPUT();
    define_compute("reference ws_site.dump");
    run0();
    const auto ref = gather_sites();
    ASSERT_NE(displace_and_run(ref), nullptr);
    remove_file("ws_site.dump");
}

// box of 2 unit cells with a communication cutoff of almost twice the box
// length: several periodic images of the file sites
TEST_F(ComputeWSTest, BruteForceReferenceFileTinyBox)
{
    BEGIN_HIDE_OUTPUT();
    command("units lj");
    command("atom_modify map array");
    command("lattice fcc 0.8442");
    command("region box block 0 2 0 2 0 2");
    command("create_box 2 box");
    command("create_atoms 1 box basis 1 1 basis 2 2 basis 3 2 basis 4 2");
    command("mass * 1.0");
    command("pair_style lj/cut 2.5");
    command("pair_coeff * * 1.0 1.0");
    command("comm_modify cutoff 6.0");
    END_HIDE_OUTPUT();
    write_reference();
    define_compute("reference ws_site.dump");
    run0();
    const auto ref = gather_sites();
    ASSERT_NE(displace_and_run(ref), nullptr);
    remove_file("ws_site.dump");
}

TEST_F(ComputeWSTest, BruteForceReferenceFileTriclinic)
{
    create_crystal(true);
    BEGIN_HIDE_OUTPUT();
    command("change_box all xy final 1.3 xz final -0.9 yz final 0.7 remap units box");
    END_HIDE_OUTPUT();
    write_reference("xs ys zs");
    define_compute("reference ws_site.dump");
    run0();
    const auto ref = gather_sites();
    ASSERT_NE(displace_and_run(ref), nullptr);
    remove_file("ws_site.dump");
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleMock(&argc, argv);

    // handle arguments passed via environment variable
    if (const char *var = getenv("TEST_ARGS")) {
        auto env = LAMMPS_NS::utils::split_words(var);
        for (const auto &arg : env)
            if (arg == "-v") verbose = true;
    }
    if ((argc > 1) && (strcmp(argv[1], "-v") == 0)) verbose = true;

    int rv = RUN_ALL_TESTS();
    MPI_Finalize();
    return rv;
}
