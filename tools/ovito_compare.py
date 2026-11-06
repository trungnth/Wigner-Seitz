#!/usr/bin/env python3
"""Compare the output of LAMMPS compute ws with the Wigner-Seitz analysis of OVITO.

The script reads a reference configuration and a current configuration that
contains the per-atom array of compute ws (occupancy, site ID, site type),
runs the Wigner-Seitz defect analysis of OVITO on the same two files, and
compares, atom by atom and site by site:

  1. occupancy, site ID and site type of all atoms
     (OVITO with output_displaced=True)
  2. the vacant sites: IDs, types and coordinates
     (OVITO with output_displaced=False; with --affine the reference sites
     of OVITO are mapped into the current box, as the compute does)
  3. the numbers of interstitials, vacancies, replacements and antisites,
     from the per-atom array of the compute, from OVITO, and optionally from
     the last thermo row of a LAMMPS log file and from a defect dump and a
     vacancy dump written by LAMMPS

It also writes the OVITO results in the format of LAMMPS dumps:

  <prefix>_defects.dump    atoms with Occupancy >= 2, id != SiteIdentifier,
                           or type != SiteType, with all columns of the
                           current file (the three compute ws columns from
                           OVITO)
  <prefix>_vacancies.dump  vacant sites: id type x y z
  <prefix>_mismatch.txt    atoms assigned to different sites, if any

Coordinates in LAMMPS dumps are rounded (6 significant digits with the
default format %g). An atom closer than this rounding to the boundary
between two sites can be assigned differently by OVITO, which sees the
rounded coordinates, and by the compute, which uses the exact ones. Each
atom assigned differently is listed with its distances to both sites and
marked as a tie when the two distances differ by less than the rounding
error. Write the dumps with "modify format float %20.15g" and use
--digits 15 for an exact comparison.

Exit code: 0 identical, 2 differences only from ties, 1 other differences.

Example (files of a LAMMPS run, see README.md):

  python3 ovito_compare.py --reference ref.dump --current final.dump.gz \\
      --defects defects.dump --vacancies vacancies.dump --log log.lammps

Requires the OVITO Python module (pip install ovito) and NumPy. For more information: 
https://www.ovito.org/docs/current/python/introduction/getting_started/installation.html
"""

import argparse
import gc
import gzip
import math
import sys
import time

import numpy as np
import ovito
from ovito.io import import_file
from ovito.modifiers import WignerSeitzAnalysisModifier

POSITION = {"x": "Position.X", "y": "Position.Y", "z": "Position.Z",
            "xu": "Position.X", "yu": "Position.Y", "zu": "Position.Z"}


def open_text(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def dump_header(path):
    """Timestep, box lines (4) and column labels of a LAMMPS text dump."""
    with open_text(path) as f:
        lines = [f.readline() for _ in range(9)]
    if not lines[0].startswith("ITEM: TIMESTEP") or not lines[8].startswith("ITEM: "):
        sys.exit(f"{path}: not a LAMMPS text dump")
    return int(lines[1]), lines[4:8], lines[8].split()[2:]


def read_small_dump(path):
    """{label: column} of the last frame of a small LAMMPS text dump (atoms or local)."""
    labels, data, n = None, None, 0
    with open_text(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("ITEM: NUMBER OF"):
                n = int(f.readline())
            elif line.startswith("ITEM: ATOMS") or line.startswith("ITEM: ENTRIES"):
                labels = line.split()[2:]
                rows = [f.readline().split() for _ in range(n)]
                data = np.array(rows, dtype=float).reshape(n, len(labels))
    if labels is None:
        sys.exit(f"{path}: no frame found")
    return labels, {lab: data[:, k] for k, lab in enumerate(labels)}


def read_last_thermo(path, compute_id):
    """(step, (I, V, R, A)) of the last thermo row with c_<ID>[1..4]."""
    keys = [f"c_{compute_id}[{k}]" for k in range(1, 5)]
    header, last = None, None
    with open(path) as f:
        for line in f:
            words = line.split()
            if words[:1] == ["Step"]:
                header = words if all(k in words for k in keys) else None
            elif line.startswith("Loop time"):
                header = None
            elif header and len(words) == len(header):
                try:
                    row = dict(zip(header, (float(w) for w in words)))
                except ValueError:
                    continue
                last = (int(row["Step"]), tuple(int(row[k]) for k in keys))
    return last


def column_map(path, labels, extra):
    """OVITO column mapping: id, type, x y z; other columns as 'LAMMPS <label>' or skipped."""
    for need in ("id", "type"):
        if need not in labels:
            sys.exit(f"{path}: column '{need}' missing")
    if not ({"x", "y", "z"} <= set(labels) or {"xu", "yu", "zu"} <= set(labels)):
        sys.exit(f"{path}: columns x y z (or xu yu zu) missing")
    cols = []
    for lab in labels:
        if lab == "id":
            cols.append("Particle Identifier")
        elif lab == "type":
            cols.append("Particle Type")
        elif lab in POSITION:
            cols.append(POSITION[lab])
        else:
            cols.append("LAMMPS " + lab if extra else None)
    return cols


def half_ulp(values, digits):
    """Largest rounding error of values printed with the given number of significant digits."""
    big = max(float(np.max(np.abs(values))), 1e-300)
    return 0.5 * 10.0 ** (math.floor(math.log10(big)) - digits + 1)


def min_image(dx, h, pbc):
    frac = np.linalg.solve(h, dx.T).T
    for dim in range(3):
        if pbc[dim]:
            frac[:, dim] -= np.round(frac[:, dim])
    return (h @ frac.T).T


def lookup(sorted_ids, ids):
    """Indices of ids in sorted_ids, and whether each id was found."""
    k = np.minimum(np.searchsorted(sorted_ids, ids), len(sorted_ids) - 1)
    return k, sorted_ids[k] == ids


def counts_of(occ, ids, sid, typ, styp, nsites):
    """Interstitials, vacancies, replacements, antisites from the per-atom values of all atoms."""
    multi = occ >= 2
    inter = int(np.sum(multi)) - len(np.unique(sid[multi]))
    vac = nsites - len(np.unique(sid[sid > 0]))
    single = occ == 1
    return (inter, vac, int(np.sum(single & (ids != sid))),
            int(np.sum(single & (typ != styp))))


def as_int(values):
    return np.rint(values).astype(np.int64)


def write_dump(path, step, box, labels, columns, intlabels, local=False):
    n = len(columns[0]) if columns else 0
    kind = "ENTRIES" if local else "ATOMS"
    with open(path, "w") as f:
        f.write(f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF {kind}\n{n}\n")
        f.writelines(box)
        f.write(f"ITEM: {kind} {' '.join(labels)}\n")
        if n:
            fmt = " ".join("%d" if lab in intlabels else "%g" for lab in labels)
            np.savetxt(f, np.column_stack(columns), fmt=fmt)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True,
                   help="dump of the reference configuration (id type x y z)")
    p.add_argument("--current", required=True,
                   help="dump of all atoms with the per-atom array of compute ws")
    p.add_argument("--ws-columns", nargs=3, default=["Occupancy", "SiteIdentifier", "SiteType"],
                   metavar=("OCC", "SITE_ID", "SITE_TYPE"),
                   help="labels of the three compute ws columns in the current file "
                        "(default: Occupancy SiteIdentifier SiteType)")
    p.add_argument("--defects", help="defect dump of compute ws (optional)")
    p.add_argument("--vacancies", help="vacancy dump (local array) of compute ws (optional)")
    p.add_argument("--log", help="LAMMPS log file with thermo columns c_<ID>[1..4] (optional)")
    p.add_argument("--compute-id", default="ws", help="compute ID in the log file (default: ws)")
    p.add_argument("--affine", action="store_true",
                   help="affine mapping ToReference (for a compute with affine yes)")
    p.add_argument("--digits", type=int, default=6,
                   help="significant digits of the coordinates in the dumps (default 6, for %%g)")
    p.add_argument("--prefix", default="ovito", help="prefix of the output files")
    a = p.parse_args()
    occ_lab, sid_lab, styp_lab = a.ws_columns
    ws_labels = tuple(a.ws_columns)
    intlabels = ("id", "type") + ws_labels
    mapping = (WignerSeitzAnalysisModifier.AffineMapping.ToReference if a.affine
               else WignerSeitzAnalysisModifier.AffineMapping.Off)
    t0 = time.time()

    def clock():
        return f"[{time.time() - t0:.0f} s]"

    print(f"OVITO {ovito.version_string}, affine mapping "
          f"{'ToReference' if a.affine else 'Off'}")
    print(f"reference  {a.reference}\ncurrent    {a.current}")

    step, box, cur_labels = dump_header(a.current)
    _, _, ref_labels = dump_header(a.reference)
    missing = [lab for lab in ws_labels if lab not in cur_labels]
    if missing:
        sys.exit(f"{a.current}: compute ws columns {missing} not found (see --ws-columns)")

    # reference sites
    ref_pipe = import_file(a.reference, columns=column_map(a.reference, ref_labels, False))
    rd = ref_pipe.compute()
    ref_ids = np.asarray(rd.particles["Particle Identifier"]).astype(np.int64)
    ref_pos = np.array(rd.particles.positions)
    ref_h, ref_o = np.array(rd.cell[:, :3]), np.array(rd.cell[:, 3])
    pbc = tuple(rd.cell.pbc)
    ref_order = np.argsort(ref_ids, kind="stable")
    ref_sorted = ref_ids[ref_order]
    nref = len(ref_ids)
    del rd
    print(f"reference: {nref} sites, pbc {pbc}  {clock()}")

    # OVITO atoms mode (output_displaced=True)
    pipe = import_file(a.current, columns=column_map(a.current, cur_labels, True))
    mod = WignerSeitzAnalysisModifier(output_displaced=True, affine_mapping=mapping)
    mod.reference = ref_pipe.source
    pipe.modifiers.append(mod)
    d = pipe.compute()
    order = np.argsort(np.asarray(d.particles["Particle Identifier"]), kind="stable")

    def prop(name):
        return np.asarray(d.particles[name])[order]

    ids = prop("Particle Identifier").astype(np.int64)
    typ = prop("Particle Type").astype(np.int64)
    pos = np.array(d.particles.positions)[order]
    occ = prop("Occupancy").astype(np.int64)
    sid = prop("Site Identifier").astype(np.int64)
    styp = prop("Site Type").astype(np.int64)
    sidx = prop("Site Index").astype(np.int64)
    extra = {lab: prop("LAMMPS " + lab) for lab in cur_labels
             if lab not in ("id", "type") and lab not in POSITION}
    ov_inter = int(d.attributes["WignerSeitz.interstitial_count"])
    ov_vac = int(d.attributes["WignerSeitz.vacancy_count"])
    cur_h, cur_o = np.array(d.cell[:, :3]), np.array(d.cell[:, 3])
    del d, pipe, mod
    gc.collect()
    print(f"current:   {len(ids)} atoms at step {step}  {clock()}")
    dcell = max(float(np.max(np.abs(cur_h - ref_h))), float(np.max(np.abs(cur_o - ref_o))))
    if dcell > 1e-6 and not a.affine:
        print(f"WARNING: the box differs from the reference box by {dcell:.3g}; "
              "use --affine if the compute was used with affine yes")

    # OVITO sites mode (output_displaced=False)
    pipe = import_file(a.current, columns=column_map(a.current, cur_labels, False))
    mod = WignerSeitzAnalysisModifier(output_displaced=False, affine_mapping=mapping)
    mod.reference = ref_pipe.source
    pipe.modifiers.append(mod)
    s = pipe.compute()
    vacant = np.asarray(s.particles["Occupancy"]) == 0
    v_ids = np.asarray(s.particles["Particle Identifier"])[vacant].astype(np.int64)
    v_typ = np.asarray(s.particles["Particle Type"])[vacant].astype(np.int64)
    v_pos = np.array(s.particles.positions)[vacant]
    del s, pipe, mod
    gc.collect()
    if a.affine:
        v_pos = (cur_h @ np.linalg.solve(ref_h, (v_pos - ref_o).T)).T + cur_o
    vo = np.argsort(v_ids)
    v_ids, v_typ, v_pos = v_ids[vo], v_typ[vo], v_pos[vo]
    print(f"sites:     {len(v_ids)} vacant  {clock()}\n")

    problems = []
    h_pos = half_ulp(ref_pos, a.digits)
    tie_tol = 4.0 * math.sqrt(3.0) * h_pos
    coord_tol = 2.0 * h_pos

    unused = np.bincount(sidx, minlength=nref) == 0
    if not np.array_equal(np.sort(ref_ids[unused]), v_ids):
        problems.append("OVITO atoms mode and sites mode give different vacant sites")

    # OVITO results written as LAMMPS dumps
    is_def = (occ >= 2) | (ids != sid) | (typ != styp)
    ov_cols = {"id": ids, "type": typ, occ_lab: occ, sid_lab: sid, styp_lab: styp}
    for lab in cur_labels:
        if lab in POSITION:
            ov_cols[lab] = pos[:, "xyz".index(lab[0])]
        elif lab not in ov_cols:
            ov_cols[lab] = extra[lab]
    out_def, out_vac = f"{a.prefix}_defects.dump", f"{a.prefix}_vacancies.dump"
    write_dump(out_def, step, box, cur_labels, [ov_cols[lab][is_def] for lab in cur_labels],
               intlabels)
    write_dump(out_vac, step, box, ["id", "type", "x", "y", "z"],
               [v_ids, v_typ, v_pos[:, 0], v_pos[:, 1], v_pos[:, 2]], intlabels, local=True)
    print(f"written: {out_def} ({int(np.sum(is_def))} atoms), {out_vac} ({len(v_ids)} sites)\n")

    # 1. all atoms
    c_occ, c_sid, c_styp = (as_int(extra[lab]) for lab in ws_labels)
    sid_diff = c_sid != sid
    occ_diff, styp_diff = c_occ != occ, c_styp != styp
    n_any = int(np.sum(occ_diff | sid_diff | styp_diff))
    differ = n_any > 0
    print(f"1. all {len(ids)} atoms, compute ws vs OVITO: site ID differs "
          f"{int(np.sum(sid_diff))}, occupancy differs {int(np.sum(occ_diff))}, site type "
          f"differs {int(np.sum(styp_diff))}")

    # 2. vacancies
    c_vac = np.setdiff1d(ref_sorted, np.unique(c_sid[c_sid > 0]))
    differ |= not np.array_equal(c_vac, v_ids)
    print(f"\n2. vacancies: compute ws {len(c_vac)}, OVITO {len(v_ids)}; only in compute "
          f"{len(np.setdiff1d(c_vac, v_ids))}, only in OVITO {len(np.setdiff1d(v_ids, c_vac))}")
    if a.vacancies:
        _, vcol = read_small_dump(a.vacancies)
        f_vid = vcol["id"].astype(np.int64)
        fo = np.argsort(f_vid)
        f_vid, f_vtyp = f_vid[fo], vcol["type"].astype(np.int64)[fo]
        f_vpos = np.column_stack([vcol[c][fo] for c in ("x", "y", "z")])
        if not np.array_equal(f_vid, c_vac):
            problems.append(f"{a.vacancies} does not list the vacant sites of the per-atom array")
        common = np.intersect1d(f_vid, v_ids)
        kf, ko = np.searchsorted(f_vid, common), np.searchsorted(v_ids, common)
        n_t = int(np.sum(f_vtyp[kf] != v_typ[ko]))
        dx = (np.abs(min_image(f_vpos[kf] - v_pos[ko], cur_h, pbc)).max(axis=1)
              if len(common) else np.zeros(0))
        n_x = int(np.sum(dx > coord_tol))
        print(f"   {a.vacancies}: {len(f_vid)} sites; with OVITO: type differs {n_t}, largest "
              f"coordinate deviation {dx.max() if len(dx) else 0:.3g} "
              f"(rounding {coord_tol:.1g}), larger {n_x}")
        if n_t or n_x:
            problems.append("types or coordinates of vacant sites differ from OVITO")

    # 3. counts
    rows = [("compute ws, per-atom array", counts_of(c_occ, ids, c_sid, typ, c_styp, nref))]
    ov_counts = counts_of(occ, ids, sid, typ, styp, nref)
    if ov_counts[:2] != (ov_inter, ov_vac):
        problems.append("OVITO: counts from the per-atom values differ from its attributes")
    if a.defects:
        _, dcol = read_small_dump(a.defects)
        d_ids = dcol["id"].astype(np.int64)
        c_def_ids = ids[(c_occ >= 2) | (ids != c_sid) | (typ != c_styp)]
        if not np.array_equal(np.sort(d_ids), c_def_ids):
            problems.append(f"{a.defects} does not list the defect atoms of the per-atom array")
        d_occ, d_sid, d_styp = (as_int(dcol[lab]) for lab in ws_labels)
        d_typ = dcol["type"].astype(np.int64)
        multi = d_occ >= 2
        single = d_occ == 1
        d_counts = (int(np.sum(multi)) - len(np.unique(d_sid[multi])),
                    rows[0][1][1] if not a.vacancies else len(f_vid),
                    int(np.sum(single & (d_ids != d_sid))),
                    int(np.sum(single & (d_typ != d_styp))))
        rows.append(("compute ws, defect/vacancy dumps", d_counts))
    if a.log:
        last = read_last_thermo(a.log, a.compute_id)
        if last is None:
            print(f"   no thermo row with c_{a.compute_id}[1..4] in {a.log}")
        elif last[0] != step:
            print(f"   NOTE: last thermo step {last[0]} is not the step {step} of the dump")
        else:
            rows.insert(0, (f"compute ws, thermo step {last[0]}", last[1]))
    print("\n3. counts                                I        V        R        A")
    for name, c in rows:
        print(f"   {name:36s}" + "".join(f"{v:9d}" for v in c))
    print(f"   {'OVITO':36s}" + "".join(f"{v:9d}" for v in ov_counts))
    if len({c for _, c in rows}) != 1:
        problems.append("the outputs of compute ws disagree with each other")
    differ |= rows[0][1] != ov_counts

    # atoms assigned to other sites: distances to both sites
    ties_only = False
    if np.any(sid_diff):
        kd = np.flatnonzero(sid_diff)
        kc, okc = lookup(ref_sorted, c_sid[kd])
        site_c = ref_pos[ref_order[kc]]
        site_o = ref_pos[sidx[kd]]
        dist_c = np.linalg.norm(min_image(pos[kd] - site_c, cur_h, pbc), axis=1)
        dist_o = np.linalg.norm(min_image(pos[kd] - site_o, cur_h, pbc), axis=1)
        delta = dist_c - dist_o
        tie = (np.abs(delta) <= tie_tol) & okc
        out_mis = f"{a.prefix}_mismatch.txt"
        with open(out_mis, "w") as f:
            f.write("# atoms assigned to different sites by compute ws and by OVITO\n"
                    "# distances from the rounded coordinates of the dumps; tie if "
                    f"|d_compute - d_ovito| <= {tie_tol:.2g}\n# id type site_compute "
                    "site_ovito d_compute d_ovito d_compute-d_ovito tie\n")
            for j, i in enumerate(kd):
                f.write(f"{ids[i]} {typ[i]} {c_sid[i]} {sid[i]} {dist_c[j]:.6f} "
                        f"{dist_o[j]:.6f} {delta[j]:.2e} {'yes' if tie[j] else 'NO'}\n")
        print(f"\natoms assigned to different sites: {len(kd)}, of them {int(np.sum(tie))} "
              f"ties within the rounding of the dumps ({tie_tol:.2g}); list in {out_mis}")
        ties_only = bool(np.all(tie))
        if not ties_only:
            problems.append("atoms assigned to different sites that are not ties")
    elif differ:
        problems.append("differences that are not explained by atoms on other sites")

    for msg in problems:
        print(f"PROBLEM: {msg}")
    if not differ and not problems:
        print("\nRESULT: OVITO AND COMPUTE WS GIVE IDENTICAL RESULTS")
        status = 0
    elif ties_only and not problems:
        print("\nRESULT: DIFFERENCES ONLY FROM ATOMS AT TIES WITHIN THE PRECISION OF THE DUMPS")
        status = 2
    else:
        print("\nRESULT: OVITO AND COMPUTE WS DIFFER (see above)")
        status = 1
    print(clock())
    return status


if __name__ == "__main__":
    sys.exit(main())
