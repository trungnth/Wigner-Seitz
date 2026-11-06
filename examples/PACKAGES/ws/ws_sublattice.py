#!/usr/bin/env python3
"""Per-type and sublattice defect statistics from the output of compute ws.

Reads the files written by in.ws, one snapshot per file:

  vacancies.<step>.dump  dump local:  id, type, x, y, z of each vacant site
  defects.<step>.dump    dump custom: id, type, x, y, z, Occupancy,
                         SiteIdentifier, SiteType (only atoms with
                         Occupancy >= 2, id != SiteIdentifier, or
                         type != SiteType)

Every site not listed in either file is occupied by exactly its original
atom, so the two files describe the occupancy of every site. For every
timestep present in both, this script prints

  * vacancies per sublattice
  * antisites and replacements on singly occupied sites
  * interstitial configurations by site type and chemical composition
  * a consistency check: each multiply occupied site must appear as often
    as its occupancy

Only the Python standard library is used.

Usage:  python3 ws_sublattice.py [directory]
"""

import glob
import os
import re
import sys
from collections import Counter, defaultdict


def read_dump(path):
    """Return (column names, rows) of a single-snapshot LAMMPS text dump."""
    with open(path) as f:
        lines = f.read().splitlines()
    count = None
    for k, line in enumerate(lines):
        if line.startswith("ITEM: NUMBER OF"):
            count = int(lines[k + 1])
        elif line.startswith(("ITEM: ATOMS", "ITEM: ENTRIES")):
            columns = line.split()[2:]
            rows = [[float(v) for v in row.split()] for row in lines[k + 1 : k + 1 + count]]
            return columns, rows
    raise ValueError(f"{path}: no ATOMS or ENTRIES section")


def by_step(directory, prefix):
    """Map timestep -> file name for files named <prefix>.<step>.dump."""
    files = {}
    for path in glob.glob(os.path.join(directory, prefix + ".*.dump")):
        match = re.search(r"\.(\d+)\.dump$", path)
        if match:
            files[int(match.group(1))] = path
    return files


def analyze(vac_file, def_file):
    vcols, vrows = read_dump(vac_file)
    dcols, drows = read_dump(def_file)
    vac_site_type = vcols.index("type")
    aid, atype, occ, site_id, site_type = (
        dcols.index(c) for c in ("id", "type", "Occupancy", "SiteIdentifier", "SiteType")
    )

    vacancies = Counter(int(r[vac_site_type]) for r in vrows)

    antisites = Counter()       # (atom type, site type) on singly occupied sites
    replacements = 0            # singly occupied by another atom
    unassigned = 0              # occupancy 0: not assigned to any site
    configs = defaultdict(list) # site ID -> atoms of a multiply occupied site
    for r in drows:
        n = int(r[occ])
        if n == 0:
            unassigned += 1
        elif n == 1:
            if r[atype] != r[site_type]:
                antisites[(int(r[atype]), int(r[site_type]))] += 1
            if r[aid] != r[site_id]:
                replacements += 1
        else:
            configs[int(r[site_id])].append(r)

    compositions = Counter()
    incomplete = []
    for sid, members in configs.items():
        n = int(members[0][occ])
        if len(members) != n:
            incomplete.append((sid, n, len(members)))
        types = Counter(int(m[atype]) for m in members)
        compositions[(int(members[0][site_type]), tuple(sorted(types.items())))] += 1
    interstitials = sum(int(m[0][occ]) - 1 for m in configs.values())

    return vacancies, antisites, replacements, compositions, interstitials, incomplete, unassigned


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    vac, defects = by_step(directory, "vacancies"), by_step(directory, "defects")
    steps = sorted(set(vac) & set(defects))
    if not steps:
        sys.exit(f"no matching vacancies.<step>.dump and defects.<step>.dump files in {directory}")

    for step in steps:
        vacancies, antisites, nrep, compositions, nint, incomplete, nunassigned = analyze(
            vac[step], defects[step]
        )
        print(f"step {step}")
        print("  vacancies per sublattice: "
              + (", ".join(f"type-{j} sites {n}" for j, n in sorted(vacancies.items())) or "none"))
        print("  antisites (atom type on site type): "
              + (", ".join(f"{i} on {j}: {n}" for (i, j), n in sorted(antisites.items())) or "none"))
        print(f"  replacements: {nrep}")
        print(f"  interstitials: {nint}, configurations by site type and (type, count) composition:")
        for (j, comp), n in sorted(compositions.items()):
            print(f"    {n} x on type-{j} site: {comp}")
        if not compositions:
            print("    none")
        for sid, n, found in incomplete:
            print(f"  WARNING: site {sid} has occupancy {n} but {found} atoms in the dump")
        if nunassigned:
            print(f"  WARNING: {nunassigned} atoms could not be assigned to a site")


if __name__ == "__main__":
    main()
