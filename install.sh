#!/bin/bash
# Install compute ws into a LAMMPS source tree.
#
# Usage:  ./install.sh /path/to/lammps
#
# Copies the source, documentation, unit test and example files to their
# places in the tree and registers them (unit test in CMake, entries in the
# compute lists of the documentation, src/.gitignore). Running it again
# updates the files and adds nothing twice. Then build LAMMPS with the
# EXTRA-COMPUTE package (see README.md).

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
LMP="${1:?usage: $0 /path/to/lammps}"
LMP="$(cd "$LMP" && pwd)"

for d in src/EXTRA-COMPUTE doc/src unittest/commands examples/PACKAGES; do
  [ -d "$LMP/$d" ] || { echo "ERROR: $LMP/$d not found: not a LAMMPS source tree"; exit 1; }
done

if [ -f "$LMP/src/VORONOI/compute_ws.cpp" ]; then
  echo "ERROR: $LMP/src/VORONOI/compute_ws.cpp exists: an older compute ws (based on Voro++)"
  echo "       with the same style name; remove src/VORONOI/compute_ws.h and .cpp first"
  exit 1
fi

copy() {    # copy <file relative to the repository and the LAMMPS tree>
  mkdir -p "$(dirname "$LMP/$1")"
  cp "$HERE/$1" "$LMP/$1"
  echo "copied     $1"
}

copy src/EXTRA-COMPUTE/compute_ws.h
copy src/EXTRA-COMPUTE/compute_ws.cpp
copy doc/src/compute_ws.rst
copy unittest/commands/test_compute_ws.cpp
for f in in.ws ws_sublattice.py log.30Sep26.ws.g++.1 log.30Sep26.ws.g++.4; do
  copy examples/PACKAGES/ws/$f
done
chmod +x "$LMP/examples/PACKAGES/ws/ws_sublattice.py"
if [ -f "$LMP/potentials/SiC.gw.zbl" ]; then
  ln -sf ../../../potentials/SiC.gw.zbl "$LMP/examples/PACKAGES/ws/SiC.gw.zbl"
  echo "linked     examples/PACKAGES/ws/SiC.gw.zbl -> ../../../potentials/SiC.gw.zbl"
fi

cmakefile="$LMP/unittest/commands/CMakeLists.txt"
if ! grep -q "test_compute_ws" "$cmakefile"; then
  cat >> "$cmakefile" <<'EOF'

if(PKG_EXTRA-COMPUTE)
  add_executable(test_compute_ws test_compute_ws.cpp)
  target_link_libraries(test_compute_ws PRIVATE lammps GTest::GMock)
  add_test(NAME ComputeWS COMMAND test_compute_ws)
endif()
EOF
  echo "registered unittest/commands/CMakeLists.txt: test_compute_ws"
fi

gitignore="$LMP/src/.gitignore"
for f in /compute_ws.cpp /compute_ws.h; do
  if ! grep -qx "$f" "$gitignore"; then
    echo "$f" >> "$gitignore"
    echo "added      src/.gitignore: $f"
  fi
done

add_entry() {    # add_entry <doc file> <line after which to insert> <new line>
  local file="$LMP/doc/src/$1"
  if ! grep -qF -- "$3" "$file"; then
    if grep -qF -- "$2" "$file"; then
      awk -v after="$2" -v new="$3" '{print} $0 == after {print new}' "$file" > "$file.tmp"
      mv "$file.tmp" "$file"
      echo "added      doc/src/$1: compute ws"
    else
      echo "NOTE       doc/src/$1: add the entry for compute ws by hand"
    fi
  fi
}
add_entry Commands_compute.rst "   * :doc:\`voronoi/atom <compute_voronoi_atom>\`" \
          "   * :doc:\`ws <compute_ws>\`"
add_entry compute.rst "* :doc:\`voronoi/atom <compute_voronoi_atom>\` - Voronoi volume and neighbors for each atom" \
          "* :doc:\`ws <compute_ws>\` - point defects (interstitials, vacancies, replacements, antisites) by Wigner-Seitz analysis"

cat <<EOF

compute ws is installed in $LMP. Next:
  CMake:       add -D PKG_EXTRA-COMPUTE=on, run cmake again (new files), then build
  legacy make: cd $LMP/src && make yes-extra-compute (again, if the package was
               already installed), then build as usual
EOF
