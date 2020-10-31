# About

This script takes a fairly naive approach to automatically add airflow
gating to a STL mesh to support metal casting via lost-PLA or similar
methods. The algorithm assumes that the input STL file will be printed
as provided, but that the casting will be done upside down (so that
metal flows in the direction of the positive Z-axis). Tube attachment
points are determined by walking up the model and identifying isolated
mounds of volume greater than a provided `min_vol` value. The tubes
are then routed along a simple bezier spline to the outside of the
meshes bounding box and down to the XY-plane).

# Usage

The simplest way to use autogate is via docker:
```
$ docker build . --tag autogate/autogate
$ docker run --rm -v /path/to/stl/directory:/stl autogate/autogate /stl/input.stl /stl/output.stl
```

This will add tubes to the input file found in
`/path/to/stl/directory/input.stl` and write the result to
`/path/to/stl/directory/output.stl`.

# Command Line Options

Autogate includes various options to control the gating algorithm or
to interecept and dump state during the process:

## Tube Generation Parameters

* `--z-step` how finely to slice the contours of the input STL
  file. Larger steps may miss peaks, but smaller steps can be
  extremely slow (default is 1mm).

* `--epsilon` how far apart points can be and be considered
  equal. Default is 1um which seems to work well enoguh.

* `--min-volume` minimum volume for a peak to be considered worth
  adding a tube to default is 1mm^3.

* `--tube-start-radius` the radius of the tube where it joins the peak
  (default is 2mm), if your model has very sharp peaks you may want
  this to be smaller.

* `--tube-end-radius` the radius of the tube where it meets the
  XY-plane. Default is 4mm. This should generally be larger than
  `tube-start-radius`.

## Introspection

* `--dump-contours` saves all of the detected contours of the STL file
  as a (more or less unreadable) JSON file (this is mostly provided as
  a speedup to be used with `--load-contours` on later runs)

* `--dump-peaks` saves just the detected peak locations as a (vaguely
  more readable) JSON file. Again, `--load-peaks` can be used to load
  this file.

* `--dump-tubes` dumps the control points and thicknesses for the
  tubes to a JSON file (that is still pretty unreadable but could
  hypothetically be used to tweak the automatic layout). In a shocking
  display of consistency, `--load-tubes` can be used to load this
  file.

* `--tubes-stl-directory` dumps individual STL files for each tube
  into a file in the given directory.

* `--scad-out` will generate an OpenSCAD file that `union`s the
  generated tube files and the original STL.

* `--preview` show a preview of the XY-plane projection of the
  detected peaks before generating tubes.

# Dependencies

Autogate uses Python 3.x and relies on the `pymesh` library for STL
file loading and manipulation, `matplotlib` for previews, and `bezier`
for generating Bezier curves.

If you don't want to use `docker`, follow the instructions at
https://pymesh.readthedocs.io/en/latest/installation.html to install
pymesh, then `pip install matplotlib bezier`. You should now be able
to run the `autogate-pymesh.py` script directly from your host command
line.
