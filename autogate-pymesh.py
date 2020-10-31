#!/usr/bin/env python
import sys
import numpy
import math
import pymesh
import matplotlib
import matplotlib.pyplot as pyplot
import mpl_toolkits.mplot3d as mplot3d
import itertools
import bezier
from math import cos, sin, pi
import json
import os
import subprocess
import tempfile
import shutil

class Contour():
    def __init__(self, component):
        self.component_mesh = component
        self.component_mesh.add_attribute("face_area")
        self.points = [component.vertices[i] for i in component.boundary_loops[0]]
        self.min_x = self.component_mesh.bbox[0][0]
        self.min_y = self.component_mesh.bbox[0][1]
        self.max_x = self.component_mesh.bbox[1][0]
        self.max_y = self.component_mesh.bbox[1][1]
        self.color = 'green'
        pass
    
    def to_json(self):
        return {'color': self.color, 'points': [[float(p[0]), float(p[1]), float(p[2])] for p in self.points]}

    @classmethod
    def from_json(self, j):
        c = Contour([numpy.array(p, dtype=numpy.float32) for p in j['points']])
        c.color = j['color']
        return c
    
    @property
    def z(self):
        return self.points[0][2]
        
    @property
    def center(self):
        return [self.min_x + (self.max_x - self.min_x)/2,
                self.min_y + (self.max_y - self.min_y)/2,
                self.z]
    
    def contains_point(self, x, epsilon=0.001):
        (dists, faces, pts) = pymesh.distance_to_mesh(self.component_mesh, [x])
        if dists == 0.0:
            return True
        return False

    def move_to_interior(self, x, space=2, epsilon=0.001):
        if self.contains_point(x, epsilon=epsilon):
            return x
        p    = numpy.array(min([p for p in self.points], key=lambda p: numpy.linalg.norm(p - x)))
        p[2] = x[2]
        vec  = p - x
        p    = p + space*(vec/numpy.linalg.norm(vec))
        p[2] = self.z
        return p
    
    def bounding_sq_area(self):
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

    def area(self):
        return sum(self.component_mesh.get_attribute("face_area"))
    
    def supports(self, c):
        other             = pymesh.form_mesh(c.component_mesh.vertices - numpy.array([0, 0, (c.z - self.z)]),
                                             c.component_mesh.faces)
        pymesh.remove_duplicated_vertices(other)
        pymesh.remove_duplicated_faces(other)
        intersection      = pymesh.boolean(self.component_mesh, other, 'intersection')
        pymesh.remove_duplicated_vertices(intersection)
        pymesh.remove_duplicated_faces(intersection)
        intersection.add_attribute("face_area")
        intersection_area = sum(intersection.get_attribute("face_area"))
        return intersection_area >= 0.2*self.area()

    def set_color(self, c):
        self.color = c
        return self
        
    def preview(self, axes=None):
        show = False
        if not axes:
            figure = pyplot.figure()
            axes = figure.subplots()
            axes.set_xlim(self.min_x - 5, self.max_x + 5)
            axes.set_ylim(self.min_y - 5, self.max_y + 5)
            show = True
        path = matplotlib.path.Path([(p[0], p[1]) for p in self.points],
                                    ([matplotlib.path.Path.MOVETO]+
                                     ([matplotlib.path.Path.LINETO]*(len(self.points)-2))+
                                     [matplotlib.path.Path.CLOSEPOLY]))
        patch = matplotlib.patches.PathPatch(path, facecolor=self.color, lw=2)
        axes.add_patch(patch)
        if show:
            pyplot.show()
        return axes

    def __eq__(self, other):
        if len(self.points) != len(other.points):
            return False
        self_ps  = sorted(self.points, key=lambda p: numpy.linalg.norm(p))
        other_ps = sorted(other.points, key=lambda p: numpy.linalg.norm(p))
        for i in range(len(self.points)):
            if not within_threshold_2d(self_ps[i], other_ps[i], 0.001):
                return False
            pass
        return True
    
    def __hash__(self):
        return int(self.min_x * self.max_x * self.min_y * self.max_y)
    
    def __str__(self):
        return "Contour(%s)" % self.points


class Mound:
    def __init__(self, contour, inherited_volume=0.0):
        if not isinstance(contour, Contour):
            raise "Mounds must be created from contours"
        self.contours = [contour]
        self.inherited_volume = inherited_volume
        pass

    def to_json(self):
        return {'contours': [c.to_json() for c in self.contours], 'inherited_volume': self.inherited_volume}

    @classmethod
    def from_json(self, j):
        cs = [Contour.from_json(cj) for cj in j['contours']]
        m  = Mound(cs[0], inherited_volume=j['inherited_volume'])
        for c in cs[1:]:
            m.grow(c)
        return m
    
    @property
    def base_z(self):
        return self.contours[0].z

    @property
    def base_contour(self):
        return self.contours[0]

    @property
    def top_z(self):
        return self.contours[-1].z
    
    @property
    def top_contour(self):
        return self.contours[-1]

    @property
    def height(self):
        return self.top_z - self.base_z

    @property
    def center(self):
        return numpy.array(self.top_contour.center)

    def move_to_interior(self, x, space=2, epsilon=0.001):
        if x[2] >= self.top_contour.z - epsilon:
            return self.top_contour.move_to_interior(x, space=space, epsilon=epsilon)
                
        for i in range(len(self.contours)-1, 0, -1):
            if abs(self.contours[i-1].z - x[2]) < epsilon or (self.contours[i-1].z < x[2] and self.contours[i].z > x[2]):
                return self.contours[i-1].move_to_interior(x, space=space, epsilon=epsilon)
            pass
        
        return self.base_contour.move_to_interior(x, space=space, epsilon=epsilon)
    
    def volume_est(self):
        v = 0
        for i in range(1, len(self.contours)):
            c0 = self.contours[i-1]
            c1 = self.contours[i]
            v += (c1.z - c0.z) * (c0.area() + c1.area())/2
            pass
        return v+self.inherited_volume
            
    def grow(self, contour):
        if not isinstance(contour, Contour):
            raise "Mounds must be created from contours"
        self.contours.append(contour)
        return self

    def supports(self, contour):
        return self.top_contour.supports(contour)

    def preview(self, axes=None):
        return self.top_contour.preview(axes)

    def __hash__(self):
        return hash(self.top_contour)

    def __eq__(self, other):
        if len(self.contours) != len(other.contours):
            return False
        for i in range(len(self.contours)):
            if self.contours[i] != other.contours[i]:
                return False
        return True
    
    def __str__(self):
        return "Mound(%s)%s" % (self.base_contour,
                                    ''.join([".grow(%s)" % c for c in self.contours[1:]]))

class Tube:
    def __init__(self, the_mesh, peak, clearance=15, apex=10, radius_start=3, radius_end=8, steps=50, fn=64, control_points=None, epsilon=0.001):
        self.radius_start = radius_start
        self.radius_end   = radius_end
        self.steps        = steps
        self.fn           = fn
        if control_points:
            self.control_points = control_points
        else:
            peak_center = peak.move_to_interior(peak.center, space=radius_start*2, epsilon=epsilon)
            edge = min([numpy.array([the_mesh.bbox[0][0] - clearance, peak_center[1], peak_center[2]]),
                        numpy.array([the_mesh.bbox[1][0] + clearance, peak_center[1], peak_center[2]]),
                        numpy.array([peak_center[0], the_mesh.bbox[0][1] - clearance, peak_center[2]]),
                        numpy.array([peak_center[0], the_mesh.bbox[1][1] + clearance, peak_center[2]])],
                       key=lambda x: numpy.linalg.norm(peak_center - x))
            self.control_points = [peak_center,
                                       peak_center + numpy.array([0,0,apex]),
                                       edge + numpy.array([0,0,apex]),
                                       edge,
                                       [edge[0], edge[1], the_mesh.bbox[0][2]]]
            pass
        pass

    def tighten(self, delta=1.0):
        start = self.control_points[0]
        cs    = self.control_points[1:-2]
        edge  = self.control_points[-2]
        foot  = self.control_points[-1]

        v     = start - edge
        v    /= numpy.linalg.norm(v)
        edge += delta*v
        foor += delta*v
        cs    = [c + (delta/len(cs)) * v for c in cs]
        self.control_points = [start] + cs + [edge, foot]
        pass

    def widen(self, delta=1.0):
        start = self.control_points[0]
        cs    = self.control_points[1:-2]
        edge  = self.control_points[-2]
        foot  = self.control_points[-1]

        v     = edge - start
        v    /= numpy.linalg.norm(v)
        edge += delta*v
        foot += delta*v
        cs    = [c + (delta/len(cs)) * v for c in cs]
        self.control_points = [start] + cs + [edge, foot]
        pass

    def translate(self, delta):
        self.control_points = [p + delta for p in self.control_points]
        pass

    def rotate(self, theta):
        theta = theta * 2 * pi/360.0
        m     = numpy.matrix([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0,0,1]])
        for i in range(1,len(self.control_points)):
            p    = self.control_points[i]
            p    = p - self.control_points[0]
            p    = (p * m).A1
            p    = p + self.control_points[0]            
            self.control_points[i] = p
            
    def to_json(self):
        return {'radius_start':   self.radius_start,
                'radius_end':     self.radius_end,
                'steps':          self.steps,
                'fn':             self.fn,
                'control_points': [[float(p[0]), float(p[1]), float(p[2])] for p in self.control_points]}

    @classmethod
    def from_json(self, j):
        t = Tube(None, None,
                 radius_start   = j['radius_start'],
                 radius_end     = j['radius_end'],
                 steps          = j['steps'],
                 fn             = j['fn'],
                 control_points = [numpy.array(p, dtype=numpy.double) for p in j['control_points']])
        if 'transform' in j:
            for tfrm in j['transform']:
                if tfrm['action'] == 'tighten':
                    t.tighten(tfrm['delta'])
                elif tfrm['action'] == 'widen':
                    t.widen(tfrm['delta'])
                elif tfrm['action'] == 'rotate':
                    t.rotate(tfrm['theta'])
                elif tfrm['action'] == 'translate':
                    t.translate(numpy.array(tfrm['detla'], dtype=double))
                    pass
                pass
            pass
        return t
    
    def path(self):
        n = numpy.asfortranarray(self.control_points)
        n = numpy.transpose(n)
        c = bezier.Curve.from_nodes(n)
        s_vals = numpy.linspace(0.0, 1.0, self.steps)
        return numpy.transpose(c.evaluate_multi(s_vals))

    def mesh(self):
        path      = self.path()
        triangles = []
        prev      = path[0]
        prev_u    = None
        prev_v    = None
        first     = True
        r_step    = (self.radius_end - self.radius_start)/len(path)
        r         = self.radius_start
        shift     = 0.0
        
        for point in path[1:]:
            prev_r  = r
            r      += r_step
            # construct an orthonormal basis(u,v) for the plane normal to
            # the vector point - prev
            norm    = point - prev
            absnorm = numpy.abs(norm)
            mindim  = numpy.where(absnorm == numpy.min(absnorm))
            if mindim == 0:
                u   = numpy.array([0, -norm[2], norm[1]])
                v   = numpy.array([norm[1]*norm[1] + norm[2]*norm[2], - norm[0]*norm[1], - norm[0]*norm[2]])
            elif mindim == 1:
                u   = numpy.array([-norm[2], 0 ,norm[1]])
                v   = numpy.array([-norm[1]*norm[0], norm[0]*norm[0] + norm[2]*norm[2], -norm[1]*norm[2]])
            else:
                u   = numpy.array([-norm[1], norm[0], 0]) 
                v   = numpy.array([-norm[2]*norm[0], -norm[2]*norm[1], norm[0]*norm[0] + norm[1]*norm[1]])
                pass
            u       = u * (r/numpy.linalg.norm(u))
            v       = v * (r/numpy.linalg.norm(v))
            # make the base of the tube, a disc in the plane defined by
            # u,v at point prev
            if first:
                first  = False
                prev_u = u
                prev_v = v
                for i in range(self.fn):
                    triangles.append(
                        numpy.array([prev + prev_u*cos( (2*pi*i)/self.fn) + prev_v*sin( (2*pi*i)/self.fn),
                                     prev,
                                     prev + prev_u*cos( (2*pi*(i+1))/self.fn) + prev_v*sin( (2*pi*(i+1))/self.fn)])
                    )
                    pass
                pass
            
            for i in range(self.fn):
                # upward pointing triangle from prev to point
                triangles.append(
                    numpy.array([prev + prev_u*cos( (2*pi*(i + shift))/self.fn) + prev_v*sin( (2*pi*(i + shift))/self.fn),
                                 prev + prev_u*cos( (2*pi*(i + shift + 1))/self.fn) + prev_v*sin( (2*pi*(i+ shift + 1))/self.fn),
                                 point + u*cos((2*pi*(i + shift + 0.5))/self.fn) + v*sin((2*pi*(i + shift + 0.5))/self.fn)])
                )
                # downward pointing triangles from point to prev
                triangles.append(
                    numpy.array([prev  + prev_u*cos((2*pi*(i + shift))/self.fn) + prev_v*sin((2*pi*(i + shift))/self.fn),
                                 point + u*cos( (2*pi*(i + shift + 0.5))/self.fn) + v*sin( (2*pi*(i + shift + 0.5))/self.fn),
                                 point + u*cos( (2*pi*(i + shift - 0.5))/self.fn) + v*sin( (2*pi*(i + shift - 0.5))/self.fn)])
                )
                pass
            prev = point
            prev_u = u
            prev_v = v
            shift  = (shift + 0.5) % 1
            pass
        # Make the top of the tube, a disc in the plane defined by
        # prev_u,prev_v at point prev
        for i in range(self.fn):
            triangles.append(
                numpy.array([prev + prev_u*cos( (2*pi*(i-0.5))/self.fn) + prev_v*sin( (2*pi*(i-0.5))/self.fn),
                            prev + prev_u*cos( (2*pi*(i+0.5))/self.fn) + prev_v*sin( (2*pi*(i+0.5))/self.fn),
                            prev])
            )
            pass
        vertices = numpy.zeros((len(triangles)*3, 3))
        faces    = numpy.zeros((len(triangles), 3))
        for i in range(len(triangles)):
            vertices[3*i] = triangles[i][0]
            vertices[3*i+1] = triangles[i][1]
            vertices[3*i+2] = triangles[i][2]
            faces[i] = numpy.array([3*i, 3*i+1, 3*i+2])
        m = pymesh.form_mesh(vertices, faces)
        return pymesh.remove_duplicated_vertices(m)[0]
    pass

def within_threshold_2d(p0, p1, threshold = 0.001):
    return numpy.linalg.norm(p0-p1) <= (threshold*threshold)

def slice(the_mesh, step=1.0, epsilon=0.001):
    n = math.ceil((the_mesh.bbox[1][2] - the_mesh.bbox[0][2])/step)
    print("slicing into %d slices" % (n))
    return [[Contour(component)
                 for component in pymesh.separate_mesh(slice)]
            for slice in pymesh.slice_mesh(the_mesh, numpy.array([0,0,1]), n)]
                
def find_peaks(contours, min_vol=1.0):
    mounds   = set([Mound(c) for c in contours[0]])
    peaks    = []
    for z_cs in contours[1:]:
        new_mounds = set()
        for m in mounds:
            cs = [c for c in z_cs if m.supports(c)]
            if len(cs) == 0:
                peaks.append(m)
            elif len(cs) == 1:
                new_mounds.add(m.grow(cs[0]))
            else:
                # fixme: proportionally weight inheritance of existing mound's mass
                vol         = m.volume_est()/len(cs)
                new_mounds  = new_mounds.union(set([Mound(c, inherited_volume=vol) for c in cs]))
                pass
            pass
        for c in z_cs:
            if len([m for m in mounds if m.supports(c)]) == 0:
                new_mounds.add(Mound(c))
                pass
            pass
        mounds = new_mounds
    peaks += mounds
    return set([p for p in peaks if p.volume_est() > min_vol])

def preview_mesh(m):
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))
    scale = m.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    pyplot.show()
    return axes

def preview_contours(cs):
    figure = pyplot.figure()
    axes   = figure.subplots()
    max_x  = None
    min_x  = None
    max_y  = None
    min_y  = None
    for c in cs:
        c.set_color(z_color(c.z, m))
        c.preview(axes)
        if (not max_x) or (c.max_x > max_x):
            max_x = c.max_x
        if (not min_x) or (c.min_x < min_x):
            min_x = c.min_x
        if (not min_y) or (c.max_y > max_y):
            max_y = c.max_y
        if (not c.min_y) (c.min_y < min_y):
            min_y = c.min_y
    axes.set_xlim(min_x, max_x)
    axes.set_ylim(min_y, max_y)
    pyplot.show()
    return axes

def load_mesh(f):
    return pymesh.load_mesh(f)

def z_color(z, the_mesh):
    return [(z - the_mesh.bbox[0][2])/(the_mesh.bbox[1][2] - the_mesh.bbox[0][2]),
            1.0 - (z - the_mesh.bbox[0][2])/(the_mesh.bbox[1][2] - the_mesh.bbox[0][2]),
            0]

import argparse
def parse_arguments():
    pa = argparse.ArgumentParser()
    pa.add_argument('--z-step', default=1.0, type=float, help='Z step between slices')
    pa.add_argument('--epsilon', default=0.001, type=float, help='Max distance between two points considered equal')
    pa.add_argument('--min-volume', default=1.0, type=float, help='Minium (estimated) volume of a differentiated peak')
    pa.add_argument('--tube-start-radius', default=2.0, type=float, help='Initial tube radius')
    pa.add_argument('--tube-end-radius', default=4.0, type=float, help='Final tube radius')
    pa.add_argument('--tube-clearance', default=5.0, type=float, help='Clearance distance from mesh edge for tube placement')
    
    pa.add_argument('--preview', default=False, const=True, action='store_const', help='Preview Peak Locations')
    pa.add_argument('--dump-contours', default=None, type=str, help='Dump contours as JSON to file')
    pa.add_argument('--load-contours', default=None, type=str, help='Load contours from JSON file instead of slicing')
    pa.add_argument('--skip-slicing', default=False, action='store_const', const=True,
                    help='Skip slicing step. Requires --load-contours, --load-peaks, or --load-tubes')
    pa.add_argument('--dump-peaks', default=None, type=str, help='Dump peaks as JSON to file')
    pa.add_argument('--load-peaks', default=None, type=str, help='Load peaks from JSON file instead of inferring from contours')
    pa.add_argument('--skip-peaks', default=False, action='store_const', const=True,
                    help='Skip peak inference step. Requires --load-peaks or --load-tubes')
    pa.add_argument('--dump-tubes', default=None, type=str, help='Dump tubes as JSON to file')
    pa.add_argument('--load-tubes', default=None, type=str, help='Load tubes from JSON file instead of autoplacing')
    pa.add_argument('--tubes-stl-directory', default=None, type=str, help='Output tube STL meshes to files in directory')
    pa.add_argument('--scad-out', default=None, type=str, help='Output OpenSCAD file (requires --tubes-stl-directory)')
    pa.add_argument('stl', metavar='stl')
    pa.add_argument('out', metavar='outfile')
    return pa.parse_args()

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        args = parse_arguments()
        m = load_mesh(args.stl)
        if args.load_contours:
            print("Loading contours from %s" % (args.load_contours), file=sys.stderr)
            with open(args.load_contours) as f:
                contours = [set([Contour.from_json(cj) for cj in l]) for l in json.load(f)]
                pass
        else:
            if not args.skip_slicing:
                contours = slice(m, step=args.z_step, epsilon=args.epsilon)
                pass
            pass

        if args.dump_contours:
            with open(args.dump_contours, 'w') as f:
                json.dump([[c.to_json() for c in z_cs] for z_cs in contours], f)
                pass
            pass

        if args.load_peaks:
            print("Loading peaks from %s" % (args.load_peaks), file=sys.stderr)
            with open(args.load_peaks) as f:
                peaks = set([Mound.from_json(cm) for cm in json.load(f)])
                pass
        else:
            if not args.skip_peaks:
                peaks = find_peaks(contours, min_vol=args.min_volume)
                pass
            pass
        
        if args.dump_peaks:
            with open(args.dump_peaks, 'w') as f:
                json.dump([p.to_json() for p in peaks], f)
                pass
            pass
        
        if args.preview:
            preview_contours(peaks)
            pass

        if args.load_tubes:
            print("Loading tubes from %s" % (args.load_tubes), file=sys.stderr)
            with open(args.load_tubes) as f:
                tubes = [Tube.from_json(tj) for tj in json.load(f)]
                pass
        else:
            tubes = [Tube(m, p,
                              clearance    = args.tube_clearance,
                              radius_start = args.tube_start_radius,
                          radius_end   = args.tube_end_radius,
                          epsilon      = args.epsilon)
                     for p in peaks]
            pass
        
        if args.dump_tubes:
            with open(args.dump_tubes, 'w') as f:
                json.dump([t.to_json() for t in tubes], f)
                pass
            pass

        if args.tubes_stl_directory:
            try:
                os.mkdir(args.tubes_stl_directory)
            except:
                pass
            i = 0
            for t in tubes:
                pymesh.save_mesh(os.path.join(args.tubes_stl_directory, "tube-%d.stl" % (i)), t.mesh(), mode=stl.Mode.ASCII)
                i += 1
                pass
            pass

        if args.scad_out:
            if not args.tubes_stl_directory:
                raise Exception('OpenSCAD output requires option --tubes-stl-directory')
            with open(args.scad_out) as f:
                f.write("union(){\n")
                f.write("\timport(\"%s\");\n" % (args.stl))
                for i in range(len(tubes)):
                    f.write("\timport(\"%s/tube-%d.stl\")\n" % (args.tubes_stl_directory, i))
                f.write("}\n")
                pass
            pass
        merged = pymesh.CSGTree({"union": [{"mesh": m}, *[{"mesh": t.mesh()} for t in tubes]]})
        pymesh.save_mesh(args.out, merged.mesh, ascii=True)
        pass
    pass
