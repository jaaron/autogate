import sys
import numpy
import math
import stl
import matplotlib
import matplotlib.pyplot as pyplot
import mpl_toolkits.mplot3d as mplot3d
from matplotlib.widgets import Slider
import itertools
from progress.bar import Bar
import bezier
from math import cos, sin, pi

class Contour():
    def __init__(self, points):
        self.points = points
        self.min_x = points[0][0]
        self.max_x = points[0][0]
        self.min_y = points[0][1]
        self.max_y = points[0][1]
        for p in points:
            if p[0] < self.min_x:
                self.min_x = p[0]
            elif p[0] > self.max_x:
                self.max_x = p[0]
                pass
            if p[1] < self.min_y:
                self.min_y = p[1]
            elif p[1] > self.max_y:
                self.max_y = p[1]
                pass
        self.color = 'green'

    @property
    def z(self):
        return self.points[0][2]
        
    @property
    def center(self):
        return sum(self.points)/len(self.points)

    def contains_point(self, x, epsilon=0.001):
        yline = [p for p in self.points if abs(p[1] - x[1]) < epsilon]
        left  = [p for p in yline if p[0] <= x[0]]
        return (len(left) % 2) == 1

    def move_to_interior(self, x, space=2, epsilon=0.001):
        if self.contains_point(x, epsilon=epsilon):
            return x
        p = min([p for p in self.points], key=lambda p: numpy.linalg.norm(p - x))
        vec = p - x
        return p + space*(vec/numpy.linalg.norm(vec))
    
    def bounding_sq_area(self):
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)
    
    def supports(self, c, max_lip = 1.0):
        if self.min_x < c.min_x:
            left = c.min_x
        else:
            left = self.min_x
        if self.max_x < c.max_x:
            right = self.max_x
        else:
            right = c.max_x
        if self.min_y < c.min_y:
            bottom = c.min_y
        else:
            bottom = self.min_y
        if self.max_y < c.max_y:
            top = self.max_y
        else:
            top = c.max_y
        return left < right and bottom < top and (top - bottom)*(right - left) >= 0.2*self.bounding_sq_area()

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
        for i in range(len(self.points)):
            if self.points[i] != other.points[i]:
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

    def volume_est(self):
        v = 0
        for i in range(1, len(self.contours)):
            c0 = self.contours[i-1]
            c1 = self.contours[i]
            v += (c1.z - c0.z) * (c0.bounding_sq_area() + c1.bounding_sq_area())/2
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
        if self.contours != other.contours:
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
            edge = min([numpy.array([the_mesh.min_[0] - clearance, peak_center[1], peak_center[2]]),
                        numpy.array([the_mesh.max_[0] + clearance, peak_center[1], peak_center[2]]),
                        numpy.array([peak_center[0], the_mesh.min_[1] - clearance, peak_center[2]]),
                        numpy.array([peak_center[0], the_mesh.max_[1] + clearance, peak_center[2]])],
                       key=lambda x: numpy.linalg.norm(peak_center - x))
            self.control_points = [peak_center,
                                       peak_center + numpy.array([0,0,apex]),
                                       edge + numpy.array([0,0,apex]),
                                       edge,
                                       [edge[0], edge[1], the_mesh.min_[2]]]
            pass
        pass

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
                    numpy.array([prev + prev_u*cos( (2*pi*i)/self.fn) + prev_v*sin( (2*pi*i)/self.fn),
                                 prev + prev_u*cos( (2*pi*(i+1))/self.fn) + prev_v*sin( (2*pi*(i+1))/self.fn),
                                 point + u*cos((2*pi*(i+0.5))/self.fn) + v*sin((2*pi*(i+0.5))/self.fn)])
                )
                # downward pointing triangles from point to prev
                triangles.append(
                    numpy.array([prev  + prev_u*cos((2*pi*i)/self.fn) + prev_v*sin((2*pi*i)/self.fn),
                                 point + u*cos( (2*pi*(i+0.5))/self.fn) + v*sin( (2*pi*(i+0.5))/self.fn),
                                 point + u*cos( (2*pi*(i-0.5))/self.fn) + v*sin( (2*pi*(i-0.5))/self.fn)])
                )
                pass
            prev = point
            prev_u = u
            prev_v = v
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
        data = numpy.zeros(len(triangles), dtype=stl.mesh.Mesh.dtype)
        for i in range(len(triangles)):
            data['vectors'][i] = triangles[i]
        return stl.mesh.Mesh(data)
    pass

def interpolate(v1, v2, z):
    return numpy.array([v1[0] + (( (v2[0] - v1[0])/(v2[2] - v1[2]) )*(z - v1[2])),
                        v1[1] + (( (v2[1] - v1[1])/(v2[2] - v1[2]))*(z - v1[2])),
                        z])

def within_threshold_2d(p0, p1, threshold = 0.001):
    return numpy.linalg.norm(p0-p1) <= (threshold*threshold)

def intersect_z_plane(triangles, z, epsilon = 0.001):
    points = []
    for triangle in triangles:
        # we assume the triangles are sorted by their minimum Z value,
        # so once we've crossed the current z threshold, we can stop
        # looking at them.
        if min(triangle[0][2], triangle[1][2], triangle[2][2]) > z:
            break
        ps = []
        if (triangle[0][2] < z < triangle[1][2] or
            triangle[1][2] < z < triangle[0][2]):
            ps.append(interpolate(triangle[0], triangle[1], z))
            pass
        if (triangle[0][2] < z < triangle[2][2] or
            triangle[2][2] < z < triangle[0][2]):
            ps.append(interpolate(triangle[0], triangle[2], z))
            pass

        if (triangle[1][2] < z < triangle[2][2] or
            triangle[2][2] < z < triangle[1][2]):
            ps.append(interpolate(triangle[1], triangle[2], z))
            pass

        if triangle[0][2] == z:
            ps.append(triangle[0])
            pass
        if triangle[1][2] == z:
            ps.append(triangle[1])
            pass
        if triangle[2][2] == z:
            ps.append(triangle[2])
            pass
        if len(ps) == 2:
            points.append(ps)
        pass
    return points

def contours_of_segments(segments, epsilon = 0.001):
    contours = []
    segmnts = list(segments) # duplicate the list
    while(len(segments) > 0):
        contour = segments.pop()
        progress = True
        while progress:
            progress = False
            # scan through the remaining segments looking for one that
            # shares and endpoint with our contour
            segments  = sorted(segments, key=lambda x: numpy.linalg.norm(contour[-1]-x))
            surviving = []
            for i in range(len(segments)):
                s = segments[i]
                if within_threshold_2d(s[0], contour[-1], epsilon):
                    progress = True
                    contour.append(s[1])
                elif within_threshold_2d(s[1], contour[-1], epsilon):
                    progress = True
                    contour.append(s[0])
                else:
                    surviving.append(s)
                pass
            segments = surviving
            pass
        # gobbled up all adjacent points, if our contour has at least
        # three vertices, add it to the list.
        if len(contour) > 2:
            if not within_threshold_2d(contour[0], contour[-1], epsilon):
                print("WARNING: open path", file=sys.stderr)
                return None
                pass
            contours.append(Contour(contour))
            pass        
        pass
    return contours

def slice(the_mesh, step=1.0, epsilon=0.001):
    slices    = []
    z         = the_mesh.min_[2]
    triangles = sorted(the_mesh.vectors, key=lambda x: min(x[0][2], x[1][2], x[2][2]))
    
    with Bar('Slicing', max=(the_mesh.max_[2] - the_mesh.min_[2])/step) as bar:
        while z < the_mesh.max_[2]:
            # triangles = itertools.dropwhile(lambda x: max(x[0][2], x[1][2], x[2][2]) < z, triangles)
            segments  = intersect_z_plane(triangles, z, epsilon=epsilon)
            cs        = contours_of_segments(segments, epsilon=epsilon)
            if cs:
                slices.append(cs)
            z     += step
            bar.next()
    return slices
                
def find_peaks(contours, min_vol=1.0):
    mounds   = [Mound(c) for c in contours[0]]
    peaks    = []
    for z_cs in contours[1:]:
        new_mounds = []
        for m in mounds:
            cs = [c for c in z_cs if m.supports(c)]
            if len(cs) == 0:
                peaks.append(m)
            elif len(cs) == 1:
                new_mounds.append(m.grow(cs[0]))
            else:
                # fixme: proportionally weight inheritance of existing mound's mass
                vol         = m.volume_est()/len(cs)
                new_mounds += [Mound(c, inherited_volume=vol) for c in cs]
                pass
            pass
        for c in z_cs:
            if len([m for m in mounds if m.supports(c)]) == 0:
                new_mounds.append(Mound(c))
                pass
            pass
        mounds = new_mounds
    peaks += mounds
    return [p.top_contour for p in peaks if p.volume_est() > min_vol]

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
    max_x  = cs[0].max_x
    min_x  = cs[0].min_x
    max_y  = cs[0].max_y
    min_y  = cs[0].min_y
    for c in cs:
        c.set_color(z_color(c.z, m))
        c.preview(axes)
        if c.max_x > max_x:
            max_x = c.max_x
        if c.min_x < min_x:
            min_x = c.min_x
        if c.max_y > max_y:
            max_y = c.max_y
        if c.min_y < min_y:
            min_y = c.min_y
    axes.set_xlim(min_x, max_x)
    axes.set_ylim(min_y, max_y)
    pyplot.show()
    return axes

def load_mesh(f):
    return stl.mesh.Mesh.from_file(f)

def z_color(z, the_mesh):
    return [(z - the_mesh.min_[2])/(the_mesh.max_[2] - the_mesh.min_[2]),
            1.0 - (z - the_mesh.min_[2])/(the_mesh.max_[2] - the_mesh.min_[2]),
            0]

import argparse
def parse_arguments():
    pa = argparse.ArgumentParser()
    pa.add_argument('--z-step', default=1.0, type=float, help='Z step between slices')
    pa.add_argument('--epsilon', default=0.001, type=float, help='Max distance between two points considered equal')
    pa.add_argument('--min-volume', default=1.0, type=float, help='Minium (estimated) volume of a differentiated peak')
    pa.add_argument('--tube-start-radius', default=2.0, type=float, help='Initial tube radius')
    pa.add_argument('--tube-end-radius', default=4.0, type=float, help='Final tube radius')
    pa.add_argument('--preview', default=False, const=True, action='store_const', help='Preview Peak Locations')
    pa.add_argument('stl', metavar='stl')
    pa.add_argument('out', metavar='outfile')
    return pa.parse_args()

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        args = parse_arguments()
        m = load_mesh(args.stl)
        contours = slice(m, step=args.z_step, epsilon=args.epsilon)
        ps = find_peaks(contours, min_vol=args.min_volume)
        if args.preview:
            preview_contours(ps)
        c  = stl.mesh.Mesh(numpy.concatenate([m.data] + [Tube(m, p,
                                                              radius_start = args.tube_start_radius,
                                                              radius_end   = args.tube_end_radius,
                                                              epsilon      = args.epsilon).mesh().data
                                                             for p in ps]))
        c.save(args.out, mode=stl.Mode.ASCII)
        pass
    pass
