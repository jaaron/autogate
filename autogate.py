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
    def __init__(self, z, contour, inherited_volume=0.0):
        self.contours = [(z,contour)]
        self.inherited_volume = inherited_volume
        pass

    @property
    def base_z(self):
        return self.contours[0][0]

    @property
    def base_contour(self):
        return self.contours[0][0]
    
    @property
    def top_z(self):
        return self.contours[-1][0]
    
    @property
    def top_contour(self):
        return self.contours[-1][1]

    @property
    def height(self):
        return self.top_z - self.base_z

    def volume_est(self):
        v = 0
        for i in range(1, len(self.contours)):
            (z0,c0) = self.contours[i-1]
            (z1,c1) = self.contours[i]
            v += (z1 - z0) * (c0.bounding_sq_area() + c1.bounding_sq_area())/2
            pass
        return v+self.inherited_volume
            
    def grow(self, z, contour):
        self.contours.append((z,contour))
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
        return "Mound(%f, %s)%s" % (self.base_z, self.base_contour,
                                        ''.join([".grow(%f, %s)" % (z, c) for (z,c) in self.contours[1:]]))
                                            
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
                slices.append((z, cs))
            z     += step
            bar.next()
    return slices

def find_peaks(contours, min_vol=1.0):
    mounds   = [Mound(contours[0][0], c) for c in contours[0][1]]
    peaks    = []
    for (z,z_cs) in contours[1:]:
        new_mounds = []
        for m in mounds:
            cs = [c for c in z_cs if m.supports(c)]
            if len(cs) == 0:
                peaks.append(m)
            elif len(cs) == 1:
                new_mounds.append(m.grow(z, cs[0]))
            else:
                # fixme: proportionally weight inheritance of existing mound's mass
                vol         = m.volume_est()/len(cs)
                new_mounds += [Mound(z, c, inherited_volume=vol) for c in cs]
                pass
            pass
        for c in z_cs:
            if len([m for m in mounds if m.supports(c)]) == 0:
                new_mounds.append(Mound(z, c))
                pass
            pass
        mounds = new_mounds
    peaks += mounds
    return [(p.top_z, p.top_contour) for p in peaks if p.volume_est() > min_vol]
        
def curve(radius=3,steps=10,arc=0.25):
    y = 3
    for i in range(steps+1):
        yield( numpy.array([radius*cos(2*pi*i*arc/steps), y, radius*sin(2*pi*i*arc/steps)]) )
        
def tube(path, r=1, fn=64):
    triangles = []
    prev      = next(path)
    prev_u    = None
    prev_v    = None
    first     = True
    for point in path:
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
            for i in range(fn):
                triangles.append(
                    numpy.array([prev + prev_u*cos( (2*pi*i)/fn) + prev_v*sin( (2*pi*i)/fn),
                                 prev,
                                 prev + prev_u*cos( (2*pi*(i+1))/fn) + prev_v*sin( (2*pi*(i+1))/fn)])
                )
                pass
            pass
        for i in range(fn):
            # upward pointing triangle from prev to point
            triangles.append(
                numpy.array([prev + prev_u*cos( (2*pi*i)/fn) + prev_v*sin( (2*pi*i)/fn),
                             prev + prev_u*cos( (2*pi*(i+1))/fn) + prev_v*sin( (2*pi*(i+1))/fn),
                             point + u*cos((2*pi*(i+0.5))/fn) + v*sin((2*pi*(i+0.5))/fn)])
            )
            # downward pointing triangles from point to prev
            triangles.append(
                numpy.array([point + u*cos( (2*pi*(i-0.5))/fn) + v*sin( (2*pi*(i-0.5))/fn),
                             point + u*cos( (2*pi*(i+0.5))/fn) + v*sin( (2*pi*(i+0.5))/fn),
                             prev  + prev_u*cos((2*pi*i)/fn) + prev_v*sin((2*pi*i)/fn)])
            )
            pass
        prev = point
        prev_u = u
        prev_v = v
        pass
    # Make the top of the tube, a disc in the plane defined by
    # prev_u,prev_v at point prev
    for i in range(fn):
        triangles.append(
            numpy.array([prev + prev_u*cos( (2*pi*(i-0.5))/fn) + prev_v*sin( (2*pi*(i-0.5))/fn),
                         prev + prev_u*cos( (2*pi*(i+0.5))/fn) + prev_v*sin( (2*pi*(i+0.5))/fn),
                         prev])
        )
        pass
    data = numpy.zeros(len(triangles), dtype=mesh.Mesh.dtype)
    for i in range(len(triangles)):
        data['vectors'][i] = triangles[i]
    return mesh.Mesh(data)

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
    cs     = [c[1].set_color(z_color(c[1].points[0][2], m)) for c in cs]
    max_x  = cs[0].max_x
    min_x  = cs[0].min_x
    max_y  = cs[0].max_y
    min_y  = cs[0].min_y
    for c in cs:
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
    pa.add_argument('--z-step', metavar='z_step', default=1.0, type=float, help='Z step between slices')
    pa.add_argument('--epsilon', metavar='epsilon', default=0.001, type=float, help='Max distance between two points considered equal')
    pa.add_argument('--min-volume', metavar='min_volume', default=1.0, type=float, help='Minium (estimated) volume of a differentiated peak')
    pa.add_argument('stl', metavar='stl')

    return pa.parse_args()

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        args = parse_arguments()
        m = load_mesh(args.stl)
        contours = slice(m, step=args.z_step, epsilon=args.epsilon)
        ps = find_peaks(contours, min_vol=args.min_volume)
        preview_contours(ps)
        pass
    pass
