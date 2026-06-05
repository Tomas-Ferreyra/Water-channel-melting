#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:36:46 2026

@author: tomasferreyrahauchar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from tqdm import tqdm
import glob 
from time import time
import h5py

from stl import mesh

plt.rcParams.update({ 'font.size':12 })

def get_data(path, surface=True, channel=True, start=0, end=None):
    """
    Load reconstructed surface and/or water channel data

    Parameters
    ----------
    path : str
        path to folder of experiment.
    surface : bool, optional
        Load reconstructed surface data. The default is True.
    channel : bool, optional
        Load water channel data. The default is True.
    start: int
        Time of expeirment start of channel recording. The default is 0.
    end: int or None. end > start
        Time of expeirment end of channel recording. The default is None.

    Returns
    -------
    Reconstructed surface data:
        top and bottom camera reconstrucions. *_t: time (sec); *_x,*_y,*_z: x,y,z positions in mm
    Water channel data:
        tiempo: time, in sec
        Q_tunnel: flow rate, in m3/h
        T_amb: ambient temperature, in °C
        T_top: outlet temperature (after measurment section), in °C
        T_bot: inlet temperature (before measurment section), in °C
    """
    
    if surface and (not channel):
        with h5py.File(path + 'reconstructed_profile.hdf5', 'r') as f:
            
            top_t = f['Up/time'][:]
            top_x = f['Up/x'][:]
            top_y = f['Up/y'][:]
            top_z = f['Up/z'][:]
    
            dow_t = f['Down/time'][:]
            dow_x = f['Down/x'][:]
            dow_y = f['Down/y'][:]
            dow_z = f['Down/z'][:]
            
        bp = np.load(path+'back_position.npz')
        back_t, back_x = bp['time'], bp['position']
            
        return top_t, top_x, top_y, top_z, dow_t, dow_x, dow_y, dow_z, back_t, back_x, 0,0,0,0,0
    
    elif (not surface) and channel:
        filename = glob.glob(path + '2*.txt')[-1]
        expdata = np.loadtxt( filename, skiprows=8, usecols=[0,2,3,4,6,19,20,21] )
    
        tiempo = expdata[:,0]
        Q_tunnel_setp = expdata[:,1]
        Q_tunnel = expdata[:,2]
        S_pump_setp = expdata[:,3]
        Pdiff_CVF = expdata[:,4]
        T_amb = expdata[:,5]
        T_top = expdata[:,6]
        T_bot = expdata[:,7]
    
        return 0,0,0,0,0,0,0,0,0,0, tiempo[start:end], Q_tunnel[start:end], T_amb[start:end], T_top[start:end], T_bot[start:end]
    
    elif surface and channel:
        with h5py.File(path + 'reconstructed_profile.hdf5', 'r') as f:
            
            top_t = f['Up/time'][:]
            top_x = f['Up/x'][:]
            top_y = f['Up/y'][:]
            top_z = f['Up/z'][:]
    
            dow_t = f['Down/time'][:]
            dow_x = f['Down/x'][:]
            dow_y = f['Down/y'][:]
            dow_z = f['Down/z'][:]
            
        bp = np.load(path+'back_position.npz')
        back_t, back_x = bp['time'], bp['position']

        filename = glob.glob(path + '2*.txt')[-1]
        expdata = np.loadtxt( filename, skiprows=8, usecols=[0,2,3,4,6,19,20,21] )
    
        tiempo = expdata[:,0]
        Q_tunnel_setp = expdata[:,1]
        Q_tunnel = expdata[:,2]
        S_pump_setp = expdata[:,3]
        Pdiff_CVF = expdata[:,4]
        T_amb = expdata[:,5]
        T_top = expdata[:,6]
        T_bot = expdata[:,7]

        return top_t, top_x, top_y, top_z, dow_t, dow_x, dow_y, dow_z, back_t, back_x, \
                    tiempo[start:end], Q_tunnel[start:end], T_amb[start:end], T_top[start:end], T_bot[start:end]
    

def aspect_ratio_fig(ax, xdata, ydata, zdata ):
    
    lxd, lxi = np.nanmax(xdata), np.nanmin(xdata)
    lyi, lyd = np.nanmin(ydata), np.nanmax(ydata)
    lzd, lzi = np.nanmin(zdata), np.nanmax(zdata)
    
    ax.set_box_aspect([2, 2 * np.abs(lyi-lyd) / np.abs(lxi-lxd) ,  2 * np.abs(lzi-lzd) / np.abs(lxi-lxd)], zoom=1.05)
    ax.set_zlim(lzd-5,lzi+5)
    ax.set_xlim(lxd+5,lxi-5)
    ax.set_ylim(lyi-5,lyd+5)
    
def create_side( vn,wn ):
    return np.vstack( (vn,wn) ).T.ravel()

def bottom_point( dzb, dyb, bot_val = -100. ):
    dny, dnz = dzb.shape
    borz = np.hstack( ( dzb[0], [dzb[0,-1]] * (dny-2), dzb[0][::-1], [dzb[0,0]] * (dny-1) ) )
    bory = np.hstack( ( [dyb[0,0]] * dnz, dyb[1:-1,0], [dyb[-1,0]] * dnz, dyb[:-1,0][::-1] ) )
    bot_po = np.vstack( ( np.vstack( (borz, bory, [bot_val ]*len(borz) ) ).T, [0,0,bot_val ] ) )
    return bot_po
    
def find_faces_bot( bot_po, flip=False ):
    nn = len(bot_po)
    faces = []
    for m in range(0,nn-2):
            # First triangle
            face1 = np.array( [m, m+1, -1] )
            if flip: faces.append(face1[::-1]) 
            else: faces.append(face1) 
    
    return np.array(faces)
    
def find_faces_grid( xx, flip=False ):
    ny, nx = xx.shape
    faces = []
    for m in range(ny-1):
        for n in range(nx-1):
            # First triangle
            face1 = np.array( [m * nx + n , m * nx + (n+1), (m+1) * nx + n] )
            if flip: faces.append(face1[::-1]) 
            else: faces.append(face1) 
            
            # Second triangle
            face2 = np.array( [m * nx + (n+1) , (m+1) * nx + (n+1), (m+1) * nx + n] )
            if flip: faces.append(face2[::-1]) 
            else: faces.append(face2) 

    return np.array(faces)

def find_faces_side( vn, flip=False ):
    nn = len(create_side(vn,vn))
    faces = []
    for m in range(0,nn-3,2):
            # First triangle
            face1 = np.array( [m, m+1, m+2] )
            if flip: faces.append(face1[::-1]) 
            else: faces.append(face1) 
            
            # Second triangle
            face2 = np.array( [m+1, m+3, m+2] )
            if flip: faces.append(face2[::-1])
            else: faces.append(face2)
    return np.array(faces)


def plot_triangles(points, faces):
    """
    Plot triangles from a set of 2D points and face indices.

    Parameters:
        points (array-like): shape (n,2), list of (x,y) coordinates
        faces (array-like): shape (m,3), each row has indices into `points`
        show_points (bool): whether to plot the points
        line_color (str): color of triangle edges
        fill (bool): whether to fill triangles
    """
    points = np.asarray(points)
    faces = np.asarray(faces)

    for tri in faces:
        triangle = points[tri]  # shape (3,2)
        # Close the triangle by repeating the first point
        triangle_closed = np.vstack([triangle, triangle[0]])
        plt.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'k-', alpha=0.5)
        
def plot_lims( lims, marker, zorder=None ):
    plt.plot( [lims[0],lims[1],lims[1],lims[0],lims[0]], [lims[2],lims[2],lims[3],lims[3],lims[2]], marker, zorder=zorder )

def loop_faces(ini, cube, vertices, faces ):
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i + ini][j] = vertices[f[j],:]
    return i+1 + ini

def rotate_to_xy_plane(X, Y, Z, center=True):
    """
    Rotate (X, Y, Z) so the best-fit plane becomes parallel to XY plane.

    Parameters
    ----------
    X, Y, Z : 2D arrays (same shape)
        Grid or point cloud coordinates.
    center : bool
        If True, rotate around centroid (recommended).
        If False, rotate around origin.

    Returns
    -------
    Xr, Yr, Zr : rotated arrays (same shape as input)
    R : rotation matrix (3x3)
    """

    # Flatten
    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = Z.ravel()

    # ---- 1. Fit plane Z = aX + bY + c ----
    A = np.c_[Xf, Yf, np.ones_like(Xf)]
    coeffs, *_ = np.linalg.lstsq(A, Zf, rcond=None)
    a, b, c = coeffs

    # ---- 2. Plane normal ----
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)

    target = np.array([0.0, 0.0, 1.0])

    # ---- 3. Rodrigues rotation ----
    v = np.cross(normal, target)
    s = np.linalg.norm(v)
    c_dot = np.dot(normal, target)

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    if s == 0: R = np.eye(3)
    else: R = np.eye(3) + vx + (vx @ vx) * ((1 - c_dot) / (s**2))

    # ---- 4. Apply rotation ----
    points = np.stack([Xf, Yf, Zf], axis=1)

    if center:
        centroid = points.mean(axis=0)
        points = points - centroid
        rotated = points @ R.T
        rotated = rotated + centroid
    else:
        rotated = points @ R.T

    # Reshape back
    Xr = rotated[:, 0].reshape(X.shape)
    Yr = rotated[:, 1].reshape(Y.shape)
    Zr = rotated[:, 2].reshape(Z.shape)

    return Xr, Yr, Zr, R

def interpolate_gaussian_stl( xdata, ydata, zdata, zn, yn, dist_threshold=3, sigmas=[5,10] ):
    sig_x, sig_y = sigmas
    sig_x2, sig_y2 = sig_x**2, sig_y**2
    values = xdata
    
    # zn = np.linspace( np.min(zdata), np.max(zdata), grid_size[0] )
    # yn = np.linspace( np.min(ydata), np.max(ydata), grid_size[1] )
    # zn,yn = np.meshgrid(zn,yn)

    points = np.column_stack((zdata, ydata))
    tree = cKDTree(points)
    grid_points = np.column_stack((zn.ravel(), yn.ravel()))
    dist, _ = tree.query(grid_points, k=1)
    gdist = dist.reshape(zn.shape)

    gg = np.full_like(zn, np.nan) * 1.

    ny,nx = np.shape(zn)
    for l in tqdm(range(ny)):
        for j in range(nx):
            zp,yp = zn[l,j], yn[l,j]
            dist2 = (zdata - zp)*(zdata - zp) / sig_x2 + (ydata - yp)*(ydata - yp) / sig_y2
            wiegh = np.exp( -dist2/2 )

            try: gg[l,j] = np.average(values, weights=wiegh)
            except ZeroDivisionError: gg[l,j] = 0.
            
    gg[ gdist > dist_threshold ] = np.nan    
    return gg

def bump_func(x, shifts, a):
    mp = x[-1]/2
    xp,xn = x[x>=mp], x[x<mp]
    
    xn = (xn-(mp-shifts[0]))/a
    xp = (xp-shifts[1])/a

    fn = np.zeros_like(xn)
    fn[:] = 1 / (1 + np.exp( (2*xn-1) / (xn**2 - xn)) )
    fn[xn <= 0], fn[xn >= 1] = 0, 1
    
    fp = np.zeros_like(xp)
    fp[:] = 1 / (1 + np.exp( (2*xp-1) / (xp**2 - xp)) )
    fp[xp <= 0], fp[xp >= 1] = 0, 1

    f = np.concatenate( [fn[::-1], fp] )
    return 1-f

def bump2d(gg, lims, a, shifts=[0,0,0,0]):
    with np.errstate(divide='ignore'):
        ny, nx = gg.shape
        
        x, y = np.arange(nx), np.arange(ny)
        limx, limy =  np.array([lims[0],lims[1]]), np.array([lims[2],lims[3]])
        shiftx, shifty =  np.array([shifts[0],shifts[1]]), np.array([shifts[2],shifts[3]])
        
        sx, sy = limx + shiftx, limy + shifty
        print(sx, limx)
        print(sy, limy)
        
        bux, buy = bump_func(x, sx, a), bump_func(y, sy, a)
    
    bux, buy = np.meshgrid(bux,buy)
    return bux * buy
    
    
all_folders = ['25-08-07','25-09-29','25-10-24','25-11-17','25-12-12','26-01-27','26-02-17','26-03-05' ]
grid_vels   = [ 0.5      , 0.3      , 0.9      , 0.3      , 0.0      , 0.3      , 0.9      , 0.3       ]
nf = len(all_folders)

starts = [ 1900 , 650  , 750  , 1250 , 2060 , 1000 , 108  , 190640 ]
ends   = [ 21700, 28000, 28600, 35250, 18980, 12580, 14980, 205690 ]

rho_ice = 916.8 # kg / m^3
latent = 334e3 # m^2 / s^2
thcon = 0.6 # m kg / s^3 °C
len_ice = 0.71 #m
#%%

from scipy.spatial import cKDTree
from scipy.signal.windows import tukey
import matplotlib.tri as mtri

f = 7
print(all_folders[f])
path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'
surface = True
channel = True
top_t, top_x, top_y, top_z, bot_t, bot_x, bot_y, bot_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data(path, surface, channel)


t1 = time()
 # i = 54 #f=0
# i = 45 #f=7 
i = 2 #f=5
filt = ~np.isnan(top_x[i])
filb = ~np.isnan(bot_x[i])

d_th = 2
# grid_size = [40, 2000]

# sigmas = [5.,3.]
sigmas = [10.,3.]

xdata = np.concatenate( [top_x[i][filt],bot_x[i][filb]] )
ydata = np.concatenate( [top_y[i][filt],bot_y[i][filb]] )
zdata = np.concatenate( [top_z[i][filt],bot_z[i][filb]] )
# zdata, ydata, xdata = zdata - np.nanmean(zdata), ydata - np.nanmean(ydata), xdata - np.nanmean(xdata)  

# zrot, yrot, xrot, Rm = rotate_to_xy_plane(zdata, ydata, xdata)

dz, dy = 1., 1. #0.25, 0.25
zb, yb = np.arange(-154,154+dz, dz), np.arange(-359,359+dy, dy)
zb,yb = np.meshgrid(zb,yb)

t1 = time()

# gg_g = interpolate_gaussian_stl( xrot, yrot, zrot, zb, yb, dist_threshold=200, sigmas=sigmas )
# zn, yn, gg = zb, yb, gg_g

t2 = time()
# print(t2-t1)


plt.figure()

ss = plt.scatter( zdata, ydata, s=2, c=xdata )
# ss = plt.scatter( zrot, yrot, s=2, c=xrot )
cbar = plt.colorbar(ss, location='bottom', aspect=40)

# si = plt.scatter( zn, yn, zorder=-1, s=4, c=gg, vmin=np.nanmin(xdata), vmax=np.nanmax(xdata) )
# cbar = plt.colorbar(si, location='bottom', aspect=40)

# plt.axis('equal')
plt.xlabel(r'$x$ (mm)')
plt.ylabel(r'$y$ (mm)')
plt.show()



#%%

a = 70 #280
lims = np.array( [ np.argmin(np.abs(np.nanmin(zdata) - zn[0]))  , np.argmin(np.abs(np.nanmax(zdata) - zn[0]))  ,
                  np.argmin(np.abs(np.nanmin(ydata) - yn[:,0])), np.argmin(np.abs(np.nanmax(ydata) - yn[:,0])) ] )
sh = [0,0,0,0]
bu = bump2d(gg, lims, a, shifts=sh)

ones = np.ones_like(gg) * 0.
fullprof = bu * gg + (1-bu) * ones



# fig = plt.figure()
# ax = plt.axes(projection='3d')

# tridat = mtri.Triangulation( zdata, ydata)
# ax.plot_trisurf( zdata, ydata, -xdata, triangles=tridat.triangles, cmap=plt.cm.jet )
# aspect_ratio_fig(ax, zdata, ydata, -xdata)

# plt.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# zg,yg,xg = zn.ravel(), yn.ravel(), gg.ravel()
# trigri = mtri.Triangulation( zg, yg)
# ax.plot_trisurf( zg, yg, -xg, triangles=trigri.triangles, cmap=plt.cm.jet )
# aspect_ratio_fig(ax, zg, yg, -xg)

# plt.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# zg,yg,xg = zn.ravel(), yn.ravel(), fullprof.ravel()
# trigri = mtri.Triangulation( zg, yg)
# ax.plot_trisurf( zg, yg, -xg, triangles=trigri.triangles, cmap=plt.cm.jet )
# aspect_ratio_fig(ax, zg, yg, -xg)

# plt.show()

plt.figure()
plt.imshow( -fullprof )
plt.show()

#%%

from stl import mesh


bottom = np.ones_like(fullprof) * (-100.)

top_face = np.column_stack( [zn.ravel(), yn.ravel(), -fullprof.ravel()] )
bot_face = np.column_stack( [zn.ravel(), yn.ravel(), bottom.ravel()] )

side1 = np.column_stack( [create_side(zn[0],zn[0]), create_side(yn[0],yn[0]), create_side(-fullprof[0],bottom[0])] )
side2 = np.column_stack( [create_side(zn[:,0],zn[:,0]), create_side(yn[:,0],yn[:,0]), create_side(-fullprof[:,0],bottom[:,0])] )
side3 = np.column_stack( [create_side(zn[-1],zn[-1]), create_side(yn[-1],yn[-1]), create_side(-fullprof[-1],bottom[-1])] )
side4 = np.column_stack( [create_side(zn[:,-1],zn[:,-1]), create_side(yn[:,-1],yn[:,-1]), create_side(-fullprof[:,-1],bottom[:,-1])] )


faces_top, faces_bot = find_faces_grid(zn), find_faces_grid(zn, flip=True)
faces_side1, faces_side2 = find_faces_side( zn[0] ), find_faces_side( zn[:,0], flip=True )
faces_side3, faces_side4 = find_faces_side( zn[-1], flip=True ), find_faces_side( zn[:,-1] )

numtri = len(faces_top) + len(faces_bot) + len(faces_side1) + len(faces_side2) + len(faces_side3) + len(faces_side4) 

cube = mesh.Mesh(np.zeros( numtri, dtype=mesh.Mesh.dtype))
ini = loop_faces(0  , cube, top_face, faces_top )
ini = loop_faces(ini, cube, bot_face, faces_bot )
ini = loop_faces(ini, cube, side1, faces_side1 )
ini = loop_faces(ini, cube, side2, faces_side2 )
ini = loop_faces(ini, cube, side3, faces_side3 )
ini = loop_faces(ini, cube, side4, faces_side4 )

save_path = './Documents/'
name = '65vel_03grid(1)_smallest.stl'

cube.save( save_path + name )

#%%%
# =============================================================================
# Import
# =============================================================================

from stl import mesh
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.feature import peak_local_max

filepath = './Documents/'

name, lims = '50vel_05grid.stl', np.array([ 351,  870,  277, 2572])
# name, lims = '65vel_03grid(1).stl', [367, 891, 321, 2560] # holes [2,8,12]
# name, lims = '65vel_03grid(2).stl', [354, 876, 328, 2534] # holes [0,2,3]

the_mesh = mesh.Mesh.from_file(filepath + name)
verts = the_mesh.vectors


nt,nb,n1,n2,n3,n4 = 7076608, 7076608, 2464, 5744, 2464, 5744
cuts = np.cumsum([nt,nb,n1,n2,n3,n4]) 

top = verts[:cuts[0],:,:]
top = top.reshape( (len(top)*3,3) )
points_top = np.unique(top, axis=0)

dz, dy = 0.25, 0.25
zb, yb = np.arange(-154,154+dz, dz), np.arange(-359,359+dy, dy)
zb,yb = np.meshgrid(zb,yb)
gt = np.zeros_like(zb) 
ny, nz = gt.shape

for n in range( len(points_top) ):    
    pz, py, px = points_top[n]
    iz, iy = int( (pz + 154)/dz ), int( (py + 359)/dy ) 
    gt[iy, iz] = px


#%%

cmin = peak_local_max( -gt, min_distance=50  )


plt.figure()

plt.imshow( gt, origin='lower', cmap='viridis' )

plt.plot( cmin[:,1], cmin[:,0], 'r.' )
for n in range(len(cmin)):
    plt.text(cmin[n,1], cmin[n,0], str(n))
plt.plot( [lims[0],lims[1],lims[1],lims[0],lims[0]], [lims[2],lims[2],lims[3],lims[3],lims[2]], 'r-'  )

plt.colorbar()
plt.show()

#%%
positions = []

plt.figure()
# for n in [2,8,12]:
for n in [0,1]:
    
    ggt = np.gradient( gaussian(gt[:,cmin[n,1]], 20) )    
    filf, filb  = np.where( np.diff( ggt[cmin[n,0]:] ) <=0)[0], np.where( np.diff( ggt[:cmin[n,0]] ) <=0)[0]
    forw, fbac = filf[0], filb[-1]
    filbf = np.where( np.diff( ggt[:fbac] ) >=0)[0]
    ffba = filbf[-1]
    
    if n in [0,1]: ffba = 670

    positions.append( cmin[n] )
    positions.append( [cmin[n,0]+forw, cmin[n,1]] )
    positions.append( [fbac, cmin[n,1]] )
    positions.append( [ffba, cmin[n,1]] )

    # plt.plot( np.diff(ggt), '.-'  )
    # plt.plot( cmin[n,0], np.diff(ggt)[cmin[n,0]], 'r.' )
    # plt.plot( ggt, '.-'  )
    # plt.plot( cmin[n,0], ggt[cmin[n,0]], 'r.' )
    
    plt.plot( gt[:,cmin[n,1]], '.-'  )
    plt.plot( cmin[n,0], gt[cmin[n,0], cmin[n,1]], 'r.' )
    plt.plot( cmin[n,0]+forw, gt[cmin[n,0]+forw, cmin[n,1]], 'b.' )
    plt.plot( fbac, gt[fbac, cmin[n,1]], 'b.' )
    plt.plot( ffba, gt[ffba, cmin[n,1]], 'b.' )

plt.grid()
plt.show()

positions = np.array(positions)

plt.figure()

plt.imshow( gt, origin='lower', cmap='viridis' )
plt.plot( positions[:,1], positions[:,0], 'r.' )
plt.plot( [lims[0],lims[1],lims[1],lims[0],lims[0]], [lims[2],lims[2],lims[3],lims[3],lims[2]], 'r-'  )
plt.colorbar()
plt.show()

for n in range(len(positions)):
    # print( np.array( [ zb[positions[n,0],positions[n,1]], yb[positions[n,0],positions[n,1]], gt[positions[n,0],positions[n,1]] ]) )
    print('[', zb[positions[n,0],positions[n,1]], ',', yb[positions[n,0],positions[n,1]], ']' ) 

#%%

dz, dy = 0.25, 0.25
zn, yn = np.arange(-154,154+dz, dz), np.arange(-359,359+dy, dy)

((zn + 154)/0.25), np.round((zn + 154)/dz)



#%%

#%%
# =============================================================================
# Make smaller (and flip faces is necessary)
# =============================================================================

save_path = './Documents/'

name, lims = '50vel_05grid.stl', np.array([ 351,  870,  277, 2572])
# name, lims = '65vel_03grid(1).stl', np.array([367, 891, 321, 2560])
# name, lims = '65vel_03grid(2).stl', np.array([354, 876, 328, 2534])

your_mesh = mesh.Mesh.from_file(save_path + name)
verts = your_mesh.vectors

nt,nb,n1,n2,n3,n4 = 7076608, 7076608, 2464, 5744, 2464, 5744
cuts = np.cumsum([nt,nb,n1,n2,n3,n4]) 

top = verts[:cuts[0],:,:]
top = top.reshape( (len(top)*3,3) )
points_top = np.unique(top, axis=0)


dz, dy = 0.25, 0.25
zb, yb = np.arange(-154,154+dz, dz), np.arange(-359,359+dy, dy)
zb,yb = np.meshgrid(zb,yb)
gt = np.zeros_like(zb) 
ny, nz = gt.shape

for n in range( len(points_top) ):    
    pz, py, px = points_top[n]
    iz, iy = int( (pz + 154)/dz ), int( (py + 359)/dy ) 
    gt[iy, iz] = px


bot_val = -100.
pad = 280
paddin = np.array([-pad,pad,-pad,pad])
plim = lims + paddin

cut_col = np.hstack( ( np.arange(1,plim[0]), np.arange(plim[1]+1,nz-1)  ) )
cut_row = np.hstack( ( np.arange(1,plim[2]), np.arange(plim[3]+1,ny-1)  ) )

dzb, dyb, dgt = np.delete(zb, cut_col, axis=1 ), np.delete(yb, cut_col, axis=1 ), np.delete(gt, cut_col, axis=1 )
dzb, dyb, dgt = np.delete(dzb, cut_row, axis=0 ), np.delete(dyb, cut_row, axis=0 ), np.delete(dgt, cut_row, axis=0 )

dny, dnz = dzb.shape

dtop = np.vstack((dzb.ravel(),dyb.ravel(),dgt.ravel())).T
dbot = bottom_point( dzb, dyb, bot_val = bot_val )

dsi1 = np.column_stack( [create_side(dzb[0],dzb[0]), create_side(dyb[0],dyb[0]), create_side(-dgt[0], np.full_like(dgt[0], bot_val) ) ] )
dsi2 = np.column_stack( [create_side(dzb[:,0],dzb[:,0]), create_side(dyb[:,0],dyb[:,0]), create_side(-dgt[:,0], np.full_like(dgt[:,0], bot_val) )] )
dsi3 = np.column_stack( [create_side(dzb[-1],dzb[-1]), create_side(dyb[-1],dyb[-1]), create_side(-dgt[-1], np.full_like(dgt[-1], bot_val) )] )
dsi4 = np.column_stack( [create_side(dzb[:,-1],dzb[:,-1]), create_side(dyb[:,-1],dyb[:,-1]), create_side(-dgt[:,-1], np.full_like(dgt[:,-1], bot_val) )] )


fa_top, fa_bot = find_faces_grid( dzb ), find_faces_bot(dbot, flip=True)
fa_si1, fa_si2 = find_faces_side( dzb[0] ), find_faces_side( dzb[:,0], flip=True )
fa_si3, fa_si4 = find_faces_side( dzb[-1], flip=True ), find_faces_side( dzb[:,-1] )

num_tri = len(fa_top) + len(fa_bot) + len(fa_si1) + len(fa_si2) + len(fa_si3) + len(fa_si4)

new_mesh = mesh.Mesh(np.zeros( num_tri, dtype=mesh.Mesh.dtype))
ini = loop_faces(0  , new_mesh, dtop, fa_top )
ini = loop_faces(ini, new_mesh, dbot, fa_bot )
ini = loop_faces(ini, new_mesh, dsi1, fa_si1 )
ini = loop_faces(ini, new_mesh, dsi2, fa_si2 )
ini = loop_faces(ini, new_mesh, dsi3, fa_si3 )
ini = loop_faces(ini, new_mesh, dsi4, fa_si4 )

save_path = './Documents/'
# name = '50vel_05grid.stl'
name = '50vel_05grid_coarser.stl'

new_mesh.save( save_path + name )

#%%


plt.figure()
# plt.scatter(pp[:,0], pp[:,1])
# plot_triangles(pp[:,:2], ff)

# plt.scatter(dtop[:,0], dtop[:,1])
# plot_triangles(dtop[:,:2], fa_top[-5000:])
# plt.scatter(dbot[:,0], dbot[:,1] )
# plot_triangles(dbot[:,:2], fa_bot[:])

plt.scatter(dsi2[:,1], dsi2[:,2] )
plot_triangles(dsi2[:,[1,2]], fa_si2[:])

plt.show()


#%%
rlim = [ zb[0][plim[0]], zb[0][plim[1]], yb[:,0][plim[2]], yb[:,0][plim[3]] ]

plt.figure()

plt.scatter( zb.ravel(), yb.ravel(), s=5  )
plt.scatter( dtop[:,0], dtop[:,1], s=5  )
plot_lims(rlim, 'r-', zorder=0)

plt.show()

#%%
pad = 280

plt.figure()
plt.imshow( gt, origin='lower' )

# plt.imshow( zb, origin='lower' )
# plt.imshow( yb, origin='lower' )


plot_lims(lims, 'r-')

# for pad in [280,250,200]:
for pad in [280]:
    paddin = np.array([-pad,pad,-pad,pad])
    plot_lims( lims + paddin  , 'b-')

plt.colorbar()
plt.show()

plt.figure()
plt.plot( gt[500,:] == 0., '.-' )
plt.vlines( lims[:2], -10,10, colors='red' )
plt.vlines( lims[:2] + paddin[:2], -10,10, colors='green' )
plt.grid()
plt.show()

plt.figure()
plt.plot( gt[:,500] == 0., '.-' )
plt.vlines( lims[2:], -10,10, colors='red' )
plt.vlines( lims[2:] + paddin[2:], -10,10, colors='green' )
plt.grid()
plt.show()

#%%
# Flip faces



save_path = './Documents/'
# name = '50vel_05grid.stl'
# name = '65vel_03grid(2).stl'

# name, lims = '65vel_03grid(1).stl', [367, 891, 321, 2560]
name, lims = '65vel_03grid(2).stl', [354, 876, 328, 2534] 

your_mesh = mesh.Mesh.from_file(save_path + name)
verts = your_mesh.vectors

nt,nb,n1,n2,n3,n4 = len(faces_top), len(faces_bot), len(faces_side1), len(faces_side2), len(faces_side3), len(faces_side4) 
cuts = np.cumsum([nt,nb,n1,n2,n3,n4]) 

vertsb = verts[cuts[0]:cuts[1], ::1, ::1]
bot = np.reshape( vertsb,  (len(vertsb)*3,3) )
points_bot = np.unique(bot, axis=0)

# verts[cuts[2]:cuts[3],:,:] = verts[cuts[2]:cuts[3], ::-1, ::1]
# verts[cuts[3]:cuts[4],:,:] = verts[cuts[3]:cuts[4], ::-1, ::1]

#%%
dz, dy = 0.25, 0.25
bz, by = np.arange(-154,154+dz, dz), np.arange(-359,359+dy, dy)
borderz = np.concatenate( (bz, [bz[-1]]*len(by), bz[::-1], [bz[0]] * len(by)  ) )
bordery = np.concatenate( ([by[0]]*len(bz), by[::1], [by[-1]] * len(bz), by  ) )
bot_po = np.vstack( ( np.vstack( (borderz, bordery, [-100]*len(borderz) ) ).T, [0,0,-100] ) )
botfa = find_faces_bot( bot_po, flip=True )

numtri = nt + n1 + n2 + n3 + n4 + len(botfa)
c1 = nt + n1 + n2 + n3 + n4 

new_mesh = mesh.Mesh(np.zeros( numtri, dtype=mesh.Mesh.dtype))

new_mesh.vectors[ :cuts[0] ] = verts[ :cuts[0] ]
new_mesh.vectors[ cuts[0]:c1 ] = verts[ cuts[1]: ]
fin = loop_faces(c1, new_mesh, bot_po, botfa )

save_path = './Documents/'
name = '65vel_03grid(3).stl'

new_mesh.save( save_path + name )

#%%
dz, dy = 0.25, 0.25
bz, by = np.arange(-154,154+dz, dz), np.arange(-359,359+dy, dy)
borderz = np.concatenate( (bz, [bz[-1]]*len(by), bz[::-1], [bz[0]] * len(by)  ) )
bordery = np.concatenate( ([by[0]]*len(bz), by[::1], [by[-1]] * len(bz), by  ) )
bot_po = np.vstack( ( np.vstack( (borderz, bordery, [-100]*len(borderz) ) ).T, [0,0,-100] ) )
botfa = find_faces_bot( bot_po )


plt.figure()
# plt.scatter( points_bot[:,0] , points_bot[:,1])

plt.plot( bot_po[:,0], bot_po[:,1] , '.' )
plot_triangles(bot_po[:,:2], botfa )

plt.grid()
plt.show()



#%%









xx, yy = np.meshgrid( np.arange(20), np.arange(30) )
zz = xx*0.

dx, dy = np.delete(xx, [1,3,5,-2,-4,-6], axis=0 ), np.delete(yy, [1,3,5,-2,-4,-6], axis=0 )
dx, dy = np.delete(dx, [1,3,5,-2,-4], axis=1 ), np.delete(dy, [1,3,5,-2,-4], axis=1 )
dz = dx*0.

pp = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
dp = np.vstack((dx.ravel(), dy.ravel(), dz.ravel())).T

ff = find_faces_grid( xx )
df = find_faces_grid( dx )


plt.figure()
# plt.scatter(pp[:,0], pp[:,1])
# plot_triangles(pp[:,:2], ff)

plt.scatter(dp[:,0], dp[:,1])
plot_triangles(dp[:,:2], df)

plt.show()





