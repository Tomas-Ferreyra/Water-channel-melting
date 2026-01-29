#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 15:41:15 2025

@author: tomasferreyrahauchar
"""
from itertools import combinations
from skimage.morphology import remove_small_objects, disk, binary_erosion, binary_closing, binary_dilation, binary_opening
import imageio.v2 as imageio
# import imageio.v3 as iio
from tqdm import tqdm
from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.signal import find_peaks

from skimage.color import rgb2gray
from skimage.filters import gaussian, frangi
from skimage.morphology import remove_small_objects, disk, binary_erosion, binary_closing, binary_dilation
from skimage.measure import label, regionprops
from skimage.util import img_as_ubyte

def get_line(x1, x2, y1, y2, x):
    m = (y2-y1)/(x2-x1)
    y = m * (x-x1) + y1
    return y, m

def position_grid(vid, back, cutoff, n_frames, interval=10, in_frame=0, lims=[None, None, None, None]):
    xmed, ymed, ies = [], [], []
    for i in tqdm(range(in_frame, n_frames, interval)):
        im = np.array(vid.get_data(i))
        im = 0.3 * im[lims[0]:lims[1], lims[2]:lims[3], 0] + \
            0.7 * im[lims[0]:lims[1], lims[2]:lims[3], 1]
        imb = im - back[lims[0]:lims[1], lims[2]:lims[3]]
        bni = np.mean(np.where(imb < cutoff), axis=1)
        xmed.append(bni[1])
        ymed.append(bni[0])
        ies.append(i)

    return np.array(xmed), np.array(ymed), np.array(ies)


def get_frames(xms, ies, h1, h2, amount=14, distan=30):
    peksp, _ = find_peaks(np.gradient(-xms), distance=distan, height=h1)
    peksm, _ = find_peaks(np.gradient(xms), distance=distan, height=h2)

    peksp, peksm = peksp[:amount-1] - 3, peksm[amount-1]-3
    peks = np.hstack((peksp, peksm))
    return ies[peks], peks


def image_position(frames, vids, lengths):
    ny, nx, _ = np.shape(vids[0].get_data(0))
    ll = np.cumsum(np.hstack((0, lengths)))
    nv = np.array([np.sum(ll <= frames[i])-1 for i in range(len(frames))])
    truframes = frames - ll[nv]

    ims = []
    for i in tqdm(range(len(frames))):
        ima = np.zeros((ny, nx))
        for j in range(-59, 1):
            im = np.array(vids[nv[i] * ((truframes[i] + j) > -1)].get_data(
                truframes[i] + j + lengths[0] * ((truframes[i] + j) < 0)))
            imn = 0.3 * im[:, :, 0] + 0.7 * im[:, :, 1]
            ima += imn
        ims.append(ima/60)
    return ims


def get_points(imb, tol_f=0.01, tol_ga=0.3):
    g1 = normalize(gaussian(imb, 1))
    gl = normalize(gaussian(imb, 10))
    imt = frangi(g1) > tol_f

    rim = remove_small_objects(imt, 45)
    pos = imt*1-rim
    ker = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    op = binary_opening(pos, ker)
    dots = remove_small_objects(op, 10)
    dots[gl < tol_ga] = 0

    lab = label(dots)
    props = regionprops(lab, -g1)
    cy = np.array([(props[i].centroid_weighted)[0] for i in range(len(props))])
    cx = np.array([(props[i].centroid_weighted)[1] for i in range(len(props))])

    tt = np.arange(len(cx)).astype(str)
    pp = np.vstack((cx, cy)).T
    return g1, pp, tt


def normalize(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def iscollinear(points):
    """
    Parameters
    ----------
    points : (N,M,2) array
        N sets of M points to check (in 2D).

    Returns
    -------
    array of booleans.
    """
    check = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        k = points[i]
        vecdiv = (k-k[0]) * (k[1]-k[0])[::-1]
        ks = np.diff(vecdiv, axis=1)[:, 0]

        check[i] = np.all(ks == 0)

    return check


def find_basis(out, position, gridpos, eps, lim_ind):
    order = np.argsort(np.sum((out - position)**2, axis=1))
    orout, orgridi = out[order], gridpos[order]

    for i in range(3, len(orout)+1):
        subs = list(combinations(np.arange(i), 3))
        subs_pts, subs_grid = orout[subs], orgridi[subs]
        is_col = iscollinear(subs_grid)
        subs_pts, subs_grid = subs_pts[~is_col], subs_grid[~is_col]

        if len(subs_pts) > 0:
            ibase = np.argmin(np.sum((subs_pts - position)**2, axis=(1, 2)))
            basep, baseg = subs_pts[ibase], subs_grid[ibase]
            break

    if len(subs_pts) > 0:
        vp1, vp2 = basep[1] - basep[0], basep[2] - basep[0]
        vg1, vg2 = baseg[1] - baseg[0], baseg[2] - baseg[0]
        vpoi = position - basep[0]

        a = np.linalg.det(np.vstack((vp1, vp2)))
        b = np.linalg.det(np.vstack((vp1, vpoi)))
        c = np.linalg.det(np.vstack((vpoi, vp2)))

        pti = c/a * vg1 + b/a * vg2 + baseg[0]
        newind = np.round(pti)
        err = np.sqrt(np.sum((newind - pti)**2))

        belongs_grid = (err < eps) and (newind[0] >= lim_ind[0]) and (
            newind[0] <= lim_ind[1]) and (newind[1] >= lim_ind[2]) and (newind[1] <= lim_ind[3])
    else:
        newind = np.zeros(2)*np.nan
        belongs_grid = False

    return newind, belongs_grid


def find_indices(points, known, eps=0.25, lim_ind=[-1e5, 1e5, -1e5, 1e5], gps=[1,1]):
    """
    Parameters
    ----------
    points : (N,2)-array
        Points to match in  a grid.
    known : list of 3 integers 
        Indeces in points for (0,0), (1,0) and (0,1).
    eps : TYPE, optional
        DESCRIPTION. The default is 0.25.
    lim_ind : list length 4, optional
        min and max indices for [left,right,bottom,top]
    Returns
    -------
    (M,2)-array
        position of points in grid.
    (M,2)-array
        grid position of points.
    """
    N, col = np.shape(points)
    if col != 2:
        return 'Points need to be 2D'

    pts = np.copy(points)
    out, gridpos = np.zeros((3, 2)), np.zeros((3, 2))
    out[0], out[1], out[2] = pts[known[0]], pts[known[1]], pts[known[2]]
    gridpos[1, 0], gridpos[2, 1] = gps[0], gps[1]

    pts = np.delete(pts, known, axis=0)

    itere = 0
    while len(pts) > 0 and itere <= N:
        itere += 1
        med = np.mean(out, axis=0)
        iclosest = np.argmin(np.sum((pts - med)**2, axis=1))
        position = pts[iclosest]
        pts = np.delete(pts, iclosest, axis=0)

        newind, belongs_grid = find_basis(out, position, gridpos, eps, lim_ind)
        if belongs_grid:
            out = np.vstack((out, position))
            gridpos = np.vstack((gridpos, newind))

    return out, gridpos.astype('int64')


def cal_direct(v, points, order):
    """
    Parameters
    ----------
    v : list
        list of coefficients from fit.
    points : (N,3) array
        (x,y,z) real world coordinates.
    order : {'1','2','2b'}
        which function to use.

    Returns
    -------
    X : array
        pixel x-position of points.
    Y : array
        pixel y-position of points.
    """
    x, y, z = points.T
    if order == '1':
        xtop = v[0]*x + v[1]*y + v[2]*z + v[3]
        ytop = v[4]*x + v[5]*y + v[6]*z + v[7]
        bot = 1
        X, Y = xtop/bot, ytop/bot
        return X, Y
    if order == '2':
        xtop = v[0]*x + v[1]*y + v[2]*z + v[3] + \
            v[4]*x**2 + v[5]*y**2 + v[6]*x*y
        ytop = v[7]*x + v[8]*y + v[9]*z + v[10] + \
            v[11]*x**2 + v[12]*y**2 + v[13]*x*y
        bot = 1
        X, Y = xtop/bot, ytop/bot
        return X, Y
    if order == '2b':
        xtop = v[0]*x + v[1]*y + v[2]*z + v[3] + \
            v[4]*x**2 + v[5]*y**2 + v[6]*x*y
        ytop = v[7]*x + v[8]*y + v[9]*z + v[10] + \
            v[11]*x**2 + v[12]*y**2 + v[13]*x*y
        bot = v[14]*x + v[15]*y + v[16]*z + 1 + \
            v[17]*x**2 + v[18]*y**2 + v[19]*x*y
        X, Y = xtop/bot, ytop/bot
        return X, Y


def calib_invzl(v, points, order, x0s=[[50, 300]]):
    """
    Parameters
    ----------
    v : list
        list of coefficients from fit.
    points : (N,3) array
        (x,y,z) coordinates. (x,y) should be pixel position of point and (z) the corresponding real world position of that point
    order : {'1','2','2b'}
        which function to use.
    x0s : (M,2) array, optional
        Initial guess for minimize. M=1 or M=N. If N=1 then it would be use as initial values for the points. M=N is for provinding initial values for all points
        . The default is [[50,300]].

    Returns
    -------
    X : array
        real world x-position of points.
    Y : array
        real world y-position of points.
    """
    x, y, z = points.T
    xr, yr = np.zeros_like(x), np.zeros_like(y)

    if order == '1':
        def func(val, x, y, z):
            xtop = v[0]*val[0] + v[1]*val[1] + v[2]*z + v[3]
            ytop = v[4]*val[0] + v[5]*val[1] + v[6]*z + v[7]
            bot = 1  # v[8]*val[0] + v[9]*val[1] + v[10]*z + 1
            eq1, eq2 = xtop/bot - x, ytop/bot - y
            return eq1**2+eq2**2
    if order == '2':
        def func(val, x, y, z):
            xtop = v[0]*val[0] + v[1]*val[1] + v[2]*z + v[3] + \
                v[4]*val[0]**2 + v[5]*val[1]**2 + v[6]*val[0]*val[1]
            ytop = v[7]*val[0] + v[8]*val[1] + v[9]*z + v[10] + \
                v[11]*val[0]**2 + v[12]*val[1]**2 + v[13]*val[0]*val[1]
            # v[14]*val[0] + v[15]*val[1] + v[16]*z + 1     + v[17]*val[0]**2 + v[18]*val[1]**2 + v[19]*val[0]*val[1]
            bot = 1
            eq1, eq2 = xtop/bot - x, ytop/bot - y
            return eq1**2+eq2**2
    if order == '2b':
        def func(val, x, y, z):
            xtop = v[0]*val[0] + v[1]*val[1] + v[2]*z + v[3] + \
                v[4]*val[0]**2 + v[5]*val[1]**2 + v[6]*val[0]*val[1]
            ytop = v[7]*val[0] + v[8]*val[1] + v[9]*z + v[10] + \
                v[11]*val[0]**2 + v[12]*val[1]**2 + v[13]*val[0]*val[1]
            bot = v[14]*val[0] + v[15]*val[1] + v[16]*z + 1 + v[17] * \
                val[0]**2 + v[18]*val[1]**2 + v[19]*val[0]*val[1]
            eq1, eq2 = xtop/bot - x, ytop/bot - y
            return eq1**2+eq2**2

    for i in tqdm(range(len(x))):
        xp, yp, zp = x[i], y[i], z[i]
        x0 = x0s[i % len(x0s)]
        def funci(sol): return func(sol, xp, yp, zp)
        mini = minimize(funci, x0, method='Nelder-Mead')
        xr[i], yr[i] = mini.x[0], mini.x[1]

    return xr, yr


def calibration_fit(real_pos, pix_pos, cal_check=False):
    """
    Parameters
    ----------
    real_pos : (N,3) array
        point positions in real world.
    pix_pos : (N,2) array
        pixel position in image.
    cal_check: bool
        if True, also returns (x,y) real world coordinates of image points, by inverting the calibration.
        Usefu; for checking if calibration worked.
    Returns
    -------
    dict
        results from least squares.
    """
    t1 = time()

    def residual_firstz(v):
        xrrs, yrrs, zrrs = real_pos.T
        spxs, spys = pix_pos.T
        xtop = v[0]*xrrs + v[1]*yrrs + v[2]*zrrs + v[3]
        ytop = v[4]*xrrs + v[5]*yrrs + v[6]*zrrs + v[7]
        bot = 1  # v[8]*xrrs + v[9]*yrrs + v[10]*zrrs + 1

        X, Y = xtop / bot, ytop / bot
        return (X - spxs)**2 + (Y - spys)**2

    def residual_secondz(v):
        xrrs, yrrs, zrrs = real_pos.T
        spxs, spys = pix_pos.T
        xtop = v[0]*xrrs + v[1]*yrrs + v[2]*zrrs + v[3] + \
            v[4]*xrrs**2 + v[5]*yrrs**2 + v[6]*xrrs*yrrs
        ytop = v[7]*xrrs + v[8]*yrrs + v[9]*zrrs + v[10] + \
            v[11]*xrrs**2 + v[12]*yrrs**2 + v[13]*xrrs*yrrs
        # v[14]*xrrs + v[15]*yrrs + v[16]*zrrs + 1     + v[17]*xrrs**2 + v[18]*yrrs**2 + v[19]*xrrs*yrrs
        bot = 1

        X, Y = xtop / bot, ytop / bot
        return (X - spxs)**2 + (Y - spys)**2

    def residual_secondzb(v):
        xrrs, yrrs, zrrs = real_pos.T
        spxs, spys = pix_pos.T
        xtop = v[0]*xrrs + v[1]*yrrs + v[2]*zrrs + v[3] + \
            v[4]*xrrs**2 + v[5]*yrrs**2 + v[6]*xrrs*yrrs
        ytop = v[7]*xrrs + v[8]*yrrs + v[9]*zrrs + v[10] + \
            v[11]*xrrs**2 + v[12]*yrrs**2 + v[13]*xrrs*yrrs
        bot = v[14]*xrrs + v[15]*yrrs + v[16]*zrrs + 1 + \
            v[17]*xrrs**2 + v[18]*yrrs**2 + v[19]*xrrs*yrrs

        X, Y = xtop / bot, ytop / bot
        return (X - spxs)**2 + (Y - spys)**2

    lal = least_squares(residual_firstz, [
                        0, 1, 0, 1, 0, 1, 0, 0], method='lm', xtol=1e-12, max_nfev=10000)

    ls0 = np.zeros(14)
    ls0[:4], ls0[7:11] = lal.x[:4], lal.x[4:8]
    la2 = least_squares(residual_secondz, ls0, method='lm',
                        xtol=1e-12, max_nfev=10000)

    ls0 = np.zeros(20)
    ls0[:14] = la2.x[:14]
    lab = least_squares(residual_secondzb, ls0, method='lm',
                        xtol=1e-12, max_nfev=10000)

    t2 = time()
    print(t2-t1)

    if cal_check:
        points = np.zeros_like(real_pos)
        points[:, :2], points[:, 2] = pix_pos, real_pos[:, 2]
        Xb, Yb = calib_invzl(lab.x, points, order='2b')
        return lab, Xb, Yb

    else:
        return lab


def rotate_coords(points, coords, angle):
    """
    Parameters
    ----------
    points : (N,3) array
        points to rotate.
    coords : (0,1 or 2) list
        which two coordinates to rotate. Eg. [1,2] rotates axis y and z
    angle : float
        angle in radians.

    Returns
    -------
    points_r : (N,3) array
        array of points rotated.
    """
    c1, c2 = points[:, coords[0]], points[:, coords[1]]
    c1r = np.cos(angle) * c1 - np.sin(angle) * c2
    c2r = np.sin(angle) * c1 + np.cos(angle) * c2

    points_r = np.copy(points)
    points_r[:, coords[0]], points_r[:, coords[1]] = c1r, c2r
    return points_r

def pick_points_on_figure(x, y, im, fig_num, max_picks=3):
    labels = ["(0,0)", "(1,0)", "(0,1)"]
    picked_points = []

    fig, ax = plt.subplots()
    ax.set_title(f"Figure {fig_num + 1}: Pick {max_picks} points")
    ax.imshow(im, cmap='gray')
    sc = ax.scatter(x, y, picker=True)

    def on_pick(event):
        if event.artist != sc:
            return
        ind = event.ind[0]
        picked_x = x[ind]
        picked_y = y[ind]

        # if (picked_x, picked_y) not in picked_points:
        idx = len(picked_points)
        picked_points.append(int(ind))
        ax.plot(picked_x, picked_y, 'ro')

        # Add label near the point
        ax.text(picked_x, picked_y, f" {labels[idx]}")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        if len(picked_points) >= max_picks:
            fig.canvas.mpl_disconnect(cid)
            plt.pause(1)
            plt.close(fig)


    cid = fig.canvas.mpl_connect('pick_event', on_pick)

    while plt.fignum_exists(fig.number):
        plt.pause(0.1)  # pause briefly to process events

    return picked_points



# %%
# =============================================================================
# Angles for calibration
# =============================================================================
t1 = time()
path = '/Volumes/Ice blocks/Scan water channel/25-11-17/'
im1 = imageio.imread(path + 'Camera back/DSC_1743.NEF')
# im2 = imageio.imread('/Volumes/Ice blocks/Scan water channel/25-09-29/Camera back/DSC_0448.NEF')

t2 = time()
t2-t1
#%%

xc = np.linspace(3536, 3505, 10)
yc, mc = get_line(3536, 3505, 20, 4250, xc)

xd = np.linspace(4308, 4304, 10)
yd, md = get_line(4308, 4304, 20, 4300, xd)

plt.figure()
plt.imshow(im1[:, :, :])
plt.plot(xc, yc, 'r-')
plt.plot(xd, yd, 'g-')
plt.plot(xc+xd[0]-xc[0], yc, 'r-')
# plt.gca().invert_yaxis()
plt.show()

print(np.arctan(np.abs((mc-md)/(1+mc*md))) * 180/np.pi, '°')
print(np.arctan(np.abs((mc-md)/(1+mc*md))), 'rad')
# %%
# Not using this anymore
# xc = np.linspace(3499, 3549, 10)
# yc, mc = get_line(3499, 3549, 100, 5500, xc)

# xd = np.linspace(3466.3, 3516.1, 10)
# yd, md = get_line(3466.3, 3516.1, 50.6, 5447.5, xd)

# plt.figure()
# plt.imshow(im2[:, :, :])
# plt.plot(xd, yd, 'g-')
# # plt.plot(xc, yc, 'r-'  )
# plt.plot(xc+xd[0]-xc[0], yc, 'r-')
# plt.show()

# print(np.arctan(np.abs((mc-md)/(1+mc*md))) * 180/np.pi, '°')
# print(np.arctan(np.abs((mc-md)/(1+mc*md))), 'rad')

# %%
# =============================================================================
# Calibration with two cameras
# =============================================================================

 
t1 = time()

# Import videos
path = '/Volumes/Ice blocks/Scan water channel/25-11-17/'
vidd1 = imageio.get_reader( path + 'Camera down/DSC_0485.MOV', 'ffmpeg')  # 8055 frames
vidd2 = imageio.get_reader( path + 'Camera down/DSC_0486.MOV', 'ffmpeg')  # 8085 frames
vidd3 = imageio.get_reader( path + 'Camera down/DSC_0487.MOV', 'ffmpeg')  # 3150 frames


vidu1 = imageio.get_reader( path + 'Camera up/DSC_3738.MOV', 'ffmpeg')  # 7056 frames
vidu2 = imageio.get_reader( path + 'Camera up/DSC_3739.MOV', 'ffmpeg')  # 7080 frames
vidu3 = imageio.get_reader( path + 'Camera up/DSC_3740.MOV', 'ffmpeg')  # 1377 frames

ny, nx, _ = np.shape(vidd1.get_data(0))

# Get background images for down and up (bottom and top camera respectively)
backd = np.array(vidd3.get_data(3140))
backd = 0.3 * backd[:, :, 0] + 0.7 * backd[:, :, 1]
backu = np.array(vidu3.get_data(1300))
backu = 0.3 * backu[:, :, 0] + 0.7 * backu[:, :, 1]

t2 = time()
print(t2-t1)

# vidd1.count_frames(), vidd2.count_frames(), vidd3.count_frames(), vidu1.count_frames(), vidu2.count_frames(), vidu3.count_frames()

#%%
plt.figure()
plt.imshow( normalize( backd[:,700:1800] ) )
plt.colorbar()
plt.show()
plt.figure()
plt.imshow( backu[:,800:1950] )
plt.colorbar()
plt.show()

#%%


cutoff = -80
vid, back, n_frames = vidd1, backd, 8055
interval=10; in_frame=0; lims=[None, None, 700, 1800]

xmed, ymed, ies = [], [], []
# for i in tqdm(range(in_frame, n_frames, interval)):
for i in [0,3000,6000]:
    im = np.array(vid.get_data(i))
    im = 0.3 * im[lims[0]:lims[1], lims[2]:lims[3], 0] + \
        0.7 * im[lims[0]:lims[1], lims[2]:lims[3], 1]
    imb = im - back[lims[0]:lims[1], lims[2]:lims[3]]
    bni = np.mean(np.where(imb < cutoff), axis=1)
    xmed.append(bni[1])
    ymed.append(bni[0])
    ies.append(i)
    
    plt.figure()
    # plt.imshow( imb )
    plt.imshow( imb<cutoff )
    plt.colorbar()
    plt.show()


#%%
t1 = time()

# Find frames when grid starts moving. We'll take the frames before those for grid z-positions
limsd = [None, None, 700, 1800]
xmedd1, ymedd1, iesd1 = position_grid( vidd1, backd, -80, 8055, in_frame=0, lims=limsd)
xmedd2, ymedd2, iesd2 = position_grid( vidd2, backd, -80, 8085, in_frame=0, lims=limsd)
iesd2 += 8055
xmd, ymd, ied = np.hstack((xmedd1, xmedd2)), np.hstack((ymedd1, ymedd2)), np.hstack((iesd1, iesd2))

limsu = [None, None, 800, 1950]
xmedu1, ymedu1, iesu1 = position_grid( vidu1, backu, -100, 7056, in_frame=0, lims=limsu)
xmedu2, ymedu2, iesu2 = position_grid( vidu2, backu, -100, 7080, in_frame=0, lims=limsu)
iesu2 += 7056
xmu, ymu, ieu = np.hstack((xmedu1, xmedu2)), np.hstack((ymedu1, ymedu2)), np.hstack((iesu1, iesu2))

N_positions = 15
framesd, pekd = get_frames(xmd, ied, 15, 4, amount=N_positions, distan=15 )
framesu, peku = get_frames(xmu, ieu, 15, 5, amount=N_positions )

plt.figure()
plt.plot(ied, xmd, '.-')
plt.plot(ied[pekd], xmd[pekd], '.')
plt.plot(ieu, xmu, '.-')
plt.plot(ieu[peku], xmu[peku], '.')
plt.show()

t2 = time()
print(t2-t1)

#%%
N_positions = 15
limsd = [None, None, 700, 1800]
limsu = [None, None, 800, 1950]

framesd = np.array([  750,  1710,  2390,  3000,  3740,  4300,  4790,  5520,  6180,  7010,  7610,  8515,  9445, 10365, 12455])

framesu = np.array([  630,  1410,  1950,  2440,  3030,  3480,  3870,  4450,  4980,  5650,  6120,  6850,  7586,  8326, 10006])

# Get images for all grid z positions
imd = image_position(framesd, [vidd1, vidd2], [8055, 8085])
imu = image_position(framesu, [vidu1, vidu2], [7056, 7080])

#%%
for i in range(1):
    plt.figure()
    plt.imshow(imd[i])
    plt.show()
    plt.figure()
    plt.imshow(imu[i])
    plt.show()

#%%
# Clicking points on the image that will act as starting base for grid index algorithm
picked_points_u = {}
picked_points_d = {}

for i in range(N_positions):
# for i in [0,13]:
    imb = (imu[i]-backu)[limsu[0]:limsu[1], limsu[2]:limsu[3]]
    g1, pp, tt = get_points(imb, tol_f=0.006, tol_ga=0.2)    
    points = pick_points_on_figure(pp[:,0], pp[:,1], imb, i)
    picked_points_u[i] = points

for i in range(N_positions):
# for i in [0,13]:
    imb = (imd[i]-backd)[limsd[0]:limsd[1], limsd[2]:limsd[3]]
    g1, pp, tt = get_points(imb, tol_f=0.008, tol_ga=0.3)    
    points = pick_points_on_figure(pp[:,0], pp[:,1], imb, i)
    picked_points_d[i] = points
    
print()
print(picked_points_u)
print()
print(picked_points_d)


#%%

# Check whether grid indices are properly being calculated

knownsu = {0: [1353, 1351, 1327], 1: [1355, 1353, 1329], 2: [1359, 1357, 1332], 3: [1362, 1363, 1336], 4: [1362, 1363, 1337], 5: [1367, 1366, 1341], 6: [1369, 1367, 1342], 
           7: [1372, 1370, 1346], 8: [1374, 1373, 1347], 9: [1376, 1375, 1350], 10: [1378, 1376, 1351], 11: [1379, 1377, 1352], 12: [1381, 1379, 1355], 13: [1382, 1380, 1355], 
           14: [1385, 1383, 1359]}

knownsd = {0: [38, 39, 12], 1: [40, 41, 14], 2: [42, 43, 16], 3: [45, 46, 19], 4: [47, 48, 21], 5: [50, 51, 24], 6: [52, 53, 26], 7: [52, 53, 26], 8: [53, 54, 27], 
           9: [56, 57, 30], 10: [59, 60, 33], 11: [60, 61, 34], 12: [63, 64, 37], 13: [65, 66, 39], 14: [68, 69, 42]}

# i = 0
for i in range(N_positions-1, -1, -1):
# for i in [0]:
    t1 = time()

    # imb = (imd[i]-backd)[limsd[0]:limsd[1], limsd[2]:limsd[3]]
    # known = knownsd[i]
    
    imb = (imu[i]-backu)[limsu[0]:limsu[1], limsu[2]:limsu[3]]
    known = knownsu[i]

    g1, pp, tt = get_points(imb, tol_f=0.006, tol_ga=0.2)
    out, gridpos = find_indices(pp, known, eps=0.25, lim_ind=[0, 25, -1e5, 54], gps=[1,1])
    gridpos[:,1] = gridpos[:,1] - 5 

    t2 = time()
    print(t2-t1, end=' ')

    plt.figure()
    plt.gca().invert_yaxis()
    # plt.imshow( g1, cmap='gray' )

    plt.plot(pp[:, 0], pp[:, 1], 'r.', markersize=10)
    plt.plot(out[:, 0], out[:, 1], 'b.', markersize=8)
    # plt.plot( pp[known,0], pp[known,1], '.' )
    for j in range(len(out)):
        texto = "({:.0f},{:.0f})".format(gridpos[j, 0], gridpos[j, 1])
        plt.text(out[j, 0], out[j, 1], texto, fontsize=9)
    plt.title(i)
    plt.grid()
    plt.show()


# %%
t1 = time()
knownsu = {0: [1353, 1351, 1327], 1: [1355, 1353, 1329], 2: [1359, 1357, 1332], 3: [1362, 1363, 1336], 4: [1362, 1363, 1337], 5: [1367, 1366, 1341], 6: [1369, 1367, 1342], 
           7: [1372, 1370, 1346], 8: [1374, 1373, 1347], 9: [1376, 1375, 1350], 10: [1378, 1376, 1351], 11: [1379, 1377, 1352], 12: [1381, 1379, 1355], 13: [1382, 1380, 1355], 
           14: [1385, 1383, 1359]}

knownsd = {0: [38, 39, 12], 1: [40, 41, 14], 2: [42, 43, 16], 3: [45, 46, 19], 4: [47, 48, 21], 5: [50, 51, 24], 6: [52, 53, 26], 7: [52, 53, 26], 8: [53, 54, 27], 
           9: [56, 57, 30], 10: [59, 60, 33], 11: [60, 61, 34], 12: [63, 64, 37], 13: [65, 66, 39], 14: [68, 69, 42]}


# Find all grid indices 
pgridd, pposd = [], []
pgridu, pposu = [], []
for i in tqdm(range(N_positions)):
    imb = (imd[i]-backd)[limsd[0]:limsd[1], limsd[2]:limsd[3]]
    g1, pp, tt = get_points(imb, tol_f=0.008, tol_ga=0.3)
    known = knownsd[i]
    outd, gridposd = find_indices( pp, known, eps=0.25, lim_ind=[0, 25, -1e5, 49], gps=[1,1] )
    gridposd[:,1] = gridposd[:,1] - 5 # only for this experiment
    
    pposd.append(outd + [float(limsd[2] or 0), float(limsd[0] or 0)] )  # add pixels removed
    pgridd.append(np.hstack((gridposd, np.zeros((len(gridposd), 1))+(N_positions-1)-i)))

for i in tqdm(range(N_positions)):
    imb = (imu[i]-backu)[limsu[0]:limsu[1], limsu[2]:limsu[3]]
    g1, pp, tt = get_points(imb, tol_f=0.006, tol_ga=0.2)
    known = knownsu[i]
    outu, gridposu = find_indices( pp, known, eps=0.25, lim_ind=[0, 25, -1e5, 54], gps=[1,1] )
    pposu.append(outu + [float(limsu[2] or 0), float(limsu[0] or 0)] )  # add pixels removed
    gridposu[:,1] = gridposu[:,1] - 5 # only for this experiment
    
    pgridu.append(np.hstack((gridposu, np.zeros((len(gridposu), 1))+(N_positions-1)-i)))

t2 = time()
print(t2-t1)
# %%

# Fit the calibration parameters
d_pos = np.vstack([pgridd[i] * [3, 6, 10] for i in range(N_positions)])
u_pos = np.vstack([pgridu[i] * [3, 6, 10] for i in range(N_positions)])

d_pix = np.vstack([pposd[i] for i in range(N_positions)])
u_pix = np.vstack([pposu[i] for i in range(N_positions)])

mw = (-7.5+3)/(294+390)
angle_xy = np.arctan(np.abs(mw))
wall_dist = np.cos(angle_xy) * -7.5 + np.sin(angle_xy) * 294
angle_yz = 0.00639389483533472
angle_xz = 0 * np.pi/180 # 0.04726 * np.pi/180 

d_pos_r = rotate_coords(d_pos, [0, 1], angle_xy)
u_pos_r = rotate_coords(u_pos, [0, 1], angle_xy)
d_pos_r = rotate_coords(d_pos_r, [1, 2], angle_yz)
u_pos_r = rotate_coords(u_pos_r, [1, 2], angle_yz)
d_pos_r = rotate_coords(d_pos_r, [0, 2], angle_xz)
u_pos_r = rotate_coords(u_pos_r, [0, 2], angle_xz)
#small angles, so order of rotation doesn't matter too much

print(np.shape(d_pos), np.shape(d_pix))

cutoff_d = [0] + list(np.where(np.diff(d_pos[:, 2]))[0] + 1) + [len(d_pos)]
# lab_d, Xb_d, Yb_d = calibration_fit(d_pos, d_pix, cal_check=True)
lab_d, Xb_d, Yb_d = calibration_fit(d_pos_r, d_pix, cal_check=True)

print(np.shape(u_pos), np.shape(u_pix))

cutoff_u = [0] + list(np.where(np.diff(u_pos[:, 2]))[0] + 1) + [len(u_pos)]
# lab_u, Xb_u, Yb_u = calibration_fit(u_pos, u_pix, cal_check=True)
lab_u, Xb_u, Yb_u = calibration_fit(u_pos_r, u_pix, cal_check=True)

#%%

# Check if angles orientation is correct
i = 13
# ini,fin = cutoff_d[i], cutoff_d[i+1]
ini,fin = cutoff_d[0], cutoff_d[-1]

plt.figure()
plt.plot(  d_pos[ini:fin,0],   d_pos[ini:fin,1], '.' )
plt.plot(d_pos_r[ini:fin,0], d_pos_r[ini:fin,1], '.' )
plt.grid()
plt.show()

# plt.figure()
# plt.imshow(imd[i])
# plt.show()

#%%

# Visual checks to see if the calibration worked
came = 'd'
if came == 'd':
    cutoff, lab, pix_pos, Xb, Yb = cutoff_d, lab_d, d_pix, Xb_d, Yb_d
    # real_pos = d_pos
    real_pos = d_pos_r
    # ims = [imd[i][:,1200:2300] for i in range(len(imd))]
    ims = [imd[i][:, :] for i in range(len(imd))]
elif came == 'u':
    cutoff, lab, pix_pos, Xb, Yb = cutoff_u, lab_u, u_pix, Xb_u, Yb_u
    # real_pos = u_pos
    real_pos = u_pos_r
    # ims = [imu[i][130:,1200:2300] for i in range(len(imd))]
    ims = [imu[i][:, :] for i in range(len(imd))]

# for i in range(14):
for i in [0, N_positions-1]:
    posi = real_pos[cutoff[i]:cutoff[i+1]]
    pixi = pix_pos[cutoff[i]:cutoff[i+1]]
    px, py = pixi.T
    xp, yp = cal_direct(lab.x, posi, '2b')

    plt.figure()
    plt.imshow(ims[i], cmap='gray')
    # plt.gca().invert_yaxis()
    plt.plot(px, py, '.')
    plt.plot(xp, yp, '.')
    plt.title(i)
    plt.show()

# for i in range(14):
for i in [0, N_positions-1]:
    ini, fin = cutoff[i], cutoff[i+1]

    plt.figure()
    plt.plot(real_pos[ini:fin, 0], real_pos[ini:fin, 1], '.')
    plt.plot(Xb[ini:fin], Yb[ini:fin], '.')
    plt.title(i)
    plt.show()


#%%

# Histograms to see difference between dots positions and calibrated ones
# dists_d = np.sqrt((d_pos[:, 0] - Xb_d)**2 + (d_pos[:, 1] - Yb_d)**2)
# dists_u = np.sqrt((u_pos[:, 0] - Xb_u)**2 + (u_pos[:, 1] - Yb_u)**2)
dists_d = np.sqrt( (d_pos_r[:,0] - Xb_d)**2 + (d_pos_r[:,1] - Yb_d)**2 )
dists_u = np.sqrt( (u_pos_r[:,0] - Xb_u)**2 + (u_pos_r[:,1] - Yb_u)**2 )

plt.figure()
plt.hist(dists_d, bins=100, density=True, alpha=0.5, label='Down camera')
plt.hist(dists_u, bins=100, density=True, alpha=0.5, label='Top camera')
plt.hist(np.hstack((dists_d, dists_u)), bins=100,
         density=True, alpha=0.5, label='Both cameras')
plt.xlabel('Distance (mm)')
plt.legend()
plt.show()


#%%

# Export calibration amgles and wall distance
cal_dat = [lab_u.x, lab_d.x, wall_dist, angle_xy, angle_yz, angle_xz]
np.savez(path+'calibration_data.npz', *cal_dat)

data = np.load(path+'calibration_data.npz')
lu,ld, wd, axy, ayz, axz = data['arr_0'], data['arr_1'], float(data['arr_2']), float(data['arr_3']), float(data['arr_4']), float(data['arr_5'])

lu ,ld, wd, axy, ayz, axz









#%%







#%%



