#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:24:38 2026

@author: tomasferreyrahauchar
"""

import glob
import tifffile
import warnings

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time
from itertools import combinations

from scipy import ndimage as ndi
from scipy.signal import find_peaks, fftconvolve
from scipy.spatial import cKDTree
from scipy.optimize import least_squares


from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm



def two_lowest_minima(arr, distance=10):
    results = np.full((arr.shape[0], 2), np.nan)
    for i, row in enumerate(arr):
        minima, _ = find_peaks(-row, distance=distance)

        if len(minima) > 0:
            lowest = np.sort( minima[np.argsort(row[minima])[:2]] )
            results[i, :len(lowest)] = lowest

    return results

# def make_mask(mim, lims, distance=10, sigma=10):
#     gim = gaussian(mim, sigma)
#     ig_f = np.gradient(gim, axis=1)
    
#     indices = two_lowest_minima(ig_f[:,lims[0]:lims[1]], distance=distance) + lims[0]
    
#     izq,der = int(np.min(indices[:,0])), int(np.max(indices[:,1]))+6
    
#     mask = np.zeros(mim.shape, dtype=bool)
    
#     for i, (idx1, idx2) in enumerate(indices):
#         if not np.isnan(idx1) and not np.isnan(idx2):
#             start, end = sorted([int(idx1), int(idx2)])
#             mask[i, start:end+1] = True
    
#     return mask, np.s_[:,izq:der]

def make_mask(mim, lims, distance=10, sigma=10):

    gim = gaussian(mim, sigma)
    ig_f = np.gradient(gim, axis=1)


    indices = two_lowest_minima( ig_f[:, lims[0]:lims[1]], distance=distance ) + lims[0]

    indices_sub = indices.astype(float)

    for i, (idx1, idx2) in enumerate(indices):
        for j, idx in enumerate((idx1, idx2)):
            x = int(idx)

            if np.isnan(idx): 
                indices_sub[i, j] = np.nan
                continue
            if x <= 0 or x >= ig_f.shape[1] - 1:
                continue

            # Three neighbouring gradient values
            g1 = ig_f[i, x - 1]
            g2 = ig_f[i, x]
            g3 = ig_f[i, x + 1]

            denominator = g1 - 2*g2 + g3
            if denominator != 0:
                delta = 0.5 * (g1 - g3) / denominator
                # Avoid unrealistic sub-pixel corrections
                if abs(delta) <= 1:
                    indices_sub[i, j] = x + delta

    # Left and right limits for each row
    left_limits = np.nanmin(indices_sub, axis=1)
    right_limits = np.nanmax(indices_sub, axis=1)

    # Integer limits for the mask
    izq = int(np.floor(np.nanmin(left_limits)))
    der = int(np.ceil(np.nanmax(right_limits))) + 6

    mask = np.zeros(mim.shape, dtype=bool)

    for i, (idx1, idx2) in enumerate(indices):
        if not np.isnan(idx1) and not np.isnan(idx2):

            start, end = sorted([int(idx1), int(idx2)])
            mask[i, start:end+1] = True

    return mask, np.s_[:, izq:der], left_limits, right_limits


# def make_mask_minmax(mim, lims, sigma=10):
#     gim = gaussian(mim, sigma)
#     ig_f = np.gradient(gim, axis=1)
    
#     ama = np.argmax( ig_f[:,lims[0]:lims[1]], axis=1 ) + lims[0]
#     ami = np.argmin( ig_f[:,lims[0]:lims[1]], axis=1 ) + lims[0]
    
#     izq,der = np.min(ama)-2, np.max(ami)+6
    
#     mask = np.zeros(mim.shape, dtype=bool)
        
#     for i in range(len(ami)):
#         idx1, idx2 = ama[i], ami[i]
#         if not np.isnan(idx1) and not np.isnan(idx2):
#             start, end = sorted([int(idx1), int(idx2)])
#             mask[i, start:end+1] = True
    
#     return mask, np.s_[:,izq:der]

def make_mask_minmax(mim, lims, sigma=10):

    gim = gaussian(mim, sigma)
    ig_f = np.gradient(gim, axis=1)

    ama = np.argmax(ig_f[:, lims[0]:lims[1]], axis=1) + lims[0]
    ami = np.argmin(ig_f[:, lims[0]:lims[1]], axis=1) + lims[0]

    ama_sub = ama.astype(float)
    ami_sub = ami.astype(float)

    for i in range(len(ama)):
        for idx, refined, is_max in [ (ama[i], ama_sub, True), (ami[i], ami_sub, False) ]:
            x = int(idx)

            if x <= 0 or x >= ig_f.shape[1] - 1:
                continue

            g1 = ig_f[i, x - 1]
            g2 = ig_f[i, x]
            g3 = ig_f[i, x + 1]

            denominator = g1 - 2*g2 + g3
            if denominator != 0:
                delta = 0.5 * (g1 - g3) / denominator

                if abs(delta) <= 1:
                    refined[i] = x + delta

    # Sub-pixel limits
    left_limits = np.minimum(ama_sub, ami_sub)
    right_limits = np.maximum(ama_sub, ami_sub)

    # Integer limits for slicing
    izq = int(np.floor(np.min(ama_sub))) - 2
    der = int(np.ceil(np.max(ami_sub))) + 6

    mask = np.zeros(mim.shape, dtype=bool)

    for i in range(len(ami)):
        idx1, idx2 = ama[i], ami[i]
        if not np.isnan(idx1) and not np.isnan(idx2):
            start, end = sorted([int(idx1), int(idx2)])
            mask[i, start:end+1] = True

    return mask, np.s_[:, izq:der], left_limits, right_limits



def cross_correlation_peak(img1, img2):
    """
    Calculate cross-correlation and return peak displacement (dy, dx).
    """

    img1 = np.asarray(img1, dtype=float)
    img2 = np.asarray(img2, dtype=float)

    # Remove mean
    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)

    # Cross-correlation
    corr = fftconvolve(img1, img2[::-1, ::-1], mode='full')

    # Find peak
    peak = np.unravel_index(np.argmax(corr), corr.shape)

    # Convert peak position to displacement
    dy = peak[0] - (img2.shape[0] - 1)
    dx = peak[1] - (img2.shape[1] - 1)

    return dy, dx, corr


def local_cross_correlation(img1, img2, n_windows_x=20, n_windows_y=20, overlap=0.5):

    h, w = img1.shape

    # Calculate window dimensions
    win_x = int(w / (n_windows_x - (n_windows_x - 1) * overlap))
    win_y = int(h / (n_windows_y - (n_windows_y - 1) * overlap))

    # Calculate step sizes
    step_x = int(win_x * (1 - overlap))
    step_y = int(win_y * (1 - overlap))

    n_cols = (w - win_x) // step_x + 1
    n_rows = (h - win_y) // step_y + 1

    positions = []
    displacements = []

    for y in range(0, h - win_y + 1, step_y):
        for x in range(0, w - win_x + 1, step_x):

            a = img1[y:y+win_y, x:x+win_x]
            b = img2[y:y+win_y, x:x+win_x]

            # Remove mean
            a = a - np.mean(a)
            b = b - np.mean(b)

            # Cross-correlation
            corr = fftconvolve(a, b[::-1, ::-1], mode='full')

            peak = np.unravel_index(np.argmax(corr), corr.shape)

            dy = peak[0] - (win_y - 1)
            dx = peak[1] - (win_x - 1)

            positions.append((y + win_y//2, x + win_x//2))
            displacements.append((dy, dx))

    return np.array(positions), np.array(displacements), (n_rows, n_cols), (win_y, win_x)


def remove_displacement_outliers(dis, grid_shape, threshold=2):    
    n_rows, n_cols = grid_shape

    grid = dis.reshape(n_rows, n_cols, 2).copy()

    magnitude = np.linalg.norm(grid, axis=2)
    median_mag = np.median(magnitude)

    outliers = magnitude > threshold * median_mag

    filtered = grid.copy()

    for i, j in zip(*np.where(outliers)):

        neighbours = []

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

            ni, nj = i + di, j + dj

            if (0 <= ni < n_rows and
                0 <= nj < n_cols and
                not outliers[ni, nj]):

                neighbours.append(grid[ni, nj])

        if neighbours:
            filtered[i, j] = np.mean(neighbours, axis=0)

    return filtered.reshape(-1, 2)


def subpixel_particle_positions(img, labeled):
    # Integer maximum positions
    peaks = np.array(ndi.maximum_position( img, labels=labeled, index=np.arange(1,np.max(labeled)+1) )) 
    positions = peaks.astype(float)

    for k, (y, x) in enumerate(peaks):
        if x == 0 or x == img.shape[1]-1:
            continue

        if y == 0 or y == img.shape[0]-1:
            continue

        Ix = img[y, x-1:x+2]
        Iy = img[y-1:y+2, x]

        # Gaussian interpolation
        if np.all(Ix > 0):
            lx = np.log(Ix)
            denom = 2 * (lx[0] - 2*lx[1] + lx[2])
            if denom != 0:
                positions[k, 1] += (lx[0] - lx[2]) / denom

        if np.all(Iy > 0):
            ly = np.log(Iy)
            denom = 2 * (ly[0] - 2*ly[1] + ly[2])
            if denom != 0:
                positions[k, 0] += (ly[0] - ly[2]) / denom

    return positions


def match_particles(pos1, pos2, positions, displacements, radius=5):

    tree_grid = cKDTree(positions)
    tree_pos2 = cKDTree(pos2)

    # Find closest displacement-grid position
    _, idx = tree_grid.query(pos1)

    # Predict positions in image 2
    predicted = pos1 + displacements[idx]

    # Find closest particle in image 2
    distances, matches = tree_pos2.query(predicted)

    # Reject matches outside radius
    valid = distances <= radius

    # Resolve duplicate matches: keep closest predictor
    candidates = np.where(valid)[0]

    # Sort by prediction distance (closest first)
    candidates = candidates[np.argsort(distances[candidates])]

    # Keep only the first occurrence of each pos2 match
    _, unique_idx = np.unique( matches[candidates], return_index=True )
    selected = candidates[unique_idx]

    # Sort by original pos1 index
    selected = np.sort(selected)

    matched_pos1 = selected
    matched_pos2 = matches[selected]

    return matched_pos1, matched_pos2, distances[selected], predicted[selected]

def find_displacements(im1, im2, imb1, imb2, threshold=2, n_windows_x=5, n_windows_y=15, overlap=0.5, radius=10, min_size=15):
    """
    threshold: how much bigger the magnitude of cross-correlation peak to filter and replace with mean
    n_windows_x: number window in horizontal direction (cross-correlation)
    n_windows_y: number window in vertical direction (cross-correlation)
    overlap: overlap percentage of windows (cross-correlation)
    
    radius: radius to search for particle match from prediction
    min_size: size of particle to remove
    """
    pos, dis, grid_shape, window = local_cross_correlation(im1, im2, n_windows_x, n_windows_y, overlap)
    dis = remove_displacement_outliers(dis, grid_shape, threshold=2)

    rim1 = remove_small_objects(imb1, min_size=15)
    rim2 = remove_small_objects(imb2, min_size=15)

    lim1, lim2 = label(imb1 - rim1*1), label(imb2 - rim2*1)

    pos1 = subpixel_particle_positions(im1, lim1)
    pos2 = subpixel_particle_positions(im2, lim2)

    mpa1, mpa2, dists, preds = match_particles(pos1, pos2, pos, -dis, radius)
    displ = pos2[mpa2] - pos1[mpa1]

    return pos1[mpa1], displ


def displacement_statistics(pos, dis, mascara, window_size=32):

    if np.isscalar(window_size):
        win_y = win_x = int(window_size)
    else:
        win_y, win_x = window_size

    h, w = mascara.shape

    n_rows = h // win_y
    n_cols = w // win_x

    # Assign particles to windows
    iy = (pos[:, 0] // win_y).astype(int)
    ix = (pos[:, 1] // win_x).astype(int)

    # Valid particles
    valid = (
        (iy >= 0) & (iy < n_rows) &
        (ix >= 0) & (ix < n_cols)
    )

    valid &= mascara[
        np.clip(pos[:, 0].astype(int), 0, h-1),
        np.clip(pos[:, 1].astype(int), 0, w-1)
    ]

    iy = iy[valid]
    ix = ix[valid]
    d = dis[valid]

    mag = np.linalg.norm(d, axis=1)

    shape = (n_rows, n_cols)

    results = {
        'magnitude_mean': np.full(shape, np.nan),
        'magnitude_std': np.full(shape, np.nan),
        'dx_mean': np.full(shape, np.nan),
        'dx_std': np.full(shape, np.nan),
        'dy_mean': np.full(shape, np.nan),
        'dy_std': np.full(shape, np.nan),
        'counts': np.zeros(shape, dtype=int)
    }

    # Group particles by window
    for y in range(n_rows):
        for x in range(n_cols):

            selected = (iy == y) & (ix == x)

            if not np.any(selected):
                continue

            results['counts'][y, x] = np.sum(selected)

            results['magnitude_mean'][y, x] = np.mean(mag[selected])
            results['magnitude_std'][y, x] = np.std(mag[selected])

            results['dx_mean'][y, x] = np.mean(d[selected, 1])
            results['dx_std'][y, x] = np.std(d[selected, 1])

            results['dy_mean'][y, x] = np.mean(d[selected, 0])
            results['dy_std'][y, x] = np.std(d[selected, 0])

    results['grid_shape'] = shape

    return results


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
    gps : [±1,±1]
        Direction the first and/or second point (for starting grid) is going
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


def rotate_coords(points, angle):
    """
    Parameters
    ----------
    points : (N,3) array
        points to rotate.
    angle : float
        angle in radians.

    Returns
    -------
    points_r : (N,3) array
        array of points rotated.
    """
    c1, c2 = points[:, 0], points[:, 1]
    c1r = np.cos(angle) * c1 - np.sin(angle) * c2
    c2r = np.sin(angle) * c1 + np.cos(angle) * c2

    points_r = np.zeros_like(points)*1.
    points_r[:, 0], points_r[:, 1] = c1r, c2r
    return points_r


def calib_func_inv(X, Y, v):
    a, b, c, d, e, f, g, h = v
    den = b*d - a*e + e*g*X - d*h*X - b*g*Y + a*h*Y

    x = (c*e - b*f - e*X + f*h*X + b*Y - c*h*Y) / den
    y = (d*X - f*g*X - a*(Y-f) - c*(d-g*Y)) / den
    return x, y

def calib_fun2_inv(X, Y, v, initial_guess=None):
    X = np.atleast_1d(X).astype(float)
    Y = np.atleast_1d(Y).astype(float)

    n = len(X)
    if initial_guess is None:
        guesses = np.zeros((n, 2))

    else:
        guesses = np.asarray(initial_guess, dtype=float)

        if guesses.shape == (2,):
            guesses = np.tile(guesses, (n, 1))
        elif guesses.shape != (n, 2):
            raise ValueError( "initial_guess must have shape (2,) or (N, 2)." )

    def residual(p, Xi, Yi):
        x, y = p
        Xpred, Ypred = calib_fun2(x, y, v)
        return [Xpred - Xi, Ypred - Yi]

    result = np.array([ least_squares( residual, guesses[i], args=(X[i], Y[i]) ).x for i in range(n) ])

    x = result[:, 0]
    y = result[:, 1]
    if np.ndim(X) == 0 or len(x) == 1:
        return x[0], y[0]
    return x, y



def calib_func(x, y, v):
    xtop = v[0]*x + v[1]*y + v[2]
    ytop = v[3]*x + v[4]*y + v[5]
    bot = v[6]*x + v[7]*y + + 1 

    X, Y = xtop / bot, ytop / bot
    return X, Y

def calib_fun2(x, y, v):
    xtop = v[0]*x + v[1]*y + v[2] + v[8]*x**2 + v[9]*y**2 + v[10]*x*y
    ytop = v[3]*x + v[4]*y + v[5] + v[11]*x**2 + v[12]*y**2 + v[13]*x*y
    bot = v[6]*x + v[7]*y + + 1 + v[14]*x**2 + v[15]*y**2 + v[16]*x*y

    X, Y = xtop / bot, ytop / bot
    return X, Y

def calibration_fit(real_pos, pix_pos, cal_check=False, p0=[0, 1, 0, 1, 0, 1, 0, 0]):
    """
    Parameters
    ----------
    real_pos : (N,3) array
        point positions in real world.
    pix_pos : (N,2) array
        pixel position in image.
    cal_check: bool
        if True, also returns (x,y) real world coordinates of image points, by inverting the calibration.
        Useful for checking if calibration worked.
    Returns
    -------
    dict
        results from least squares.
    """
    t1 = time()

    def residual_firstz(v):
        xrrs, yrrs = real_pos.T
        spxs, spys = pix_pos.T
        xtop = v[0]*spxs + v[1]*spys + v[2]
        ytop = v[3]*spxs + v[4]*spys + v[5]
        bot = v[6]*spxs + v[7]*spys + + 1 

        X, Y = xtop / bot, ytop / bot
        return (X - xrrs)**2 + (Y - yrrs)**2

    def residual_sec(v):
        xrrs, yrrs = real_pos.T
        spxs, spys = pix_pos.T
        xtop = v[0]*spxs + v[1]*spys + v[2] + v[8]*spxs**2 + v[9]*spys**2 + v[10]*spxs*spys
        ytop = v[3]*spxs + v[4]*spys + v[5] + v[11]*spxs**2 + v[12]*spys**2 + v[13]*spxs*spys
        bot = v[6]*spxs + v[7]*spys + + 1 + v[14]*spxs**2 + v[15]*spys**2 + v[16]*spxs*spys

        X, Y = xtop / bot, ytop / bot
        return (X - xrrs)**2 + (Y - yrrs)**2

    lal = least_squares(residual_firstz, p0, method='lm', xtol=1e-12, max_nfev=10000)
    p02 = np.zeros(17)
    p02[:8] = lal.x
    la2 = least_squares(residual_sec, p02, method='lm', xtol=1e-12, max_nfev=10000)

    t2 = time()
    print(t2-t1)

    return lal, la2

def distance( preal, precov ):
    return np.sqrt( (preal[:,0] - precov[:,0])**2 + (preal[:,1] - precov[:,1])**2  ) 




#%%
# f0
exp = 'f0'
main_folder = '/Volumes/PTV channel/'+exp
files_folder = '/Volumes/PTV channel/'+exp+'/'+exp+'_C001H*' 
folders_path = np.sort( glob.glob( files_folder ) )

cals_folder = '/Volumes/PTV channel/'+exp+'/*al*'
cal_path = np.sort( glob.glob( cals_folder ) )

#%%
# =============================================================================
# Calibration
# =============================================================================

cals = np.sort( glob.glob( cal_path[0] + '/*.tif'   ) )

calim = np.zeros((1024,1280))
num = 0
for i in tqdm(range(100)):
    img1 = tifffile.imread(cals[i])
    calim += img1; num += 1
calim = calim/num


if exp in ['f6','f0']:
    threshold = 0.8
    lims = [10,1010, 560,820]
    m1, m2 = 10, 20
    lim_ind=[-25, 0, -1e5, 44]

elif exp == 'f5':
    threshold = 0.8
    lims = [10,None, 807,1080]
    m1, m2 = 14, 50
    

lls = np.s_[lims[0]:lims[1], lims[2]:lims[3] ]

lcal = calim[lls]
gcal = gaussian( lcal, 10 )
bcal = lcal / gcal < threshold

re1 = remove_small_objects(bcal, min_size=m1)
re2 = remove_small_objects(bcal, min_size=m2)
dots = label(re1 - re2*1) 

props = regionprops(dots, -lcal / gcal)
cy = np.array([(props[i].centroid_weighted)[0] for i in range(len(props))]) #+ lims[0]
cx = np.array([(props[i].centroid_weighted)[1] for i in range(len(props))]) #+ lims[2]
pp = np.vstack((cx, cy)).T


known = pick_points_on_figure(cx, cy, lcal / gcal, 0)
print( known )

pp = pp + [lims[2],lims[0]]
out, gridpos = find_indices(pp, known, eps=0.6, lim_ind=[-25, 0, -1e5, 44], gps=[-1,1])
gridpos[:,1] = gridpos[:,1] - 0
gridpos[:,0] = gridpos[:,0] + 25

pos = gridpos * [3, 6]

if exp in ['f6','f0']:
    mw = (3.5-8) / (366+318) 
    angle_xy = np.arctan(mw)
    wall_dist = np.cos(angle_xy) * (-3.5) + np.sin(angle_xy) * 366

elif exp == 'f5':
    mw = (3.5-7) / (366+318) 
    angle_xy = np.arctan(mw)
    wall_dist = np.cos(angle_xy) * (-3.5) + np.sin(angle_xy) * 366

rpos = rotate_coords(pos, - angle_xy)

lal, la2 = calibration_fit(rpos, out, p0=[0.3,0,-200,0,-.2,50,0,0])

print( wall_dist  )
plt.close('all')

plt.figure()
plt.gca().invert_yaxis()

plt.plot(pp[:, 0], pp[:, 1], 'r.', markersize=10)
plt.plot(out[:, 0], out[:, 1], 'b.', markersize=8)
# plt.plot( pp[known,0], pp[known,1], '.' )
for j in range(len(out)):
    texto = "({:.0f},{:.0f})".format(gridpos[j, 0], gridpos[j, 1])
    # texto = "({:.0f},{:.0f})".format(pos[j, 0], pos[j, 1])

    plt.text(out[j, 0], out[j, 1], texto, fontsize=9)

plt.title(i)
plt.grid()
plt.show()

angle = - angle_xy
plt.figure()
plt.plot( pos[:,0], pos[:,1], 'r.' )
plt.plot( rpos[:,0], rpos[:,1], 'g.' )

xt = np.linspace(-8,-3.5,100)
yt = (366+318) / (-3.5+8) * (xt+3.5) + 366
plt.plot( xt, yt, 'b-'  )
plt.plot( np.cos(angle) * xt - np.sin(angle) * yt, np.sin(angle) * xt + np.cos(angle) * yt, 'm-'  )


plt.grid()
# plt.axis('equal')
plt.show()


plt.figure()
plt.plot( rpos[:,0], rpos[:,1], '.')
X, Y = calib_func(out[:,0], out[:,1], lal.x)
realp1 = np.vstack((X, Y)).T
plt.plot( X, Y, '.')

X, Y = calib_fun2(out[:,0], out[:,1], la2.x)
realp2 = np.vstack((X, Y)).T
plt.plot( X, Y, '.')
# plt.axis('equal')
plt.show()


dist1 = distance(rpos, realp1)
dist2 = distance(rpos, realp2)
plt.figure()
plt.hist(dist1, bins=40, alpha=0.5)
plt.hist(dist2, bins=40, alpha=0.5)
plt.show()

#%%

# files = []
# for i in range(len(folders_path)):
#     afiles = np.sort( glob.glob( folders_path[i] + '/' +exp+'_C*.tif' ) )
#     files.append(afiles)

# files = np.concatenate(files)


xpo, ypo = rpos[:,0], rpos[:,1]
xpi, ypi = calib_fun2_inv(xpo, ypo, la2.x)

if exp == 'f5':
    x_vals = np.arange(0, 123 + 3, 3)
    y_vals = np.arange(102, -72 - 6, -6)
else:
    x_vals = np.arange(0, 123 + 3, 3)
    y_vals = np.arange(54, -126 - 6, -6)

xxx, yyy = np.meshgrid(x_vals, y_vals)
npos = np.column_stack((xxx.ravel(), yyy.ravel()))
# npos = rotate_coords(npos, - angle_xy)

nxpi, nypi = calib_fun2_inv(npos[:,0], npos[:,1], la2.x)

# pn = 7
# nxp = [90]*pn + [100]*pn + [110]*pn
# nyp = [50,25,0,-25,-50,-75, -100] * 3
# nxpi, nypi = calib_fun2_inv(nxp, nyp, la2.x)


plt.figure()
plt.imshow( calim, cmap='gray' )

plt.plot( xpi, ypi, '.')
plt.plot( pp[:,0], pp[:,1], '.')

plt.plot( nxpi, nypi, '.')
plt.show()

# i = 0
# im1 = tifffile.imread(files[i])



# plt.figure()
# plt.imshow( im1, vmax=150 )
# # plt.imshow( calim, cmap='gray', alpha=0.3 )

# plt.plot( xpi, ypi, '.')
# plt.plot( pp[:,0], pp[:,1], '.')

# plt.plot( nxpi, nypi, 'r.')
# plt.show()

#%%
# =============================================================================
# Analysis
# =============================================================================

files = []
for i in range(len(folders_path)):
    afiles = np.sort( glob.glob( folders_path[i] + '/' +exp+'_C*.tif' ) )
    files.append(afiles)

files = np.concatenate(files)

# get mask
t1 = time()

lims, sigma = [500,1000], 10  #f0 minmax 
# lims, sigma = [500,1000], 15 #f6 minmax 
# lims = [0, 1200] #test 4

threshold = 5

imm = np.zeros((1024,1280))
num = 0

li = range(0,500,2)
for i in tqdm(li):
    img1 = tifffile.imread(files[i])
    img2 = tifffile.imread(files[i+1])

    imm += img1
    num += 1

mim = imm/num


if exp == 'f0':
    lims, sigma = [500,1000], 10  
    mask, nlim, left_l, righ_l = make_mask_minmax(mim, lims, sigma=sigma)
elif exp == 'f6':
    lims, sigma = [500,1000], 15
    mask, nlim, left_l, righ_l  = make_mask_minmax(mim, lims, sigma=sigma)
elif exp == 'f5':
    lims, sigma = [800,1200], 16
    # mask, nlim, left_l, righ_l = make_mask(mim, lims, distance = 100, sigma=sigma)
    mask, nlim, left_l, righ_l = make_mask_minmax(mim, lims, sigma=sigma)

# mask, nlim = make_mask(mim, lims, distance = 100)
# mask, nlim = make_mask_minmax(mim, lims, sigma=10)

t2 = time()
print(t2-t1)



fig, ax = plt.subplots(1,2, layout='constrained')
ax[0].imshow( mask )
ax[1].imshow( mask[nlim] )

aa = np.arange(1024)
ax[0].plot( left_l, aa, '-' )
ax[0].plot( righ_l, aa, '-' )
plt.show()

#%%

# gim = gaussian(mim, 18)
# ig_f = np.gradient(gim, axis=1)

# aa = np.arange(1024)
# ama = np.argmax( ig_f[:,lims[0]:lims[1]], axis=1 ) + lims[0]
# ami = np.argmin( ig_f[:,lims[0]:lims[1]], axis=1 ) + lims[0]

# # gim = gaussian(mim, 10)
# # ig_f = np.gradient(gim, axis=1)

# # indices = two_lowest_minima(ig_f[:,lims[0]:lims[1]], distance=100) + lims[0]
# # ama, ami = indices[:,0], indices[:,1]

# # izq,der = int(np.min(indices[:,0])), int(np.max(indices[:,1]))+6

# # mask = np.zeros(mim.shape, dtype=bool)

# plt.figure()
# plt.imshow( mim , vmax=250 )
# # plt.imshow( gim, vmax=100 )
# # plt.imshow( ig_f[:,750:1200] )

# # plt.plot( ig_f[60,500:1000] )
# plt.plot( ama, aa, 'r-' )
# plt.plot( ami, aa, 'r-' )

# plt.show()

#%%

nnmask = np.zeros_like(calim)
nnmask[nlim] = mask

plt.figure()

plt.imshow(calim, cmap='gray')
plt.imshow( mim , vmax=250, alpha=0.2 )

# plt.imshow( mim , vmax=250, alpha=1. )
# plt.imshow( nnmask , alpha=0.5, cmap='viridis' )

plt.show()

#%%
run = 1
save = 0

show_bar = 1
show_plot = 0

threshold = 4
threshold_d = 1.5 
n_windows_x = 9  #7 for f5, 9 for f6
n_windows_y = 14 #12 for f5, 14 for f6
overlap = 0.5
radius = 10
min_size = 15


if run:
    
    all_f = range(0,len(files),2)
    # all_f = range(0,2000,2)
    
    all_pos, all_dis = [], []
    for i in tqdm(all_f, disable= not show_bar):
    
        img1 = tifffile.imread(files[i])
        img2 = tifffile.imread(files[i+1])
    
        im1 = (img1/mim * mask)[nlim]
        im2 = (img2/mim * mask)[nlim]
        
        imb1 = im1 > threshold
        imb2 = im2 > threshold
    
        posi, displ = find_displacements(im1, im2, imb1, imb2, threshold_d, n_windows_x, n_windows_y, overlap, radius, min_size)
        
        all_pos.append(posi)
        all_dis.append(displ)
    
        if show_plot:
            plt.figure(  )
            plt.imshow(mask[nlim], alpha=0.3)
            plt.plot( [posi[:,1], (posi+displ)[:,1]], [posi[:,0], (posi+displ)[:,0]], 'g-'  )
            plt.scatter( posi[:,1], posi[:,0], s=2, c='r' )
            plt.show()
            
    all_posi = np.concatenate(all_pos) 
    all_pos, all_dis = np.concatenate(all_pos) + [0, nlim[1].start], np.concatenate(all_dis) 

    if save:    
        np.save( main_folder+'/pos.npy', all_posi )
        np.save( main_folder+'/dis.npy', all_dis )

else:
    all_posi = np.load(main_folder+'/pos.npy')
    all_pos = all_posi + [0, nlim[1].start]
    
    all_dis  = np.load(main_folder+'/dis.npy')
        

#%%

 

plt.figure(  )
# plt.imshow(mask[nlim], alpha=0.3)
plt.imshow(mask, alpha=0.3)


# plt.scatter( all_pos[::10,1], all_pos[::10,0], s=2, c='r' )

plt.scatter( all_pos[:200,1], all_pos[:200,0], s=2, c='r' )
plt.plot( [all_pos[:200,1], (all_pos+all_dis)[:200,1]], [all_pos[:200,0], (all_pos+all_dis)[:200,0]], 'g-'  )

plt.show()


left_rlx, left_rly = calib_fun2( left_l, np.arange(1024), la2.x)
righ_rlx, righ_rly = calib_fun2( righ_l, np.arange(1024), la2.x)


Xw, Yw = calib_fun2( all_pos[:,1], all_pos[:,0], la2.x)
Xw2, Yw2 = calib_fun2( all_pos[:,1] + all_dis[:,1], all_pos[:,0] + all_dis[:,0], la2.x)

all_rp = np.vstack([Xw,Yw]).T
all_rd = np.vstack([Xw2,Yw2]).T - all_rp

# Xw1, Yw1 = calib_func( all_pos[:,1], all_pos[:,0], lal.x)

plt.figure(  )
plt.plot( Xw[:200], Yw[:200], 'r.' )
plt.plot( [Xw[:200], Xw2[:200]], [Yw[:200], Yw2[:200]], 'g-' )

plt.plot( left_rlx, left_rly, 'b-' )
plt.plot( righ_rlx, righ_rly, 'b-' )

plt.axis('equal')
plt.show()


#%%



t1 = time()
stats = displacement_statistics(all_posi, all_dis, mask[nlim], window_size=10)
t2 = time()
t2-t1

#%%


fig, ax = plt.subplots(1,4, layout='constrained', figsize=(16,5))
# ax = ax.flatten()

ax[0].set_title('Counts')
im0 = ax[0].imshow( stats['counts'] ) 
fig.colorbar(im0, ax=ax[0])

ax[1].set_title('Magnitud')
im0 = ax[1].imshow( stats['magnitude_mean'] ) 
fig.colorbar(im0, ax=ax[1])

ax[2].set_title('Dx')
im0 = ax[2].imshow( stats['dx_mean'] ) 
fig.colorbar(im0, ax=ax[2])

ax[3].set_title('Dy')
im0 = ax[3].imshow( stats['dy_mean'] ) 
fig.colorbar(im0, ax=ax[3])

plt.show()


#%%


threshold = 2 
n_windows_x = 5 
n_windows_y = 15
overlap = 0.5
radius = 10
min_size = 15

posi, displ = find_displacements(im1, im2, imb1, imb2, threshold, n_windows_x, n_windows_y, overlap, radius, min_size)

plt.figure(  )

plt.imshow(mask[nlim], alpha=0.3)

plt.plot( [posi[:,1], (posi+displ)[:,1]], [posi[:,0], (posi+displ)[:,0]], 'g-'  )
plt.scatter( posi[:,1], posi[:,0], s=2, c='r' )
plt.show()

#%%
# =============================================================================
# In real values
# =============================================================================


def nangauss(altus, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):    
        V = altus.copy()
        V[np.isnan(altus)] = 0
        W = 0 * altus.copy() + 1
        W[np.isnan(altus)] = 0
        VV,WW = ndi.gaussian_filter(V, sigma), ndi.gaussian_filter(W, sigma)
        gdp = VV/WW
        gdp[np.isnan(altus)] = np.nan
        return gdp


def displacement_statistics_mm( all_pos, all_dis, window_size=1.0, lim_lx=None, lim_ly=None, lim_rx=None, lim_ry=None, dt = 0.006 ):

    # Window dimensions
    if np.isscalar(window_size):
        win_y = win_x = float(window_size)
    else:
        win_y, win_x = window_size

    # Determine vertical extent
    if lim_ly is not None and lim_ry is not None:
        y_min = max(np.min(lim_ly), np.min(lim_ry))
        y_max = min(np.max(lim_ly), np.max(lim_ry))
    else:
        y_min = np.min(all_pos[:, 0])
        y_max = np.max(all_pos[:, 0])
    
    # Sort boundaries by y-coordinate
    idx_l = np.argsort(lim_ly)
    idx_r = np.argsort(lim_ry)
    
    lim_ly_sorted = np.asarray(lim_ly)[idx_l]
    lim_lx_sorted = np.asarray(lim_lx)[idx_l]
    
    lim_ry_sorted = np.asarray(lim_ry)[idx_r]
    lim_rx_sorted = np.asarray(lim_rx)[idx_r]
    
    # Interpolate boundaries at particle y-coordinates
    left_x = np.interp( all_pos[:, 0], lim_ly_sorted, lim_lx_sorted, left=np.nan, right=np.nan )
    
    right_x = np.interp( all_pos[:, 0], lim_ry_sorted, lim_rx_sorted, left=np.nan, right=np.nan )
    
    # Select particles between the two curves
    valid = ( (all_pos[:, 1] >= left_x) &
              (all_pos[:, 1] <= right_x)  )

    pos = all_pos[valid]
    dis = all_dis[valid] / dt

    # Horizontal extent of the selected region
    if len(pos) == 0:
        raise ValueError("No particles found inside the selected region.")

    x_min = np.min(pos[:, 1])
    x_max = np.max(pos[:, 1])

    # Grid dimensions
    n_rows = int(np.ceil((y_max - y_min) / win_y))
    n_cols = int(np.ceil((x_max - x_min) / win_x))

    # Assign particles to windows
    iy = ((pos[:, 0] - y_min) // win_y).astype(int)
    ix = ((pos[:, 1] - x_min) // win_x).astype(int)

    # Handle particles exactly on the upper boundary
    iy = np.clip(iy, 0, n_rows - 1)
    ix = np.clip(ix, 0, n_cols - 1)

    # Displacement magnitude
    mag = np.linalg.norm(dis, axis=1)

    shape = (n_rows, n_cols)

    results = {
        'magnitude_mean': np.full(shape, np.nan),
        'magnitude_std': np.full(shape, np.nan),
        'dx_mean': np.full(shape, np.nan),
        'dx_std': np.full(shape, np.nan),
        'dy_mean': np.full(shape, np.nan),
        'dy_std': np.full(shape, np.nan),
        'counts': np.zeros(shape, dtype=int)
    }

    # Group particles by window
    for y in tqdm(range(n_rows)):
        for x in range(n_cols):

            selected = (iy == y) & (ix == x)

            if not np.any(selected):
                continue

            results['counts'][y, x] = np.sum(selected)

            results['magnitude_mean'][y, x] = np.mean(mag[selected])
            results['magnitude_std'][y, x] = np.std(mag[selected])

            results['dx_mean'][y, x] = np.mean(dis[selected, 1])
            results['dx_std'][y, x] = np.std(dis[selected, 1])

            results['dy_mean'][y, x] = np.mean(dis[selected, 0])
            results['dy_std'][y, x] = np.std(dis[selected, 0])

    results['grid_shape'] = shape
    results['x_edges'] = x_min + np.arange(n_cols + 1) * win_x
    results['y_edges'] = y_min + np.arange(n_rows + 1) * win_y

    return results

#%%
order_calib = 2
dt = 0.006 #sec

lim_lx, lim_ly = calib_fun2( left_l, np.arange(1024), la2.x)
lim_rx, lim_ry = calib_fun2( righ_l, np.arange(1024), la2.x)


if order_calib == 2:
    Xw, Yw = calib_fun2( all_pos[:,1], all_pos[:,0], la2.x)
    Xw2, Yw2 = calib_fun2( all_pos[:,1] + all_dis[:,1], all_pos[:,0] + all_dis[:,0], la2.x)

elif order_calib == 1:
    Xw, Yw = calib_func( all_pos[:,1], all_pos[:,0], lal.x)
    Xw2, Yw2 = calib_func( all_pos[:,1] + all_dis[:,1], all_pos[:,0] + all_dis[:,0], lal.x)

all_rp = np.vstack([Yw,Xw]).T
all_rd = np.vstack([Yw2,Xw2]).T - all_rp

t1 = time()
stats = displacement_statistics_mm( all_rp, all_rd, window_size=1.,
    lim_lx=lim_lx, lim_ly=lim_ly, lim_rx=lim_rx, lim_ry=lim_ry, dt=dt )
t2 = time()

print(t2-t1)

np.savez_compressed(main_folder+'/displacement_statistics(o2_g10).npz', **stats)
#%%

# Load saved results
# loaded = np.load(main_folder+'/displacement_statistics(o1_g5).npz')
# loaded = np.load(main_folder+'/displacement_statistics(o2_g5).npz')

loaded = np.load(main_folder+'/displacement_statistics(o2_g10).npz')



# loaded = np.load('/Volumes/PTV channel/f6'+'/displacement_statistics(o2_g5).npz')
# loaded = np.load('/Volumes/PTV channel/f5'+'/displacement_statistics(o2_g5).npz')
# loaded = np.load('/Volumes/PTV channel/f0'+'/displacement_statistics(o2_g5).npz')

stats = {key: loaded[key] for key in loaded.files}



#%%

plt.figure()

# plt.plot( lim_lx, lim_ly, 'b-'  )
# plt.plot( lim_rx, lim_ry, 'r-'  )

# plt.plot( all_rp[::1000,1], all_rp[::1000,0], '.'  )


# plt.axis('equal')
plt.show()


extent = [ stats['x_edges'][0], stats['x_edges'][-1], stats['y_edges'][0], stats['y_edges'][-1] ]

fig, ax = plt.subplots(1, 4, layout='constrained', figsize=(16, 5))

ax[0].set_title('Counts')
im0 = ax[0].imshow(stats['counts'], origin='lower', extent=extent, aspect='auto')
fig.colorbar(im0, ax=ax[0])

ax[1].set_title('Magnitude')
im1 = ax[1].imshow(stats['magnitude_mean'], origin='lower', extent=extent, aspect='auto')
fig.colorbar(im1, ax=ax[1])

ax[2].set_title('Dx')
im2 = ax[2].imshow(stats['dx_mean'], origin='lower', extent=extent, aspect='auto')
fig.colorbar(im2, ax=ax[2])

ax[3].set_title('Dy')
im3 = ax[3].imshow(stats['dy_mean'], origin='lower', extent=extent, aspect='auto')
fig.colorbar(im3, ax=ax[3])

for a in ax:
    a.set_xlabel('X (mm)')
    a.set_ylabel('Y (mm)')

plt.show()



#%%
extent = [ stats['x_edges'][0], stats['x_edges'][-1], stats['y_edges'][0], stats['y_edges'][-1] ]

x_centers = (stats['x_edges'][:-1] + stats['x_edges'][1:]) / 2
y_centers = (stats['y_edges'][:-1] + stats['y_edges'][1:]) / 2
Xg, Yg = np.meshgrid(x_centers, y_centers)
vx, vy = stats['dx_mean'], stats['dy_mean']

ss = 5
gvy, gvx = nangauss(vy , ss), nangauss(vx , ss)
vy_dy, vy_dx = np.gradient(gvy, y_centers, x_centers)
vx_dy, vx_dx = np.gradient(gvx, y_centers, x_centers)

vort = vy_dx - vx_dy


fig, ax = plt.subplots(1, 4, layout='constrained', figsize=(16, 5))
im0 = ax[0].imshow( vy, origin='lower', extent=extent, aspect='auto')
fig.colorbar(im0, ax=ax[0])
im0 = ax[1].imshow( gvy, origin='lower', extent=extent, aspect='auto')
fig.colorbar(im0, ax=ax[1])
im0 = ax[2].imshow( vy_dx, origin='lower', extent=extent, aspect='auto')
fig.colorbar(im0, ax=ax[2])
im0 = ax[3].imshow( vy_dy, origin='lower', extent=extent, aspect='auto')
fig.colorbar(im0, ax=ax[3])
plt.show()

fig, ax = plt.subplots(1, 4, layout='constrained', figsize=(16, 5))
im0 = ax[0].imshow( vx, origin='lower', extent=extent, aspect='auto')
fig.colorbar(im0, ax=ax[0])
im0 = ax[1].imshow( gvx, origin='lower', extent=extent, aspect='auto')
fig.colorbar(im0, ax=ax[1])
im0 = ax[2].imshow( vx_dx, origin='lower', extent=extent, aspect='auto')
fig.colorbar(im0, ax=ax[2])
im0 = ax[3].imshow( vx_dy, origin='lower', extent=extent, aspect='auto')
fig.colorbar(im0, ax=ax[3])
plt.show()

plt.figure(figsize=(4,5))
plt.imshow( vort, origin='lower', extent=extent, aspect='auto' )
plt.colorbar()
plt.show()

#%%

# from scipy.io import savemat

# savemat(main_folder + '/scallop_plot.mat', {
#     'x': x_centers, 'y': y_centers,
#     'gvx': gvx, 'gvy': gvy, 'vort': vort,      # NaNs are preserved 
#     })



#%%

# def adaptive_quiver(ax, X, Y, U, V, min_spacing=1, max_spacing=5, **kwargs):

#     speed = np.sqrt(U**2 + V**2)

#     # Normalize velocity magnitude
#     smin = np.nanmin(speed)
#     smax = np.nanmax(speed)

#     if smax > smin:
#         speed_norm = (speed - smin) / (smax - smin)
#     else:
#         speed_norm = np.zeros_like(speed)

#     # Local spacing: larger for higher velocities
#     spacing = min_spacing + speed_norm * (max_spacing - min_spacing)

#     ny, nx = speed.shape

#     selected = np.zeros((ny, nx), dtype=bool)

#     # Visit grid points, prioritizing a regular grid
#     for iy in range(ny):
#         for ix in range(nx):

#             if not np.isfinite(speed[iy, ix]):
#                 continue

#             if not np.isfinite(U[iy, ix]) or not np.isfinite(V[iy, ix]):
#                 continue

#             # Check distance to already selected arrows
#             yy, xx = np.where(selected)

#             if len(yy) == 0:
#                 selected[iy, ix] = True
#                 continue

#             distances = np.sqrt(
#                 (X[iy, ix] - X[yy, xx])**2 +
#                 (Y[iy, ix] - Y[yy, xx])**2
#             )

#             if np.all(distances >= spacing[iy, ix]):
#                 selected[iy, ix] = True

#     q = ax.quiver( X[selected], Y[selected], U[selected], V[selected], **kwargs )

#     return q, selected



def draw_arrows(ax, x, y, u, v, arrow_scale,
                head_frac=0.35,      # head length as a fraction of arrow length
                head_aspect=0.6,     # head full width / head length
                shaft_frac=0.06,     # shaft full width / arrow length
                min_head=0.4,        # smallest head length (data units)
                min_shaft=0.12,      # smallest shaft width (data units)
                pivot='tail', **kwargs):
    speed = np.hypot(u, v)
    L = speed * arrow_scale
    ok = L > 0
    x, y, u, v, speed, L = x[ok], y[ok], u[ok], v[ok], speed[ok], L[ok]

    ex, ey = u / speed, v / speed          # unit direction
    nx, ny = -ey, ex                       # unit normal

    hl = np.minimum(np.maximum(head_frac * L, min_head), L)   # head length
    hw = 0.5 * head_aspect * hl                               # head half-width
    sw = 0.5 * np.maximum(shaft_frac * L, min_shaft)          # shaft half-width
    sw = np.minimum(sw, hw)                                   # shaft never wider than head

    if pivot == 'middle':
        x0, y0 = x - 0.5 * L * ex, y - 0.5 * L * ey
    else:
        x0, y0 = x, y

    bx, by = x0 + (L - hl) * ex, y0 + (L - hl) * ey   # head base centre
    tx, ty = x0 + L * ex, y0 + L * ey                 # tip

    verts = np.stack([
        np.column_stack([x0 + sw * nx, y0 + sw * ny]),
        np.column_stack([bx + sw * nx, by + sw * ny]),
        np.column_stack([bx + hw * nx, by + hw * ny]),
        np.column_stack([tx, ty]),
        np.column_stack([bx - hw * nx, by - hw * ny]),
        np.column_stack([bx - sw * nx, by - sw * ny]),
        np.column_stack([x0 - sw * nx, y0 - sw * ny]),
    ], axis=1)                                         # (N, 7, 2)

    pc = PolyCollection(verts, **kwargs)
    ax.add_collection(pc)
    return pc


def adaptive_quiver(ax, X, Y, U, V, arrow_scale=None, spacing_factor=1.2,
                    min_spacing=None, max_arrow_len=None, **kwargs):
    """
    Quiver with locally adaptive density.

    arrow_scale    : data-units of arrow length per unit of velocity
                     (arrow length = speed * arrow_scale). Single global
                     factor, so arrows stay to scale.
    spacing_factor : required distance between arrows, in units of the
                     longer of the two arrows' drawn lengths.
    min_spacing    : floor on spacing (data units), e.g. grid pitch.
    max_arrow_len  : if given, sets arrow_scale so the fastest arrow has
                     this length (in data units, e.g. mm).
    """
    speed = np.hypot(U, V)
    valid = np.isfinite(speed) & np.isfinite(U) & np.isfinite(V)

    if arrow_scale is None:
        if max_arrow_len is None:
            raise ValueError("give arrow_scale or max_arrow_len")
        arrow_scale = max_arrow_len / np.nanmax(speed)

    length = speed * arrow_scale          # drawn length in data units

    if min_spacing is None:
        # grid pitch
        min_spacing = np.nanmedian(np.abs(np.diff(X[0, :])))

    # Candidates sorted by speed (fastest first): fast arrows claim space first
    idx = np.argwhere(valid)
    order = np.argsort(-speed[valid])
    idx = idx[order]

    pts = np.column_stack([X[valid], Y[valid]])[order]
    lens = length[valid][order]

    selected = []
    sel_pts = np.empty((0, 2))
    sel_len = np.empty(0)

    # Simple greedy selection using a KD-tree of accepted points,
    # rebuilt in chunks for speed.
    accepted_pts = []
    accepted_len = []
    tree = None
    pending = 0
    rebuild_every = 50

    for k in range(len(pts)):
        p, l = pts[k], lens[k]
        need = max(min_spacing, spacing_factor * l)

        ok = True
        # check against tree (accepted before last rebuild)
        if tree is not None:
            r = max(need, spacing_factor * max_len_so_far)
            nb = tree.query_ball_point(p, r)
            for j in nb:
                d = np.hypot(*(p - tree.data[j]))
                if d < max(need, spacing_factor * tree_len[j]):
                    ok = False
                    break
        # check against recent, not-yet-in-tree points
        if ok:
            for q, ql in zip(accepted_pts[len(accepted_pts) - pending:],
                             accepted_len[len(accepted_len) - pending:]):
                if np.hypot(*(p - q)) < max(need, spacing_factor * ql):
                    ok = False
                    break

        if ok:
            selected.append(k)
            accepted_pts.append(p)
            accepted_len.append(l)
            pending += 1
            if pending >= rebuild_every:
                tree = cKDTree(np.array(accepted_pts))
                tree_len = np.array(accepted_len)
                max_len_so_far = tree_len.max()
                pending = 0

    sel = np.array(selected, dtype=int)
    iy, ix = idx[sel, 0], idx[sel, 1]

    mask = np.zeros_like(speed, dtype=bool)
    mask[iy, ix] = True

    # angles='xy', scale_units='xy' with scale=1/arrow_scale makes the arrow
    # length in DATA units = speed * arrow_scale (true to scale on screen)

    q = ax.quiver(X[mask], Y[mask], U[mask], V[mask], angles='xy', scale_units='xy', scale=1.0 / arrow_scale, **kwargs)
    
    # draw_arrows(ax, X[mask], Y[mask], U[mask], V[mask], arrow_scale,
    #         head_frac=0.3,      # head length = 30% of arrow length
    #         head_aspect=0.6,    # head width = 60% of head length
    #         shaft_frac=0.05,    # shaft width = 5% of arrow length
    #         min_head=0,         # no floor: heads scale purely with length
    #         min_shaft=0,
    #         facecolor='k', edgecolor='none')
    # q = None
    
    return q, mask

#%%

def select_arrows(X, Y, U, V, arrow_scale, along_factor=1.05, perp_gap=1.5, min_gap=0.6):
    """Raster-order greedy selection with anisotropic exclusion.
    along_factor : required spacing along the arrow direction, in arrow lengths
    perp_gap     : required spacing perpendicular to the arrow (data units)
    min_gap      : floor on the along-arrow spacing (data units)
    """
    speed = np.hypot(U, V)
    valid = np.isfinite(speed) & (speed > 0)
    L = speed * arrow_scale
    iy, ix = np.nonzero(valid)                     # raster order
    sel_xy = np.empty((0, 2)); sel_dir = np.empty((0, 2)); sel_L = np.empty(0)
    keep = []
    for y_i, x_i in zip(iy, ix):
        p = np.array([X[y_i, x_i], Y[y_i, x_i]])
        e = np.array([U[y_i, x_i], V[y_i, x_i]]) / speed[y_i, x_i]
        l = L[y_i, x_i]
        if len(sel_L):
            d = sel_xy - p
            # test in the frame of the candidate and of each accepted arrow
            a1 = np.abs(d @ e);               p1 = np.abs(d @ np.array([-e[1], e[0]]))
            a2 = np.abs(np.sum(d * sel_dir, 1))
            p2 = np.abs(d[:, 0] * -sel_dir[:, 1] + d[:, 1] * sel_dir[:, 0])
            need1 = np.maximum(min_gap, along_factor * l)
            need2 = np.maximum(min_gap, along_factor * sel_L)
            clash = ((a1 < need1) & (p1 < perp_gap)) | ((a2 < need2) & (p2 < perp_gap))
            if clash.any():
                continue
        keep.append((y_i, x_i))
        sel_xy = np.vstack([sel_xy, p]); sel_dir = np.vstack([sel_dir, e]); sel_L = np.append(sel_L, l)
    mask = np.zeros_like(speed, dtype=bool)
    for y_i, x_i in keep:
        mask[y_i, x_i] = True
    return mask


# def draw_open_arrows(ax, x, y, u, v, arrow_scale, head_frac=0.3, head_angle_deg=25, **kw):
#     speed = np.hypot(u, v)
#     ex, ey = u / speed, v / speed
#     L = speed * arrow_scale
#     hl = head_frac * L
#     a = np.deg2rad(head_angle_deg)
#     tx, ty = x + L * ex, y + L * ey                      # tip
#     c, s = np.cos(a), np.sin(a)
#     # rotate the reversed direction by +/- angle to get the two head strokes
#     bx, by = -ex, -ey
#     h1x, h1y = tx + hl * (c * bx - s * by), ty + hl * (s * bx + c * by)
#     h2x, h2y = tx + hl * (c * bx + s * by), ty + hl * (-s * bx + c * by)
#     segs = np.concatenate([
#         np.stack([np.column_stack([x, y]),   np.column_stack([tx, ty])], 1),
#         np.stack([np.column_stack([tx, ty]), np.column_stack([h1x, h1y])], 1),
#         np.stack([np.column_stack([tx, ty]), np.column_stack([h2x, h2y])], 1),
#     ])
#     lc = LineCollection(segs, **kw)
#     ax.add_collection(lc)
#     return lc



def draw_arrows(ax, x, y, u, v, arrow_scale,
                head_frac=0.28,        # head length as a fraction of arrow length
                head_half_angle=14,    # degrees; smaller = slimmer head
                shaft_frac=0.025,      # shaft full width as a fraction of arrow length
                min_shaft=0.0,         # optional floor on shaft width (data units)
                max_head=None,         # optional cap on head length (data units)
                **kw):
    speed = np.hypot(u, v)
    ok = speed > 0
    x, y, u, v, speed = x[ok], y[ok], u[ok], v[ok], speed[ok]

    ex, ey = u / speed, v / speed
    nx, ny = -ey, ex
    L = speed * arrow_scale

    hl = head_frac * L
    if max_head is not None:
        hl = np.minimum(hl, max_head)
    hl = np.minimum(hl, L)
    hw = hl * np.tan(np.deg2rad(head_half_angle))              # head half-width
    sw = 0.5 * np.maximum(shaft_frac * L, min_shaft)           # shaft half-width
    sw = np.minimum(sw, hw)

    tx, ty = x + L * ex, y + L * ey                            # tip
    bx, by = tx - hl * ex, ty - hl * ey                        # head base centre

    verts = np.stack([
        np.column_stack([x  + sw * nx, y  + sw * ny]),
        np.column_stack([bx + sw * nx, by + sw * ny]),
        np.column_stack([bx + hw * nx, by + hw * ny]),
        np.column_stack([tx, ty]),
        np.column_stack([bx - hw * nx, by - hw * ny]),
        np.column_stack([bx - sw * nx, by - sw * ny]),
        np.column_stack([x  - sw * nx, y  - sw * ny]),
    ], axis=1)

    pc = PolyCollection(verts, **kw)
    ax.add_collection(pc)
    return pc

#%%


# vmin = np.nanmin(vort)
# vmax = np.nanmax(vort)

# norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

vmax = np.nanmax(np.abs(vort))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


cmap = plt.cm.RdBu_r.copy()
cmap.set_bad('gray')

vort_masked = np.ma.masked_invalid(vort)
vy_masked = np.ma.masked_invalid(gvy)

fig, ax = plt.subplots(figsize=(5, 6))


im = plt.imshow( vort_masked, origin='lower', extent=extent, aspect='auto', cmap=cmap, norm=norm )
# im = plt.imshow( vy, origin='lower', extent=extent, aspect='auto', cmap=cmap, norm=norm )
cbar = plt.colorbar(im, ax=ax, label='Vorticity')

step = 10
# q = ax.quiver( Xg[::step], Yg[::step], gvx[::step], gvy[::step], color='black', scale=5e3, pivot='middle' )
# q = adaptive_quiver(ax, Xg, Yg, gvx, gvy, min_spacing=1, max_spacing=5, color='black', scale=5e3, pivot='middle')

q, sel = adaptive_quiver( ax, Xg, Yg, gvx, gvy,
        arrow_scale=None,
        max_arrow_len=20,          
        spacing_factor=.4,       
        min_spacing=3.,
        color='black', width=0.004, headwidth=4, headlength=4,
        pivot='middle')

ax.quiverkey(q, 0.85, 1.03, 400.0, '400 mm/s', labelpos='E')


# arrow_scale = 10.0 / np.nanmax(np.hypot(gvx, gvy))     # fastest arrow = 6 mm long (tune)

# mask_n = select_arrows(Xg, Yg, gvx, gvy, arrow_scale,
#                       along_factor=1.05, perp_gap=1.5, min_gap=0.1)

# draw_open_arrows(ax, Xg[mask_n], Yg[mask_n], gvx[mask_n], gvy[mask_n], arrow_scale,
#                   colors='k', linewidths=0.6)

# # reference arrow, like the "0.5 m/s" key in the paper
# ref = 0.5                                              # same units as gvx
# draw_open_arrows(ax, np.array([5.0]), np.array([-22.0]),
#                   np.array([ref]), np.array([0.0]), arrow_scale, colors='k', linewidths=0.6)
# ax.text(5.0, -20.5, '0.5 m/s', fontsize=8)



ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')

ax.axis('equal')

filename = main_folder + '/Figs/ptv.pdf' 
plt.savefig(filename, dpi=200, bbox_inches='tight')


plt.show()

#%%


#%%
# =============================================================================
# For testing some things internally
# =============================================================================
i = 30

img1 = tifffile.imread(files[i])
img2 = tifffile.imread(files[i+1])

im1 = (img1/mim * mask)[nlim]
im2 = (img2/mim * mask)[nlim]

imb1 = im1 > 4
imb2 = im2 > 4


t1 = time()

pos, dis, grid_shape, window = local_cross_correlation(im1, im2, n_windows_x=9, n_windows_y=14, overlap=0.5)
dis = remove_displacement_outliers(dis, grid_shape, threshold=1.5)

t2 = time()

rim1 = remove_small_objects(imb1, min_size=15)
rim2 = remove_small_objects(imb2, min_size=15)

lim1, lim2 = label(imb1 - rim1*1), label(imb2 - rim2*1)

pos1 = subpixel_particle_positions(im1, lim1)
pos2 = subpixel_particle_positions(im2, lim2)

t3 = time()

mpa1, mpa2, dists, preds = match_particles(pos1, pos2, pos, -dis, radius=10)
displ = pos2[mpa2] - pos1[mpa1]

t4 = time()

print(t4-t1, t3-t2, t2-t1)
print( f'Number matches: {len(mpa1)}' )
print(grid_shape, window)


plt.figure()
# plt.imshow( corrb )

plt.imshow( imb1 )
plt.imshow( imb2, alpha=0.3 )
plt.imshow( mask[nlim], alpha=0.2 )

plt.quiver(pos[:,1], pos[:,0], -dis[:,1], dis[:,0], color='r')


plt.show()


# plt.figure()
# plt.imshow( lim1>0 )
# # plt.imshow( lim2>0 )
# plt.show()

fig, ax = plt.subplots(1,2, layout='constrained')

ax[0].imshow( im1 )
ax[0].imshow( mask[nlim], alpha=0.2 )
ax[0].scatter( pos1[:,1], pos1[:,0], s=2, c='r' )

ax[1].imshow( im2 )
ax[1].imshow( mask[nlim], alpha=0.2 )
ax[1].scatter( pos2[:,1], pos2[:,0], s=2, c='r' )
plt.show()


plt.figure(  )

plt.imshow(mask[nlim], alpha=0.3)

# plt.plot( [pos1[:,1], preds[:,1]], [pos1[:,0], preds[:,0]], '-'  )
plt.plot( [pos1[mpa1,1], pos2[mpa2,1]], [pos1[mpa1,0], pos2[mpa2,0]], 'g-'  )
# plt.quiver( pos1[mpa1,1], pos1[mpa1,0], -displ[:,1], displ[:,0],  )

plt.scatter( pos1[:,1], pos1[:,0], s=2, c='r' )
plt.scatter( pos2[:,1], pos2[:,0], s=2, c='b' )

plt.show()


#%%

pos1[mpa1][:3], pos2[mpa2][:3], displ[:3]


#%%
# =============================================================================
# Averaged PIV
# =============================================================================

import h5py
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor

# def fit_corr_gaussian(corr, r=5):
#     def gaussian(coords, A, x0, y0, sx, sy, offset):
#         x, y = coords
#         return offset + A * np.exp( -(x-x0)**2/(2*sx**2) - (y-y0)**2/(2*sy**2) )

#     i0, j0 = np.unravel_index(np.argmax(corr), corr.shape)

#     i_min, j_min = max(0, i0-r), max(0, j0-r)
#     region = corr[i_min:i0+r+1, j_min:j0+r+1]

#     I, J = np.indices(region.shape)
#     I += i_min
#     J += j_min

#     p0 = [corr[i0, j0], i0, j0, 2, 2, 0]

#     popt, _ = curve_fit( gaussian, (I.ravel(), J.ravel()), region.ravel(), p0=p0, maxfev=5000 )

#     _, i_center, j_center, sigma_i, sigma_j, _ = popt

#     center = np.array([i_center, j_center]) - np.array(corr.shape)//2
#     std = np.array([sigma_i, sigma_j])

#     return center, std

def calculate_mean_std(files, nlim, num):
    Aimm = np.zeros((1024,1280))[nlim]
    Aivar = np.zeros((1024,1280))[nlim]

    Bimm = np.zeros((1024,1280))[nlim]
    Bivar = np.zeros((1024,1280))[nlim]

    li = range(0,num,2)
    for j,i in enumerate(tqdm(li)):
        img1 = tifffile.imread(files[i])[nlim].astype(np.float64)
        img2 = tifffile.imread(files[i+1])[nlim].astype(np.float64)

        delta = img1 - Aimm
        Aimm += delta / (j+1)
        Aivar += delta * (img1 - Aimm)

        delta = img2 - Bimm
        Bimm += delta / (j+1)
        Bivar += delta * (img2 - Bimm)


    Aisd = np.sqrt(Aivar / (num/2))
    Bisd = np.sqrt(Bivar / (num/2))
    
    return Aimm, Aisd, Bimm, Bisd



def _standardize(X):
    """Remove the per-pixel temporal mean and divide by the temporal std.
    Pixels with zero variance become NaN (they have no defined correlation)."""
    X = X.astype(np.float32)
    X -= X.mean(-1, keepdims=True)
    s = np.sqrt((X**2).mean(-1, keepdims=True))
    s[s == 0] = np.nan
    return X / s


def corr_rows(A_stack, B_stack, r0, r1, size):
    """Pearson correlation for every pixel in rows r0:r1.

    Returns corr[i+size, j+size, y-r0, x] = corr(A[y,x,:], B[y+i,x+j,:]),
    NaN where (y+i, x+j) falls outside the image (edges)."""
    H, W, n = A_stack.shape
    b0, b1 = max(0, r0 - size), min(H, r1 + size)       # B rows needed (halo)
    A = _standardize(A_stack[r0:r1])                     # (h, W, n)
    B = _standardize(B_stack[b0:b1])                     # (h+halo, W, n)

    corr = np.full((2*size+1, 2*size+1, r1-r0, W), np.nan, np.float32)
    for i in range(-size, size+1):
        ylo, yhi = max(r0, -i), min(r1, H - i)           # rows whose y+i is inside

        if ylo >= yhi:
            continue

        Ai = A[ylo-r0:yhi-r0]
        Bi = B[ylo+i-b0:yhi+i-b0]

        for j in range(-size, size+1):
            xlo, xhi = max(0, -j), min(W, W - j)         # columns whose x+j is inside
            if xlo >= xhi:
                continue

            corr[i+size, j+size, ylo-r0:yhi-r0, xlo:xhi] = np.einsum( 'yxt,yxt->yx', Ai[:, xlo:xhi], Bi[:, xlo+j:xhi+j]) / n

    return corr


def _gaussian(coords, A, x0, y0, sx, sy, offset):
    x, y = coords
    return offset + A*np.exp(-(x-x0)**2/(2*sx**2) - (y-y0)**2/(2*sy**2))


def fit_corr_gaussian(corr, r=5):
    """Your fit, made safe for truncated (NaN) correlation maps.
    Returns (di, dj, sigma_i, sigma_j, peak); all NaN if the fit is not possible."""
    bad = (np.nan,)*5
    if np.all(np.isnan(corr)):
        return bad
    i0, j0 = np.unravel_index(np.nanargmax(corr), corr.shape)
    i_min, j_min = max(0, i0-r), max(0, j0-r)
    region = corr[i_min:i0+r+1, j_min:j0+r+1]
    if np.isnan(region).any() or region.shape[0] < 3 or region.shape[1] < 3:
        return bad
    I, J = np.indices(region.shape)
    I = I + i_min
    J = J + j_min
    p0 = [corr[i0, j0], i0, j0, 2, 2, 0]
    try:
        popt, _ = curve_fit(_gaussian, (I.ravel(), J.ravel()), region.ravel(),
                            p0=p0, maxfev=5000)
    except (RuntimeError, ValueError):
        return bad
    _, ic, jc, si, sj, _ = popt
    c = np.array(corr.shape)//2
    return ic - c[0], jc - c[1], abs(si), abs(sj), corr[i0, j0]


def single_pixel_fit(A_stack, B_stack, size=10, r=5, rows=4, mask=None,
                     keep_corr_at=None):
    """Single-pixel ensemble correlation + Gaussian fit at every pixel.

    A_stack, B_stack : (H, W, n) arrays or h5py datasets
    size  : search half-width in px (window is 2*size+1 square)
    rows  : image rows processed per chunk (memory/speed trade-off)
    mask  : optional (H, W) bool; only these pixels are fitted
    Returns maps (H, W): di (row shift), dj (col shift), sigma_i, sigma_j, peak."""

    H, W, n = A_stack.shape
    out = np.full((5, H, W), np.nan, np.float32)

    for r0 in tqdm(range(0, H, rows)):
        r1 = min(r0 + rows, H)
        corr = corr_rows(A_stack, B_stack, r0, r1, size)

        for y in range(r1 - r0):
            for x in range(W):

                if mask is not None and not mask[r0+y, x]:
                    continue

                out[:, r0+y, x] = fit_corr_gaussian(corr[:, :, y, x], r)
    return out


def new_mask(A_std, A_mean, sigma=10):
    gim = gaussian( A_std * A_mean, sigma)
    ig_f = np.gradient(gim, axis=1)

    aro = np.arange(1024)
    ami = np.argmin(ig_f, axis = 1)

    mask = np.zeros_like(A_std)
    for i in range(1024):
        mask[i,:ami[i]] = 1
        
    indices_sub = np.zeros_like(ami) * np.nan
    for j, idx in enumerate(ami):

        if np.isnan(idx): 
            continue
        if idx <= 0 or idx >= ig_f.shape[1] - 1:
            continue

        # Three neighbouring gradient values
        g1, g2, g3 = ig_f[j, idx - 1], ig_f[j, idx], ig_f[j, idx + 1]

        denominator = g1 - 2*g2 + g3
        if denominator != 0:
            delta = 0.5 * (g1 - g3) / denominator
            # Avoid unrealistic sub-pixel corrections
            if abs(delta) <= 1:
                indices_sub[j] = idx + delta

    return mask, aro, indices_sub

#%%

do = 0
show_bar = 1
rango = len(files)


all_f = range(0, rango, 2)
n = len(all_f)

batch_size = 64
workers = 4

# Determine cropped image dimensions
img0 = tifffile.imread(files[0])[nlim]
H, W = img0.shape
dtype = img0.dtype


def read_pair(i):
    img1 = tifffile.imread(files[i])[nlim]
    img2 = tifffile.imread(files[i+1])[nlim]
    return img1, img2


if do:
    with h5py.File(main_folder+'/tiffs_data.h5', 'w') as f:
    
        # Create datasets
        chunks = (min(8,H), min(8,W), min(64,n))
    
        A_stack = f.create_dataset( 'A_stack', shape=(H,W,n), dtype=dtype, chunks=chunks )
    
        B_stack = f.create_dataset( 'B_stack', shape=(H,W,n), dtype=dtype, chunks=chunks )
    
        # Read and write in batches
        with ThreadPoolExecutor(max_workers=workers) as executor:
    
            for start in tqdm(range(0, n, batch_size), disable=not show_bar):
    
                stop = min(start + batch_size, n)
                indices = range(2*start, 2*stop, 2)
    
                A_batch = np.empty((stop-start,H,W), dtype=dtype)
                B_batch = np.empty((stop-start,H,W), dtype=dtype)
    
                for k, (img1,img2) in enumerate(executor.map(read_pair, indices)):
                    A_batch[k] = img1
                    B_batch[k] = img2
    
                A_stack[:,:,start:stop] = A_batch.transpose(1,2,0)
                B_stack[:,:,start:stop] = B_batch.transpose(1,2,0)

    print('Saved')

else: 
    
    
    print('Not saved')    


#%%

#get means and stds for A and B
num = 8000

t1 = time()

A_mean, A_std, B_mean, B_std = calculate_mean_std(files, nlim, num)
mask, aro, sub_edge = new_mask(A_std, A_mean)

if exp in ['f6','f0']:
    mask2, aro, sub_edge2 = new_mask(A_std[:,::-1], A_mean[:,::-1])
    rows, cols = mask.shape
    mask2, sub_edge2 = mask2[:,::-1], cols - sub_edge2 
    
    mask = mask * mask2

t2 = time()
print(t2-t1)

#%%

fig, ax = plt.subplots(1,4, figsize=(12,5), layout='constrained' )

im = ax[0].imshow( A_mean, vmax=150  )
fig.colorbar(im, ax=ax[0])

im = ax[1].imshow( B_mean - A_mean)
fig.colorbar(im, ax=ax[1])

im = ax[2].imshow( A_std )
fig.colorbar(im, ax=ax[2])

im = ax[3].imshow( B_std - A_std )
fig.colorbar(im, ax=ax[3])

plt.show()


fig, ax = plt.subplots(1,2, figsize=(6,5), layout='constrained' )

ax[0].imshow( A_std )
# plt.imshow( B_mean, vmax=300 )
ax[0].plot( sub_edge, aro, 'r-')


ax[1].imshow(mask)
ax[1].plot( sub_edge, aro, 'r-')

if exp in ['f6','f0']:
    ax[0].plot( sub_edge2, aro, 'r-')
    ax[1].plot( sub_edge2, aro, 'r-')
plt.show()


#%%
# check one correlation
size = 25
r, c = 380, 317

t1 = time()
with h5py.File(main_folder+'/tiffs_data.h5', 'r') as f:
    A_stack = f['A_stack']
    B_stack = f['B_stack']

    line_A = A_stack[r,c,:]
    ny,nx,nt = A_stack.shape
    
    corr = np.zeros((2*size+1, 2*size+1))
    for i in tqdm(range(-size, size+1)):
        for j in range(-size, size+1):
    
            ri, cj = r+i, c+j
            if 0 <= ri < ny and 0<= cj < nx:
                line_B = B_stack[r+i, c+j, :]
                corr[i+size, j+size] = np.corrcoef(line_A, line_B)[0,1]
                # corr[i+size, j+size] = np.cov(line_A, line_B)[0,1]
    
cy, cx, sdy, sdx, cov = fit_corr_gaussian(corr, r=7)

t2 = time()

print()
print('Time: ', t2-t1)
print('Sigmas: ', sdx, sdy)
print('Disp: ', cx, cy)

print()
print('Disp R: ', nR['dj'][r,c], nR['di'][r,c])
vx, vy = nangauss(nR['dj'] , ss), nangauss(nR['di'] , ss)
print('Disp Rs: ', vx[r,c], vy[r,c])
#%%

plt.figure()
plt.imshow( corr, extent=(-size,size,size,-size) )
plt.plot(cx,cy, 'r.')
plt.colorbar()
# filename = main_folder + '/Figs/corr_wall(1).pdf' 
# plt.savefig(filename, dpi=200, bbox_inches='tight')
plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# I, J = np.meshgrid(
#     np.arange(-size, size+1),
#     np.arange(-size, size+1)
# )

# ax.plot_surface(J, I, corr, cmap='viridis', edgecolor='none')

# ax.set_xlabel('Column offset (j)')
# ax.set_ylabel('Row offset (i)')
# ax.set_zlabel('Correlation')

# plt.show()


plt.figure()
plt.imshow( mask )

plt.quiver( c, r, cx, cy, scale_units='xy', angles='xy', pivot='tail', scale=1 )
plt.quiver( c, r, nR['dj'][r,c], nR['di'][r,c], color='red', scale_units='xy', angles='xy', pivot='tail', scale=1 )
plt.quiver( c, r, vx[r,c], vy[r,c], color='blue', scale_units='xy', angles='xy', pivot='tail', scale=1 )

plt.show()


#%%



#%%

with h5py.File(main_folder+'/tiffs_data.h5', 'r') as f:
    A_stack = f['A_stack']
    B_stack = f['B_stack']

    line_A = A_stack[r,c,:]
    
    dx,dy = 1, -14
    line_B = B_stack[r+dx,c+dy,:]

    plt.figure()
    plt.plot( line_A )
    plt.plot( line_B )
    plt.show()



#%%

#%%









#%%

def nangauss(altus, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):    
        V = altus.copy()
        V[np.isnan(altus)] = 0
        W = 0 * altus.copy() + 1
        W[np.isnan(altus)] = 0
        VV,WW = ndi.gaussian_filter(V, sigma), ndi.gaussian_filter(W, sigma)
        gdp = VV/WW
        gdp[np.isnan(altus)] = np.nan
        return gdp


def _z(X, mean, std):
    """Standardise a (h, w, n) block with static per-pixel stats.
    Returns Z (invalid pixels set to 0) and ok (h, w) bool."""
    mean = np.asarray(mean, np.float32)
    std = np.asarray(std, np.float32)
    ok = np.isfinite(std) & (std > 0) & np.isfinite(mean)
    Z = np.array(X, dtype=np.float32)
    Z -= np.where(ok, mean, 0)[..., None]
    Z /= np.where(ok, std, 1)[..., None]
    Z[~ok] = 0
    return Z, ok


def _bbox(mask, H, W):
    if mask is None:
        return 0, H, 0, W
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError('mask is empty')
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


# ----------------------------------------------------------------------------
# pass 1: coarse block-ensemble correlation -> predictor
# ----------------------------------------------------------------------------
def coarse_blocks(A_stack, B_stack, A_mean, A_std, B_mean, B_std, mask,
                  size, block, n_coarse, min_fill, rows):
    H, W, n = A_stack.shape
    nc = min(n_coarse, n)
    ya, yb, xa, xb = _bbox(mask, H, W)
    K = 2*size + 1

    nby, nbx = -(-(yb-ya)//block), -(-(xb-xa)//block)
    Cb = np.full((K, K, nby, nbx), np.nan, np.float32)
    rows_c = block * max(1, rows // block)
    Wp = nbx * block
    bx0, bx1 = max(0, xa-size), min(W, xb+size)          # B column range

    for r0 in tqdm(range(ya, yb, rows_c), desc='coarse'):
        r1 = min(r0 + rows_c, yb)
        h = r1 - r0
        hp = -(-h//block) * block
        b0, b1 = max(0, r0-size), min(H, r1+size)
        ZA, okA = _z(A_stack[r0:r1, xa:xb, :nc], A_mean[r0:r1, xa:xb], A_std[r0:r1, xa:xb])
        wA = okA & (mask[r0:r1, xa:xb] if mask is not None else True)
        ZA[~wA] = 0
        wA = wA.astype(np.float32)
        ZB, okB = _z(B_stack[b0:b1, bx0:bx1, :nc], B_mean[b0:b1, bx0:bx1], B_std[b0:b1, bx0:bx1])
        wB = okB.astype(np.float32)
        P = np.zeros((hp, Wp), np.float32)
        Wt = np.zeros((hp, Wp), np.float32)
        wdt = xb - xa
        by0 = (r0 - ya)//block

        for i in range(-size, size+1):
            ylo, yhi = max(r0, -i), min(r1, H - i)       # rows y with y+i inside
            if ylo >= yhi:
                continue
            for j in range(-size, size+1):
                xlo, xhi = max(xa, -j), min(xb, W - j)   # cols x with x+j inside
                if xlo >= xhi:
                    continue
                P[:] = 0
                Wt[:] = 0
                sa = (slice(ylo-r0, yhi-r0), slice(xlo-xa, xhi-xa))
                sb = (slice(ylo+i-b0, yhi+i-b0), slice(xlo+j-bx0, xhi+j-bx0))
                P[sa] = np.einsum('yxt,yxt->yx', ZA[sa], ZB[sb])
                Wt[sa] = wA[sa] * wB[sb]
                ps = P.reshape(hp//block, block, nbx, block).sum((1, 3))
                ws = Wt.reshape(hp//block, block, nbx, block).sum((1, 3))
                c = np.where(ws >= min_fill*block*block, ps/(nc*np.maximum(ws, 1e-9)), np.nan)
                Cb[i+size, j+size, by0:by0+hp//block] = c
    return Cb, (ya, xa)


def build_predictor(Cb, origin, H, W, size, block, outlier_tol=3.0, min_peak=0.0):
    """Block correlation maps -> integer predictor (Pi, Pj) at every pixel."""
    K = 2*size + 1
    nby, nbx = Cb.shape[2:]
    flat = np.where(np.isnan(Cb), -np.inf, Cb).reshape(K*K, nby, nbx)
    idx = flat.argmax(0)
    peak = np.take_along_axis(flat, idx[None], 0)[0]

    Pi_b = (idx//K - size).astype(np.float32)
    Pj_b = (idx % K - size).astype(np.float32)
    bad = ~np.isfinite(peak) | (peak < min_peak)
    Pi_m, Pj_m = np.where(bad, np.nan, Pi_b), np.where(bad, np.nan, Pj_b)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mi = ndi.generic_filter(Pi_m, np.nanmedian, size=3, mode='nearest')
        mj = ndi.generic_filter(Pj_m, np.nanmedian, size=3, mode='nearest')
    with np.errstate(invalid='ignore'):
        bad |= (np.abs(Pi_m - mi) > outlier_tol) | (np.abs(Pj_m - mj) > outlier_tol)
    if bad.all():
        raise RuntimeError('coarse pass found no valid block; check mask / size / images')

    _, (iy, ix) = ndi.distance_transform_edt(bad, return_indices=True)
    Pi_f, Pj_f = Pi_b[iy, ix], Pj_b[iy, ix]               # bad blocks <- nearest good
    ya, xa = origin
    Yb, Xb = np.meshgrid((np.arange(H)-ya+0.5)/block-0.5,
                         (np.arange(W)-xa+0.5)/block-0.5, indexing='ij')
    Pi = np.clip(np.rint(ndi.map_coordinates(Pi_f, [Yb, Xb], order=1, mode='nearest')), -size, size)
    Pj = np.clip(np.rint(ndi.map_coordinates(Pj_f, [Yb, Xb], order=1, mode='nearest')), -size, size)
    return Pi.astype(np.int32), Pj.astype(np.int32), (Pi_b, Pj_b, peak, bad)



def _gaussian(coords, A, x0, y0, sx, sy, offset):
    x, y = coords
    return offset + A*np.exp(-(x-x0)**2/(2*sx**2) - (y-y0)**2/(2*sy**2))


def fit_corr_gaussian(corr, r=3):
    """Your Gaussian fit. Returns (di, dj, sigma_i, sigma_j, peak) relative to the
    window centre; all NaN if the map has NaNs near the peak, the peak is on the
    border of the window (predictor too far off), or the fit fails."""
    bad = (np.nan,)*5
    if np.all(np.isnan(corr)):
        return bad
    i0, j0 = np.unravel_index(np.nanargmax(corr), corr.shape)
    if i0 in (0, corr.shape[0]-1) or j0 in (0, corr.shape[1]-1):
        return bad
    i_min, j_min = max(0, i0-r), max(0, j0-r)
    region = corr[i_min:i0+r+1, j_min:j0+r+1]
    if np.isnan(region).any() or min(region.shape) < 3:
        return bad
    I, J = np.indices(region.shape)
    try:
        popt, _ = curve_fit(_gaussian, ((I+i_min).ravel(), (J+j_min).ravel()), region.ravel(),
                            p0=[corr[i0, j0], i0, j0, 1.5, 1.5, 0], maxfev=5000)
    except (RuntimeError, ValueError):
        return bad
    _, ic, jc, si, sj, _ = popt
    c = np.array(corr.shape)//2
    return ic - c[0], jc - c[1], abs(si), abs(sj), corr[i0, j0]


def fit_3pt(corr):
    """Vectorised 3-point Gaussian (log-parabola) peak fit for corr (K, K, h, w).
    Returns (5, h, w): di, dj, sigma_i, sigma_j, peak (relative to window centre)."""
    K, _, h, w = corr.shape
    flat = np.where(np.isnan(corr), -np.inf, corr).reshape(K*K, h, w)
    idx = flat.argmax(0)
    i0, j0 = idx//K, idx % K
    interior = (i0 > 0) & (i0 < K-1) & (j0 > 0) & (j0 < K-1)
    i0c, j0c = np.clip(i0, 1, K-2), np.clip(j0, 1, K-2)

    def at(di, dj):
        return np.take_along_axis(flat, ((i0c+di)*K + (j0c+dj))[None], 0)[0]
    c0, cim, cip, cjm, cjp = at(0, 0), at(-1, 0), at(1, 0), at(0, -1), at(0, 1)
    out = np.full((5, h, w), np.nan, np.float32)
    with np.errstate(all='ignore'):
        ok = interior & np.all(np.stack([c0, cim, cip, cjm, cjp]) > 0, 0)
        l0, lim, lip, ljm, ljp = [np.log(v) for v in (c0, cim, cip, cjm, cjp)]
        di2, dj2 = lim - 2*l0 + lip, ljm - 2*l0 + ljp         # second differences (<0)
        ok &= (di2 < 0) & (dj2 < 0)
        out[0] = i0c + (lim - lip)/(2*di2) - K//2
        out[1] = j0c + (ljm - ljp)/(2*dj2) - K//2
        out[2] = 1/np.sqrt(-di2)
        out[3] = 1/np.sqrt(-dj2)
        out[4] = c0
    out[:, ~ok] = np.nan
    return out


def fine_pass(A_stack, B_stack, A_mean, A_std, B_mean, B_std, mask, Pi, Pj,
              margin=3, r=3, rows=32, method='gauss'):
    H, W, n = A_stack.shape
    K = 2*margin + 1
    out = np.full((5, H, W), np.nan, np.float32)
    ya, yb, xa0, xb0 = _bbox(mask, H, W)
    
    for r0 in tqdm(range(ya, yb, rows), desc='fine'):
        r1 = min(r0 + rows, yb)
        mk = mask[r0:r1] if mask is not None else np.ones((r1-r0, W), bool)
        cols = np.where(mk.any(0))[0]
        if cols.size == 0:
            continue
        xa, xb = cols[0], cols[-1] + 1
        h, w = r1 - r0, xb - xa
        Pi_c = np.where(mk, Pi[r0:r1], 0)[:, xa:xb]           # unmasked pixels: dummy 0
        Pj_c = np.where(mk, Pj[r0:r1], 0)[:, xa:xb]
        mk = mk[:, xa:xb]
        pi_min, pi_max = Pi_c[mk].min(), Pi_c[mk].max()
        pj_min, pj_max = Pj_c[mk].min(), Pj_c[mk].max()
        b0, b1 = max(0, r0+pi_min-margin), min(H, r1+pi_max+margin)
        bx0, bx1 = max(0, xa+pj_min-margin), min(W, xb+pj_max+margin)
        if b1 <= b0 or bx1 <= bx0:
            continue
        ZA, okA = _z(A_stack[r0:r1, xa:xb], A_mean[r0:r1, xa:xb], A_std[r0:r1, xa:xb])
        ZB, okB = _z(B_stack[b0:b1, bx0:bx1], B_mean[b0:b1, bx0:bx1], B_std[b0:b1, bx0:bx1])
        valid_A = okA & mk
        ZA[~valid_A] = 0
        y = np.arange(r0, r1)[:, None]
        x = np.arange(xa, xb)[None, :]
        corr = np.full((K, K, h, w), np.nan, np.float32)
        for di in range(-margin, margin+1):
            yy = y + Pi_c + di
            for dj in range(-margin, margin+1):
                xx = x + Pj_c + dj
                inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                yi = np.clip(yy - b0, 0, b1-b0-1)
                xi = np.clip(xx - bx0, 0, bx1-bx0-1)
                v = np.einsum('yxt,yxt->yx', ZA, ZB[yi, xi]) / n
                corr[di+margin, dj+margin] = np.where(inside & okB[yi, xi] & valid_A, v, np.nan)
        if method == '3pt':
            res = fit_3pt(corr)
        else:
            res = np.full((5, h, w), np.nan, np.float32)
            for yy_, xx_ in zip(*np.where(valid_A)):
                res[:, yy_, xx_] = fit_corr_gaussian(corr[:, :, yy_, xx_], r=min(r, margin))
        res[0] += Pi_c                                        # back to total displacement
        res[1] += Pj_c
        sl = (slice(None), slice(r0, r1), slice(xa, xb))
        out[sl] = np.where(valid_A[None], res, np.nan)
    return out

_MAPS = ('di', 'dj', 'sigma_i', 'sigma_j', 'peak', 'Pi', 'Pj')

def save_result(path, R, **params):
    """Save the dict returned by coarse_to_fine to one compressed .npz file.
    Extra keyword arguments (size=..., block=..., method=..., etc.) are stored
    alongside as a record of how the result was made."""
    Pi_b, Pj_b, peak_b, bad_b = R['coarse']
    np.savez_compressed(path, **{k: R[k] for k in _MAPS},
                        coarse_Pi=Pi_b, coarse_Pj=Pj_b, coarse_peak=peak_b,
                        coarse_bad=bad_b,
                        **{'param_' + k: np.asarray(v) for k, v in params.items()})


def load_result(path):
    """Inverse of save_result. Returns (R, params); R has the same layout as the
    dict from coarse_to_fine."""
    d = np.load(path)
    R = {k: d[k] for k in _MAPS}
    R['coarse'] = (d['coarse_Pi'], d['coarse_Pj'], d['coarse_peak'], d['coarse_bad'])
    params = {k[6:]: d[k].item() for k in d.files if k.startswith('param_')}
    return R, params


def result_from_fine_out(out, Pi=None, Pj=None):
    """Rebuild the maps dict from the raw (5, H, W) array returned by fine_pass."""
    R = dict(di=out[0], dj=out[1], sigma_i=out[2], sigma_j=out[3], peak=out[4])
    if Pi is not None:
        R.update(Pi=Pi, Pj=Pj)
    return R


def disp_to_world(R, calib_fun, v, row0=0, col0=0, h=0.5):
    """Convert pixel displacements / peak widths to world units.

    R                : dict from coarse_to_fine (or load_result); uses R['di'], R['dj']
                       (displacement in rows / columns, px) and R['sigma_i'], R['sigma_j']
                       (correlation-peak widths along rows / columns, px)
    calib_fun        : your calib_fun2(x, y, v) -> (X, Y); x = image column, y = image row
    v                : calibration parameters
    row0, col0       : pixel position, in the FULL image the calibration was made on,
                       of the first row / column of the arrays (the nlim crop offset)
    h                : half step (px) for the finite-difference Jacobian

    Returns a copy of R with the extra keys X, Y (world position of each pixel),
    dX, dY (world displacement) and sX, sY (world peak widths). NaNs stay NaN."""

    di, dj = R["di"], R["dj"]
    sigma_i, sigma_j = R["sigma_i"], R["sigma_j"]
    H, W = di.shape
    ii, jj = np.mgrid[0:H, 0:W]
    x = (jj + col0).astype(float)           # pixel x = column
    y = (ii + row0).astype(float)           # pixel y = row

    X0, Y0 = calib_fun(x, y, v)             # where the particle is in frame A
    X1, Y1 = calib_fun(x + dj, y + di, v)   # where it is in frame B
    dX, dY = X1 - X0, Y1 - Y0               # exact for the nonlinear mapping

    # local Jacobian d(X,Y)/d(x,y), central differences
    Xp, Yp = calib_fun(x + h, y, v) 
    Xm, Ym = calib_fun(x - h, y, v)
    J11, J21 = (Xp - Xm)/(2*h), (Yp - Ym)/(2*h)      # d/dx (columns)
    
    Xp, Yp = calib_fun(x, y + h, v)
    Xm, Ym = calib_fun(x, y - h, v)
    J12, J22 = (Xp - Xm)/(2*h), (Yp - Ym)/(2*h)      # d/dy (rows)

    # peak is Gaussian with std sigma_j along x and sigma_i along y -> propagate
    sX = np.sqrt((J11*sigma_j)**2 + (J12*sigma_i)**2)
    sY = np.sqrt((J21*sigma_j)**2 + (J22*sigma_i)**2)

    return {**R, "X": X0, "Y": Y0, "dX": dX, "dY": dY, "sX": sX, "sY": sY}

#%%

run = 0

# mask = None
size = 30
block = 16
n_coarse = 500
min_fill = 0.25
rows = 32

min_peak = 0.0
outlier_tol = 3.0

method = 'gaussian' # 'gaussian' or 3pt
r = 7
margin = 12

path = main_folder + '/ctf_result(all_0).npz'

t1 = time()
if run:
    with h5py.File(main_folder+'/tiffs_data.h5', 'r') as f:
        A_stack = f['A_stack']
        B_stack = f['B_stack']
        
        H, W, n = A_stack.shape
        if mask is not None:
            mask = np.asarray(mask, bool)
    
        t1i = time()
        Cb, origin = coarse_blocks(A_stack, B_stack, A_mean, A_std, B_mean, B_std, mask, size, block, n_coarse, min_fill, rows)
        t2i = time()
        print(t2i-t1i)
        Pi, Pj, coarse = build_predictor(Cb, origin, H, W, size, block, outlier_tol, min_peak)
        t3i = time()
        print(t3i-t2i)
        out = fine_pass(A_stack, B_stack, A_mean, A_std, B_mean, B_std, mask, Pi, Pj, margin, r, rows, method)
        out_dic = dict(di=out[0], dj=out[1], sigma_i=out[2], sigma_j=out[3], peak=out[4], Pi=Pi, Pj=Pj, coarse=coarse)
        t4i = time()
        print(t4i-t3i)
        
    save_result(path, out_dic, size=size, block=block, n_coarse=n_coarse, margin=margin, method=method)

    
t2 = time()

print( t2 - t1 )

#%%
ss = 3

R1, params = load_result(main_folder + '/ctf_result(all_0).npz')
R = R1

fig, ax = plt.subplots(1,5, figsize=(20,5), layout='constrained'  )

ax[0].set_title('dj')
im = ax[0].imshow( nangauss( R['dj'], ss ) , vmax = 3, vmin=-3 )
# im = ax[0].imshow( nangauss( R['dj'], ss ) < -5 )
fig.colorbar(im, ax=ax[0])

ax[1].set_title('di')
im = ax[1].imshow( nangauss( R['di'], ss ), vmax = 20, vmin=-20 )
fig.colorbar(im, ax=ax[1])

ax[2].set_title('sigma_j')
im = ax[2].imshow( nangauss( R['sigma_j'], ss ) , vmax = 10, vmin=0 )
fig.colorbar(im, ax=ax[2])

ax[3].set_title('sigma_i')
im = ax[3].imshow( nangauss( R['sigma_i'], ss ) , vmax = 10, vmin=0 )
fig.colorbar(im, ax=ax[3])

ax[4].set_title('peak')
# im = ax[4].imshow( nangauss( R['peak'], ss ) ) # , vmax = 100, vmin=0 )
im = ax[4].imshow( np.isnan(R['di']) ) # , vmax = 100, vmin=0 )
fig.colorbar(im, ax=ax[4])

plt.show()


#%%


def remove_outliers(R, fil_vals, sizes, whichs = ['dj','di','sigma_j','sigma_i'] ):
    
    was = np.zeros_like( R['dj'] )
    
    nR = {}

    for which, fil_val, size in zip(whichs, fil_vals, sizes):
        arr = np.copy( R[which] )
        nanmask = np.isnan(arr)
        nanmask = ~remove_small_objects(~nanmask, min_size=100)
        arr[nanmask] = np.nan
    
        resu_med = ndi.generic_filter( arr, function=np.nanmedian, size=size, mode='reflect' )
        resu_med[nanmask] = np.nan
    
        aaa = (arr - resu_med ) / np.nanstd(resu_med)
        fil = (aaa > fil_val) + (aaa<-fil_val)
        was += fil
        
        arr[fil] = np.nan
        pots = np.where(fil)
        for i in tqdm(range(len(pots[0]))):
            x,y = pots[1][i], pots[0][i]
            n1,n2 = arr[y+1,x], arr[y-1,x]
            n3,n4 = arr[y,x+1], arr[y,x-1]
            arr[y,x] = np.nanmean( [n1,n2,n3,n4] )
            
        nR[which] = arr

    return was, nR
    



t1 = time()

# which = 'sigma_j'
# arr = R[which]
# nanmask = np.isnan(arr)
# nanmask = ~remove_small_objects(~nanmask, min_size=100)
# arr[nanmask] = np.nan

# resu_med = ndi.generic_filter( arr, function=np.nanmedian, size=6, mode='reflect' )
# # resu_std = ndi.generic_filter( arr, function=np.nanstd, size=5, mode='reflect' )
# resu_std = np.nanstd(arr)

# resu_med[nanmask] = np.nan

# fil_val = 9.

# print( np.nanstd(resu_med) )

# # aaa = (arr - resu_med ) / resu_std
# aaa = (arr - resu_med ) / np.nanstd(resu_med)

# fil = (aaa > fil_val) + (aaa<-fil_val)
# wa = np.where(fil)

if exp == 'f5':
    sizes = [5,5,6,5]
    fil_vals = [2.5,1,8.,6.]
# elif exp == 'f6':
elif exp in ['f6','f0']:
    sizes = [5,5,6,5]
    fil_vals = [2.5,1,8.,6.]

was, nR = remove_outliers(R, fil_vals, sizes, whichs = ['dj','di','sigma_j','sigma_i'] )

t2 = time()
print(t2-t1)

#%%


# which = 'dj'

# plt.figure()
# plt.imshow( R[which], vmax=1000 )#, vmax=20, vmin=-20)
# # plt.imshow(R[which], vmax=20, vmin=0)
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow( nR[which] )#, vmax=20, vmin=-20)
# # plt.imshow(nR[which], vmax=20, vmin=0)
# plt.colorbar()
# plt.show()


ss = 3

fig, ax = plt.subplots(1,4, figsize=(16,5), layout='constrained'  )

ax[0].set_title('dj')
im = ax[0].imshow( nangauss( nR['dj'], ss ) )#, vmax = 3, vmin=-3 )
fig.colorbar(im, ax=ax[0])

ax[1].set_title('di')
im = ax[1].imshow( nangauss( nR['di'], ss ) )#, vmax = 20, vmin=-20 )
fig.colorbar(im, ax=ax[1])

ax[2].set_title('sigma_j')
im = ax[2].imshow( nangauss( nR['sigma_j'], ss ) )#, vmax = 10, vmin=0 )
fig.colorbar(im, ax=ax[2])

ax[3].set_title('sigma_i')
im = ax[3].imshow( nangauss( nR['sigma_i'], ss ) )#, vmax = 10, vmin=0 )
fig.colorbar(im, ax=ax[3])

plt.show()

# plt.figure()
# plt.imshow( R[which], vmax=1000 )#, vmax=20, vmin=-20)
# # plt.imshow(R[which], vmax=20, vmin=0)
# plt.colorbar()
# plt.plot(wa[1], wa[0], 'r.')
# plt.show()

# plt.figure()
# plt.imshow( resu_med, vmax=1000 )#, vmax=20, vmin=-20)
# # plt.imshow(R[which], vmax=20, vmin=0)
# plt.colorbar()
# plt.plot(wa[1], wa[0], 'r.')
# plt.show()

#%%

dt = 0.006
ss = 3

fig, ax = plt.subplots(1,2, figsize=(8,5), layout='constrained')
im = ax[0].imshow( nangauss( nR['dj'], ss ) ) 
fig.colorbar(im, ax=ax[0])

im = ax[1].imshow( nangauss( nR['di'], ss ) ) 
fig.colorbar(im, ax=ax[1])
plt.show()

t1 = time()
row0 = 0 if nlim[0].start is None else nlim[0].start
col0 = 0 if nlim[1].start is None else nlim[1].start
print(row0, col0)

Rw = disp_to_world(nR, calib_fun2, la2.x, row0=row0, col0=col0, h=0.5)
t2 = time()
print(t2-t1)

# Rw['dX'], Rw['dY'], Rw['sX'], Rw['sY'], Rw['X'], Rw['Y']


fig, ax = plt.subplots(1,2, figsize=(8,5), layout='constrained')

ax[0].set_title( 'Vx' )
im = ax[0].imshow( nangauss( Rw['dX'] / dt, ss ) ) 

# im = ax[0].imshow( nangauss( Rw['sX'] / dt, ss ) ) 
# im = ax[0].imshow( nangauss( Rw['X'], ss ) ) 
fig.colorbar(im, ax=ax[0])

ax[1].set_title( 'Vy' )
im = ax[1].imshow( nangauss( Rw['dY'] / dt, ss ) ) 

# im = ax[1].imshow( nangauss( Rw['sY'] / dt, ss ) ) 
# im = ax[1].imshow( nangauss( Rw['Y'], ss ) ) 
fig.colorbar(im, ax=ax[1])
plt.show()


#%%

# Trying to plot


def draw_arrows(ax, x, y, u, v, arrow_scale,
                head_frac=0.35,      # head length as a fraction of arrow length
                head_aspect=0.6,     # head full width / head length
                shaft_frac=0.06,     # shaft full width / arrow length
                min_head=0.4,        # smallest head length (data units)
                min_shaft=0.12,      # smallest shaft width (data units)
                pivot='tail', **kwargs):
    speed = np.hypot(u, v)
    L = speed * arrow_scale
    ok = L > 0
    x, y, u, v, speed, L = x[ok], y[ok], u[ok], v[ok], speed[ok], L[ok]

    ex, ey = u / speed, v / speed          # unit direction
    nx, ny = -ey, ex                       # unit normal

    hl = np.minimum(np.maximum(head_frac * L, min_head), L)   # head length
    hw = 0.5 * head_aspect * hl                               # head half-width
    sw = 0.5 * np.maximum(shaft_frac * L, min_shaft)          # shaft half-width
    sw = np.minimum(sw, hw)                                   # shaft never wider than head

    if pivot == 'middle':
        x0, y0 = x - 0.5 * L * ex, y - 0.5 * L * ey
    else:
        x0, y0 = x, y

    bx, by = x0 + (L - hl) * ex, y0 + (L - hl) * ey   # head base centre
    tx, ty = x0 + L * ex, y0 + L * ey                 # tip

    verts = np.stack([
        np.column_stack([x0 + sw * nx, y0 + sw * ny]),
        np.column_stack([bx + sw * nx, by + sw * ny]),
        np.column_stack([bx + hw * nx, by + hw * ny]),
        np.column_stack([tx, ty]),
        np.column_stack([bx - hw * nx, by - hw * ny]),
        np.column_stack([bx - sw * nx, by - sw * ny]),
        np.column_stack([x0 - sw * nx, y0 - sw * ny]),
    ], axis=1)                                         # (N, 7, 2)

    pc = PolyCollection(verts, **kwargs)
    ax.add_collection(pc)
    return pc


# def adaptive_quiver(ax, X, Y, U, V, arrow_scale=None, spacing_factor=1.2,
#                     min_spacing=None, max_arrow_len=None, **kwargs):
#     """
#     Quiver with locally adaptive density.

#     arrow_scale    : data-units of arrow length per unit of velocity
#                      (arrow length = speed * arrow_scale). Single global
#                      factor, so arrows stay to scale.
#     spacing_factor : required distance between arrows, in units of the
#                      longer of the two arrows' drawn lengths.
#     min_spacing    : floor on spacing (data units), e.g. grid pitch.
#     max_arrow_len  : if given, sets arrow_scale so the fastest arrow has
#                      this length (in data units, e.g. mm).
#     """
#     speed = np.hypot(U, V)
#     valid = np.isfinite(speed) & np.isfinite(U) & np.isfinite(V)

#     if arrow_scale is None:
#         if max_arrow_len is None:
#             raise ValueError("give arrow_scale or max_arrow_len")
#         arrow_scale = max_arrow_len / np.nanmax(speed)

#     length = speed * arrow_scale          # drawn length in data units

#     if min_spacing is None:
#         # grid pitch
#         min_spacing = np.nanmedian(np.abs(np.diff(X[0, :])))

#     # Candidates sorted by speed (fastest first): fast arrows claim space first
#     idx = np.argwhere(valid)
#     order = np.argsort(-speed[valid])
#     idx = idx[order]

#     pts = np.column_stack([X[valid], Y[valid]])[order]
#     lens = length[valid][order]

#     selected = []
#     sel_pts = np.empty((0, 2))
#     sel_len = np.empty(0)

#     # Simple greedy selection using a KD-tree of accepted points,
#     # rebuilt in chunks for speed.
#     accepted_pts = []
#     accepted_len = []
#     tree = None
#     pending = 0
#     rebuild_every = 50

#     for k in range(len(pts)):
#         p, l = pts[k], lens[k]
#         need = max(min_spacing, spacing_factor * l)

#         ok = True
#         # check against tree (accepted before last rebuild)
#         if tree is not None:
#             r = max(need, spacing_factor * max_len_so_far)
#             nb = tree.query_ball_point(p, r)
#             for j in nb:
#                 d = np.hypot(*(p - tree.data[j]))
#                 if d < max(need, spacing_factor * tree_len[j]):
#                     ok = False
#                     break
#         # check against recent, not-yet-in-tree points
#         if ok:
#             for q, ql in zip(accepted_pts[len(accepted_pts) - pending:],
#                              accepted_len[len(accepted_len) - pending:]):
#                 if np.hypot(*(p - q)) < max(need, spacing_factor * ql):
#                     ok = False
#                     break

#         if ok:
#             selected.append(k)
#             accepted_pts.append(p)
#             accepted_len.append(l)
#             pending += 1
#             if pending >= rebuild_every:
#                 tree = cKDTree(np.array(accepted_pts))
#                 tree_len = np.array(accepted_len)
#                 max_len_so_far = tree_len.max()
#                 pending = 0

#     sel = np.array(selected, dtype=int)
#     iy, ix = idx[sel, 0], idx[sel, 1]

#     mask = np.zeros_like(speed, dtype=bool)
#     mask[iy, ix] = True

#     # angles='xy', scale_units='xy' with scale=1/arrow_scale makes the arrow
#     # length in DATA units = speed * arrow_scale (true to scale on screen)

#     q = ax.quiver(X[mask], Y[mask], U[mask], V[mask], angles='xy', scale_units='xy', scale=1.0 / arrow_scale, **kwargs)
    
#     # draw_arrows(ax, X[mask], Y[mask], U[mask], V[mask], arrow_scale,
#     #         head_frac=0.3,      # head length = 30% of arrow length
#     #         head_aspect=0.6,    # head width = 60% of head length
#     #         shaft_frac=0.05,    # shaft width = 5% of arrow length
#     #         min_head=0,         # no floor: heads scale purely with length
#     #         min_shaft=0,
#     #         facecolor='k', edgecolor='none')
#     # q = None
    
#     return q, mask

def adaptive_quiver(ax, X, Y, U, V, delta_t, gain=1.0, spacing_factor=1.0,
                    min_spacing=1.0, **kwargs):
    """
    Arrow length (data units) = speed * delta_t * gain  (scale_units='xy', angles='xy').
    Required spacing between arrows = max(min_spacing, spacing_factor * longer arrow length).
    Fast arrows are placed first; NaNs are skipped.
    """
    arrow_scale = delta_t * gain
    speed = np.hypot(U, V)
    valid = np.isfinite(speed) & np.isfinite(U) & np.isfinite(V)
    length = speed * arrow_scale

    iy, ix = np.nonzero(valid)
    order = np.argsort(-speed[iy, ix])
    iy, ix = iy[order], ix[order]
    px, py, pl = X[iy, ix], Y[iy, ix], length[iy, ix]

    sx = np.empty(len(px)); sy = np.empty(len(px)); sl = np.empty(len(px))
    keep = np.zeros(len(px), dtype=bool)
    k = 0
    for i in range(len(px)):
        if k:
            need = np.maximum(min_spacing, spacing_factor * np.maximum(pl[i], sl[:k]))
            if np.any(np.hypot(sx[:k] - px[i], sy[:k] - py[i]) < need):
                continue
        sx[k], sy[k], sl[k] = px[i], py[i], pl[i]
        keep[i] = True
        k += 1

    mask = np.zeros(speed.shape, dtype=bool)
    mask[iy[keep], ix[keep]] = True

    q = ax.quiver(X[mask], Y[mask], U[mask], V[mask],
                  angles='xy', scale_units='xy', scale=1.0 / arrow_scale, **kwargs)
    return q, mask

def grad_curvilinear(F, X, Y):
    """Return dF/dx, dF/dy on a grid where X, Y are 2D arrays of the point positions."""
    Fi, Fj = np.gradient(F, axis=(0, 1))          # d/d(row index), d/d(col index)
    Xi, Xj = np.gradient(X, axis=(0, 1))
    Yi, Yj = np.gradient(Y, axis=(0, 1))
    det = Xj * Yi - Xi * Yj                       # Jacobian determinant
    dFdx = (Fj * Yi - Fi * Yj) / det
    dFdy = (Fi * Xj - Fj * Xi) / det

    return dFdy, dFdx

#%%

edge_x, edge_y = calib_fun2(sub_edge+col0, aro, la2.x)

extent = [ np.min(Rw['X']), np.max(Rw['X']), np.min(Rw['Y']), np.max(Rw['Y']) ]

Xg, Yg = Rw['X'], Rw['Y']
vx, vy = Rw['dX'] / dt, Rw['dY'] / dt

# ss = (7,3)
ss = (5,2)
gvy, gvx = nangauss(vy , ss), nangauss(vx , ss)
vy_dy, vy_dx = grad_curvilinear(gvy, Xg, Yg)
vx_dy, vx_dx = grad_curvilinear(gvy, Xg, Yg)

vort = vy_dx - vx_dy


# vmax = np.nanmax(np.abs(vort))
# norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

# cmap = plt.cm.RdBu_r.copy()
# cmap.set_bad('gray')
# vort_masked = np.ma.masked_invalid(vort)


# fig, ax = plt.subplots(figsize=(5, 6))
# im = ax.imshow( vort_masked, cmap=cmap, norm=norm )
# cbar = plt.colorbar(im, ax=ax, label='Vorticity')
# plt.show()

# #%%

vmax = np.nanmax(np.abs(vort))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


cmap = plt.cm.RdBu_r.copy()
cmap.set_bad('gray')

vort_masked = np.ma.masked_invalid(vort)
vy_masked = np.ma.masked_invalid(gvy)

fig, ax = plt.subplots(figsize=(5, 6))

plt.plot(edge_x, edge_y, 'r-')

# im = plt.imshow( vy, origin='lower', extent=extent, aspect='auto', cmap=cmap, norm=norm )
# im = plt.imshow( vort_masked, extent=extent, aspect='auto', cmap=cmap, norm=norm )
# cbar = plt.colorbar(im, ax=ax, label='Vorticity (1/s)')

pc = ax.pcolormesh(Xg, Yg, vort_masked, cmap=cmap, norm=norm, shading='nearest', rasterized=True)
fig.colorbar(pc, ax=ax, label='Vorticity (1/s)')

step = 10
# q = ax.quiver( Xg[::step], Yg[::step], gvx[::step], gvy[::step], color='black', scale=5e3, pivot='middle' )
# q = adaptive_quiver(ax, Xg, Yg, gvx, gvy, min_spacing=1, max_spacing=5, color='black', scale=5e3, pivot='middle')

# q, sel = adaptive_quiver( ax, Xg, Yg, gvx, gvy,
#         arrow_scale=None,
#         max_arrow_len=20,          
#         spacing_factor=.4,       
#         min_spacing=3.,
#         color='black', width=0.004, headwidth=4, headlength=4,
#         pivot='tail')

dt_arrow = .025      # seconds of displacement represented by a velocity unit
q, sel = adaptive_quiver(ax, Xg, Yg, gvx, gvy,
                         delta_t=dt_arrow, gain=1.0,
                         spacing_factor=0.4, min_spacing=3.0,
                         color='black', width=0.004, headwidth=4, headlength=4,
                         pivot='tail', zorder=4)

ax.quiverkey(q, 0.85, 1.03, 400.0, '400 mm/s', labelpos='E')



# arrow_scale = 10.0 / np.nanmax(np.hypot(gvx, gvy))     # fastest arrow = 6 mm long (tune)

# mask_n = select_arrows(Xg, Yg, gvx, gvy, arrow_scale,
#                       along_factor=1.05, perp_gap=1.5, min_gap=0.1)

# draw_open_arrows(ax, Xg[mask_n], Yg[mask_n], gvx[mask_n], gvy[mask_n], arrow_scale,
#                   colors='k', linewidths=0.6)

# # reference arrow, like the "0.5 m/s" key in the paper
# ref = 0.5                                              # same units as gvx
# draw_open_arrows(ax, np.array([5.0]), np.array([-22.0]),
#                   np.array([ref]), np.array([0.0]), arrow_scale, colors='k', linewidths=0.6)
# ax.text(5.0, -20.5, '0.5 m/s', fontsize=8)



ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')

ax.axis('equal')

filename = main_folder + '/Figs/ptv.pdf' 
# plt.savefig(filename, dpi=200, bbox_inches='tight')


plt.show()

#%%

from scipy.interpolate import griddata

# # --- region to the right of the edge (edge_x as a function of Y) ---
# o = np.argsort(edge_y)
# edge_x_at = np.interp(Yg, np.asarray(edge_y)[o], np.asarray(edge_x)[o],
#                       left=np.nan, right=np.nan)
# beyond = Xg > edge_x_at            # False where the edge is undefined (NaN compares False)

# # --- fill NaN holes by interpolation ---
# def fill_holes(F, X, Y):
#     F = F.copy()
#     good = np.isfinite(F)
#     hole = ~good
#     if hole.any():
#         pts = np.column_stack([X[good], Y[good]])
#         F[hole] = griddata(pts, F[good], (X[hole], Y[hole]), method='linear')
#         left = ~np.isfinite(F)                      # outside the convex hull
#         if left.any():
#             F[left] = griddata(pts, F[good], (X[left], Y[left]), method='nearest')
#     return F

# gvx_f = fill_holes(gvx, Xg, Yg)
# gvy_f = fill_holes(gvy, Xg, Yg)

# dvy_dy, dvy_dx = grad_curvilinear(gvy_f, Xg, Yg)
# dvx_dy, dvx_dx = grad_curvilinear(gvx_f, Xg, Yg)
# vort = dvy_dx - dvx_dy

# # gray only to the right of the edge
# vort_masked = np.ma.masked_where(beyond, vort)

# # arrows: use the original data, dropping anything beyond the edge
# gvx_a = np.where(beyond, np.nan, gvx)
# gvy_a = np.where(beyond, np.nan, gvy)

# fig, ax = plt.subplots(figsize=(5, 6))
# ax.set_facecolor('gray')
# pc = ax.pcolormesh(Xg, Yg, vort_masked, cmap=cmap, norm=norm,
#                    shading='nearest', rasterized=True)
# fig.colorbar(pc, ax=ax, label='Vorticity (1/s)')
# ax.plot(edge_x, edge_y, 'r-', lw=1, zorder=5)

# q, sel = adaptive_quiver(ax, Xg, Yg, gvx_a, gvy_a, 
#                          arrow_scale=None,
#                          max_arrow_len=20,          
#                          spacing_factor=.4,       
#                          min_spacing=3.,
#                          color='black', width=0.004, headwidth=4, headlength=4,
#                          pivot='tail')  


o = np.argsort(edge_y)
ey, ex = np.asarray(edge_y)[o], np.asarray(edge_x)[o]

# velocity/vorticity exist only to the left of the edge (cells outside the edge's Y range are untouched)
edge_x_at = np.interp(Yg, ey, ex, left=np.nan, right=np.nan)
beyond = Xg > edge_x_at                           # False where the edge is undefined

vort_masked = np.ma.masked_invalid(np.where(beyond, np.nan, vort))
gvx_a = np.where(beyond, np.nan, gvx)
gvy_a = np.where(beyond, np.nan, gvy)

fig, ax = plt.subplots(figsize=(5, 6))
ax.set_facecolor('white')

pc = ax.pcolormesh(Xg, Yg, vort_masked, cmap=cmap, norm=norm,
                   shading='nearest', rasterized=True, zorder=1)
fig.colorbar(pc, ax=ax, label='Vorticity (1/s)')

xmax = np.nanmax(Xg)
ax.fill_betweenx(ey, ex, xmax, color='gray', lw=0, zorder=2)   # only between min(edge_y) and max(edge_y)
ax.plot(edge_x, edge_y, 'r-', lw=1, zorder=5)

if exp == 'f6':
    dt_arrow = .02      # seconds of displacement represented by a velocity unit
    q, sel = adaptive_quiver(ax, Xg, Yg, gvx, gvy,
                             delta_t=dt_arrow, gain=1.0,
                             spacing_factor=0.6, min_spacing=2.0,
                             color='black', width=0.006, headwidth=4, headlength=4,
                             pivot='tail', zorder=4)
elif exp == 'f0':
    dt_arrow = .02      # seconds of displacement represented by a velocity unit
    q, sel = adaptive_quiver(ax, Xg, Yg, gvx, gvy,
                             delta_t=dt_arrow, gain=1.0,
                             spacing_factor=0.6, min_spacing=2.0,
                             color='black', width=0.004, headwidth=4, headlength=4,
                             pivot='tail', zorder=4)

ax.quiverkey(q, 0.85, 1.03, 400.0, '400 mm/s', labelpos='E')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')

if exp == 'f6':
    ax.set_xlim(25,135)
    ax.set_ylim(-125,50)
elif exp == 'f5':
    ax.set_xlim(35,87)
    ax.set_ylim(-75,100)
elif exp == 'f0':
    ax.set_xlim(20,130)
    ax.set_ylim(-125,50)

# ax.axis('equal', adjustable='box')
ax.set_aspect('equal', adjustable='box')

filename = main_folder + '/Figs/sp-ptv.pdf' 
plt.savefig(filename, dpi=200, bbox_inches='tight')


plt.show()

#%%

def translate_slice(recorte, slim):
    return tuple(
        slice( (s.start or 0) - (n.start or 0),
                (s.stop - (n.start or 0)) if s.stop is not None else None,
                s.step )
        for s, n in zip(recorte, slim) )

i = 100

img1 = tifffile.imread(files[i])
img2 = tifffile.imread(files[i+1])

recorte = np.s_[100:1000,600:919]

plt.figure()
plt.imshow( img1[recorte], cmap='gray' )

plt.imshow( img2[recorte], cmap='viridis', alpha=0.3)

# recorte_nlim = np.s_[ recorte[0].start:recorte[0].stop, recorte[1].start - 592:recorte[1].stop - 592 ]

stepx, stepy = 5, 10
allstep = np.s_[::stepy,::stepx]
ny, nx = img1[recorte].shape
uu, vv = np.meshgrid(np.arange(nx), np.arange(ny))
recorte_nlim = translate_slice(recorte, nlim)
plt.quiver( uu[allstep], vv[allstep], 
           nR['dj'][recorte_nlim][allstep], nR['di'][recorte_nlim][allstep], 
           color='red', scale=1, scale_units='xy', angles='xy'  )


ss = (5,3)
ny, nx = img1[recorte].shape
uu, vv = np.meshgrid(np.arange(nx), np.arange(ny))

# ggvx = ndi.gaussian_filter(  np.where( ~np.isnan(nR['dj']), nR['dj'], 0),  ss)
# ggvy = ndi.gaussian_filter(  np.where( ~np.isnan(nR['di']), nR['di'], 0),  ss) 
ggvx, ggvy = nangauss(nR['dj'], ss), nangauss(nR['di'], ss)

recorte_nlim = translate_slice(recorte, nlim)


plt.quiver( uu[allstep], vv[allstep], 
           ggvx[recorte_nlim][allstep], ggvy[recorte_nlim][allstep], 
           color='yellow', scale=1, scale_units='xy', angles='xy'  )

plt.plot(sub_edge[recorte_nlim[0]] - recorte_nlim[1].start, aro[recorte_nlim[0]] - recorte_nlim[0].start, 'r-')

plt.axis('equal')

plt.show()

#%%

plt.figure()
plt.imshow( mask )
plt.show()


#%%
# Look at predictor
run = 1

# mask = None
size = 30
block = 16
n_coarse = 500
min_fill = 0.25
rows = 32

min_peak = 0.0
outlier_tol = 3.0

method = 'gaussian' # 'gaussian' or 3pt
r = 7
margin = 12

if run:
    with h5py.File(main_folder+'/tiffs_data.h5', 'r') as f:
        A_stack = f['A_stack']
        B_stack = f['B_stack']
        
        H, W, n = A_stack.shape
        if mask is not None:
            mask = np.asarray(mask, bool)
    
        t1i = time()
        Cb, origin = coarse_blocks(A_stack, B_stack, A_mean, A_std, B_mean, B_std, mask, size, block, n_coarse, min_fill, rows)
        t2i = time()
        print(t2i-t1i)
        Pi, Pj, coarse = build_predictor(Cb, origin, H, W, size, block, outlier_tol, min_peak)
        t3i = time()
        print(t3i-t2i)


#%%


def plot_predictor(Pi, Pj, diagnostics=None, mask=None):

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Validate mask
    if mask is not None:
        mask = np.asarray(mask)

        if mask.shape != Pi.shape:
            raise ValueError("Mask must have the same shape as Pi and Pj.")

        if Pj.shape != Pi.shape:
            raise ValueError("Pi and Pj must have the same shape.")

        mask = mask.astype(bool)

    # Apply mask to displacement fields
    if mask is not None:
        Pi_plot = np.where(mask, Pi, np.nan)
        Pj_plot = np.where(mask, Pj, np.nan)
    else:
        Pi_plot = Pi
        Pj_plot = Pj

    # 1. Horizontal displacement
    im0 = axes[0, 0].imshow(Pj_plot, cmap="coolwarm")
    axes[0, 0].set_title("Horizontal displacement (Pj)")
    plt.colorbar(im0, ax=axes[0, 0], label="Pixels")

    # 2. Vertical displacement
    im1 = axes[0, 1].imshow(Pi_plot, cmap="coolwarm")
    axes[0, 1].set_title("Vertical displacement (Pi)")
    plt.colorbar(im1, ax=axes[0, 1], label="Pixels")

    # 3. Displacement magnitude
    magnitude = np.sqrt(Pi**2 + Pj**2)

    if mask is not None:
        magnitude = np.where(mask, magnitude, np.nan)

    im2 = axes[0, 2].imshow(magnitude, cmap="viridis")
    axes[0, 2].set_title("Displacement magnitude")
    plt.colorbar(im2, ax=axes[0, 2], label="Pixels")

    # 4. Displacement vectors
    step = max(1, min(Pi.shape) // 30)

    Y, X = np.mgrid[0:Pi.shape[0]:step,
                    0:Pi.shape[1]:step]

    Pi_vec = Pi[::step, ::step]
    Pj_vec = Pj[::step, ::step]

    if mask is not None:
        mask_vec = mask[::step, ::step]
        Pi_vec = np.where(mask_vec, Pi_vec, np.nan)
        Pj_vec = np.where(mask_vec, Pj_vec, np.nan)

    axes[1, 0].quiver(
        X, Y,
        Pj_vec,
        Pi_vec,
        angles="xy",
        scale_units="xy",
        scale=1
    )

    axes[1, 0].invert_yaxis()
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_title("Displacement vectors")

    # 5. Optional coarse-level diagnostics
    if diagnostics is not None:

        Pi_b, Pj_b, peak, bad = diagnostics

        im3 = axes[1, 1].imshow(peak, cmap="viridis")
        axes[1, 1].set_title("Maximum coarse correlation")
        plt.colorbar(im3, ax=axes[1, 1])

        im4 = axes[1, 2].imshow(
            bad, cmap="gray_r", vmin=0, vmax=1
        )
        axes[1, 2].set_title("Rejected coarse blocks")
        plt.colorbar(im4, ax=axes[1, 2], ticks=[0, 1])

    else:
        axes[1, 1].axis("off")
        axes[1, 2].axis("off")

    # 6. Axis labels
    for ax in axes.flat:
        if ax.has_data():
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.show()



plot_predictor(Pi, Pj, coarse, mask = mask)



#%%
ss = (3,7)
ny, nx = nR['dj'].shape
uu, vv = np.meshgrid(np.arange(nx), np.arange(ny))

gvx, gvy = nangauss(nR['dj'], ss), nangauss(nR['di'], ss)

vxdy, vxdx = np.gradient(gvx)
vydy, vydx = np.gradient(gvy)
vort = vydx - vxdy

stepx, stepy = 20, 20
plt.figure()

# plt.quiver(uu[::stepx], vv[::stepy], nR['dj'][::stepx], nR['di'][::stepy], pivot='tail', scale=100 )
# plt.quiver(uu[::stepx], vv[::stepy], gvx[::stepx], gvy[::stepy], pivot='tail', scale=100 )
# plt.quiver(uu[::stepx], vv[::stepy], nR['di'][::stepx] * 0, nR['di'][::stepy], pivot='tail', scale=100 )

# plt.imshow( nR['dj'] )
# plt.imshow( nR['di'] )
plt.imshow( vort )
plt.plot( sub_edge, aro, 'r-' )

plt.axis('equal')
plt.show()

#%%


ss = 0
step = 16
ny, nx = Pj.shape
uu, vv = np.meshgrid(np.arange(nx), np.arange(ny))

gvx, gvy = nangauss(Pj[::step], ss), nangauss(Pi[::step], ss)

vxdy, vxdx = np.gradient(gvx*1.)
vydy, vydx = np.gradient(gvy*1.)
vort = vydx - vxdy

stepx, stepy = 16, 16
plt.figure()

# plt.quiver(uu[::stepx], vv[::stepy], nR['dj'][::stepx], nR['di'][::stepy], pivot='tail', scale=100 )

# plt.quiver(uu[::stepx], vv[::stepy], gvx, gvy, pivot='tail', scale=1000 )


# plt.quiver(uu[::stepx], vv[::stepy], nR['di'][::stepx] * 0, nR['di'][::stepy], pivot='tail', scale=100 )

# plt.imshow( nR['dj'] )
# plt.imshow( nR['di'] )
# plt.imshow( vort, alpha=1)  
plt.imshow( vort, alpha=1, extent=(0,nx,ny,0))  

plt.plot( sub_edge, aro, 'r-' )

plt.axis('equal')
plt.show()

#%%

# step = max(1, min(Pi.shape) // 30)
step = 16

Y, X = np.mgrid[0:Pi.shape[0]:step,
                0:Pi.shape[1]:step]

Pi_vec = Pi[::step, ::step]
Pj_vec = Pj[::step, ::step]

fig, ax = plt.subplots(1,1)

if mask is not None:
    mask_vec = mask[::step, ::step]
    Pi_vec = np.where(mask_vec, Pi_vec, np.nan)
    Pj_vec = np.where(mask_vec, Pj_vec, np.nan)

ax.quiver(
    X, Y,
    Pj_vec,
    Pi_vec,
    angles="xy",
    scale_units="xy",
    scale=.5 )

plt.plot( sub_edge, aro, 'r-' )

ax.invert_yaxis()
ax.set_aspect("equal")
plt.show()


#%%
# =============================================================================
# One pixel corr
# =============================================================================
show_bar = 1

rango = 2000
# rango = len(files)

all_f = range(0,rango,2)

A_stack, B_stack = np.zeros((1024,1280,rango//2), dtype='uint16'), np.zeros((1024,1280,rango//2), dtype='uint16')
A_stack, B_stack = A_stack[nlim], B_stack[nlim]


for j, i in enumerate(tqdm(all_f, disable=not show_bar)):
    img1 = tifffile.imread(files[i])
    img2 = tifffile.imread(files[i+1])

    A_stack[:,:,j] = img1[nlim]
    B_stack[:,:,j] = img2[nlim]


#%%
size = 35

t1 = time()

r, c = 840, 300

line_A = A_stack[r,c,:]

corr = np.zeros((2*size+1, 2*size+1))
for i in tqdm(range(-size, size+1)):
    for j in range(-size, size+1):

        line_B = B_stack[r+i, c+j, :]

        corr[i+size, j+size] = np.corrcoef(line_A, line_B)[0,1]

center, std = fit_corr_gaussian(corr, r=5)

t2 = time()

print()
print(center)
print(std)


plt.figure()
plt.imshow( A_stack[:,:,0] )
plt.imshow( B_stack[:,:,0], alpha=0.2 )
plt.show()


plt.figure()
# plt.plot( line_A )
# plt.plot( line_B )
plt.imshow( corr, extent=(-size,size,size,-size) )

plt.plot(center[1],center[0], 'r.')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

I, J = np.meshgrid(
    np.arange(-size, size+1),
    np.arange(-size, size+1)
)

ax.plot_surface(J, I, corr, cmap='viridis', edgecolor='none')

ax.set_xlabel('Column offset (j)')
ax.set_ylabel('Row offset (i)')
ax.set_zlabel('Correlation')

plt.show()


#%%


#%%

#parallel tiff reading. For 1000 ims (rango 2000) takes ~35 seconds less

#%%






#%%





#%%


#%%

#%%

a = np.random.random((30))


alen = len(a)
mm = 0
mm2 = 0
for i in range(len(a)):
    mm += a[i] / alen
    mm2 += a[i]**2 / alen
    
    

sd = np.sqrt(mm2 - mm**2)

print(np.mean(a), mm)
print(np.std(a), sd)


#%%


alen = len(a)
mm = 0
var = 0
for i in range(len(a)):
    delta = a[i] - mm
    mm += delta / (i+1)
    var += delta * (a[i] - mm)
sd = np.sqrt(var / alen)
    
print(np.mean(a), mm)
print(np.std(a), sd)
    

#%%





#%%





#%%

import h5py
from concurrent.futures import ThreadPoolExecutor

show_bar = 1
rango = len(files)


all_f = range(0, rango, 2)
n = len(all_f)

batch_size = 64
workers = 4

# Determine cropped image dimensions
img0 = tifffile.imread(files[0])[nlim]
H, W = img0.shape
dtype = img0.dtype



def read_pair(i):
    img1 = tifffile.imread(files[i])[nlim]
    img2 = tifffile.imread(files[i+1])[nlim]
    return img1, img2


with h5py.File(main_folder+'/tiffs_data.h5', 'w') as f:

    # Create datasets
    chunks = (min(8,H), min(8,W), min(64,n))

    A_stack = f.create_dataset( 'A_stack', shape=(H,W,n), dtype=dtype, chunks=chunks )

    B_stack = f.create_dataset( 'B_stack', shape=(H,W,n), dtype=dtype, chunks=chunks )

    # Read and write in batches
    with ThreadPoolExecutor(max_workers=workers) as executor:

        for start in tqdm(range(0, n, batch_size), disable=not show_bar):

            stop = min(start + batch_size, n)
            indices = range(2*start, 2*stop, 2)

            A_batch = np.empty((stop-start,H,W), dtype=dtype)
            B_batch = np.empty((stop-start,H,W), dtype=dtype)

            for k, (img1,img2) in enumerate(executor.map(read_pair, indices)):
                A_batch[k] = img1
                B_batch[k] = img2

            A_stack[:,:,start:stop] = A_batch.transpose(1,2,0)
            B_stack[:,:,start:stop] = B_batch.transpose(1,2,0)




#%%
size = 35
r, c = 840, 300

with h5py.File(main_folder+'/tiffs_data.h5', 'r') as f:
    A_stack = f['A_stack']
    B_stack = f['B_stack']

    line_A = A_stack[r,c,:]
    
    corr = np.zeros((2*size+1, 2*size+1))
    for i in tqdm(range(-size, size+1)):
        for j in range(-size, size+1):
    
            line_B = B_stack[r+i, c+j, :]
            corr[i+size, j+size] = np.corrcoef(line_A, line_B)[0,1]

center, std = fit_corr_gaussian(corr, r=5)


#%%

plt.figure()
plt.imshow( corr, extent=(-size,size,size,-size) )
plt.plot(center[1],center[0], 'r.')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

I, J = np.meshgrid(
    np.arange(-size, size+1),
    np.arange(-size, size+1)
)

ax.plot_surface(J, I, corr, cmap='viridis', edgecolor='none')

ax.set_xlabel('Column offset (j)')
ax.set_ylabel('Row offset (i)')
ax.set_zlabel('Correlation')

plt.show()



#%%


#%%






#%%





