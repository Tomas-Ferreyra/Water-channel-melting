#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 14:46:17 2025

@author: tomasferreyrahauchar
"""

import imageio.v2 as imageio
import cv2
# import imageio.v3 as iio
from tqdm import tqdm
from time import time, sleep
import h5py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import linregress
from scipy.optimize import least_squares, minimize, curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.linalg import pinv
from scipy.interpolate import griddata


from skimage.color import rgb2gray
from skimage.filters import gaussian, try_all_threshold, rank, butterworth, frangi, sato
from skimage.morphology import remove_small_objects, disk, binary_erosion, binary_closing
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.feature import blob_log
from skimage.util import img_as_ubyte


def _project_error(sol, xp, yp, rp, v, coord):
    if coord == 0: X, Y, Z = rp, sol[0], sol[1]
    elif coord == 1: X, Y, Z = sol[0], rp, sol[1]
    else: X, Y, Z = sol[0], sol[1], rp

    xtop = (v[0]*X + v[1]*Y + v[2]*Z + v[3]
            + v[4]*X*X + v[5]*Y*Y + v[6]*X*Y)

    ytop = (v[7]*X + v[8]*Y + v[9]*Z + v[10]
            + v[11]*X*X + v[12]*Y*Y + v[13]*X*Y)

    bot = (v[14]*X + v[15]*Y + v[16]*Z + 1
           + v[17]*X*X + v[18]*Y*Y + v[19]*X*Y)

    return np.array([xtop / bot - xp, ytop / bot - yp ])

def calib_invzl_fast(v, points, coord, dis_bar=False, method='SLSQP', x0s=[[50,300]]):

    N = points.shape[0]
    r1 = np.empty(N)
    r2 = np.empty(N)

    _ls = least_squares
    _proj = _project_error

    for i in tqdm(range(N), disable=dis_bar):
        xp, yp, rp = points[i]
        x0 = x0s[i % len(x0s)]

        res = _ls(_proj, x0, args=(xp, yp, rp, v, coord), method=method)
        r1[i], r2[i] = res.x

    return r1, r2



def calib_invzl(v, points, coord, dis_bar=False, method='SLSQP', x0s=[[50,300]]):
    """
    Parameters
    ----------
    v : list
        list of coefficients from fit.
    points : (N,3) array
        First 2 coordinates should be pixel position of point (x,y)
        The last coordinate should be the one real world position of that point (x,y,or z depending on coord parameter).
    coord : 0,1 or 2 int
        which real world coordinate is not solved for.
    x0s : (M,2) array, optional
        Initial guess for minimize. M=1 or M=N. If N=1 then it would be use as initial values for the points. M=N is for provinding initial values for all points
        . The default is [[50,300]].
    dis_bar: bool
        False for showing the progression bar. True for not showing it
    method: 'SLSQP' or 'Nelder-Mead'
        Which method to use for the minimization. (Could technically use any method from scipy.optimize.minize, but this two have been teste to work)
        SLSQP is faster, Nelder-Mead is more trustworthy (I think)

    Returns
    -------
    X : array
        real world x-position of points.
    Y : array
        real world y-position of points.
    """
    N,dim = np.shape(points)
    if dim != 3: print('Second dimension length should be 3')
    
    r1, r2 = np.zeros(N), np.zeros(N)

    def func(bol, impos):
        xtop = v[0] *bol[0] + v[1] *bol[1] + v[2] *bol[2] + v[3]  + v[4] *bol[0]**2 + v[5] *bol[1]**2 + v[6] *bol[0]*bol[1]
        ytop = v[7] *bol[0] + v[8] *bol[1] + v[9] *bol[2] + v[10] + v[11]*bol[0]**2 + v[12]*bol[1]**2 + v[13]*bol[0]*bol[1]
        bot  = v[14]*bol[0] + v[15]*bol[1] + v[16]*bol[2] + 1     + v[17]*bol[0]**2 + v[18]*bol[1]**2 + v[19]*bol[0]*bol[1]
        eq1, eq2 = xtop/bot - impos[0], ytop/bot - impos[1]
        return [eq1**2,eq2**2]
    
    if coord == 0:
        fun = lambda val,im_x,im_y,real_pos: func( [real_pos,val[0],val[1]], [im_x,im_y] )  
    elif coord == 1:
        fun = lambda val,im_x,im_y,real_pos: func( [val[0],real_pos,val[1]], [im_x,im_y] )  
    elif coord == 2:
        fun = lambda val,im_x,im_y,real_pos: func( [val[0],val[1],real_pos], [im_x,im_y] )  
    
    for i in tqdm(range(N), disable=dis_bar):
        xp, yp, rp = points[i]
        x0 = x0s[i%len(x0s)]
        funci = lambda sol: fun(sol, xp, yp, rp)
        # mini = minimize(funci, x0, method=method)
        mini = least_squares(funci, x0, method=method)
        r1[i], r2[i] = mini.x[0], mini.x[1]

    return r1, r2

def normalize(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def grayscale_im(im):
    return 0.3 * im[:,:,0] + 0.7 * im[:,:,1]

def subpixel(img, xmi, ymi ):
    im = np.pad(img, 1)
    ypi,xpi = ymi+1, xmi+1
    pm, pa, pp = np.log(im[ypi,xpi]+1e-10), np.log(im[ypi,xpi-1]+1e-10), np.log(im[ypi,xpi+1]+1e-10)
    corr = (pa-pp) / (pa+pp - 2*pm )
    return xpi + corr / 2 - 1

def laser_edges(im, sigma=20):
    ny,nx = np.shape(im)
    g2 = gaussian(im, sigma)
    dg = normalize( np.gradient(g2,axis=1) )
    mdg = np.median(dg)

    ili = np.arange(ny)

    # imi = np.argmin(dg, axis=1  ) 
    # ima = np.argmax(dg, axis=1  ) 
    # ima = np.array( [ (find_peaks( dg[i,:], height= mdg+0.1, prominence=0.1)[0])[0] for i in range(len(dg)) ] )
    # imi = np.array( [ (find_peaks(-dg[i,:], height=-mdg+0.1, prominence=0.1)[0])[0] for i in range(len(dg)) ] )

    ima, imi = np.full(len(dg), np.nan, dtype=int), np.full(len(dg), np.nan, dtype=int)
    
    for j in range(len(dg)):
        arr1 = find_peaks( dg[j,:], height= mdg+0.1, prominence=0.1)[0]
        arr2 = find_peaks(-dg[j,:], height=-mdg+0.1, prominence=0.1)[0]    
        ima[j] = np.concatenate( (arr1[:1],[np.argmax(dg[j,:])]) )[0]
        imi[j] = np.concatenate( (arr2[:1],[np.argmin(dg[j,:])]) )[0]

    smi, sma = subpixel(normalize(dg), imi, ili), subpixel(normalize(dg), ima, ili)
    return ili, smi, sma

def laser_edges_fast(im, sigma=20):
    ny, nx = im.shape

    g2 = gaussian(im, sigma)
    dg = np.gradient(g2, axis=1)
    dg = normalize(dg)

    mdg = np.median(dg)
    ili = np.arange(ny)

    ima = np.empty(ny, dtype=int)
    imi = np.empty(ny, dtype=int)

    _fp = find_peaks
    _argmax = np.argmax
    _argmin = np.argmin

    for j in range(ny):
        row = dg[j]

        pmax, _ = _fp(row, height=mdg+0.1, prominence=0.1)
        pmin, _ = _fp(-row, height=-mdg+0.1, prominence=0.1)

        ima[j] = pmax[0] if pmax.size else _argmax(row)
        imi[j] = pmin[0] if pmin.size else _argmin(row)

    smi = subpixel(dg, imi, ili)
    sma = subpixel(dg, ima, ili)

    return ili, smi, sma


def cuad(x,a,b,c):
    return a * x**2 + b * x + c 

# def fit_wall_pixels(ili, sma, distance=100):
#     order = np.argsort(sma)
#     da = np.where( np.diff(sma[order]) > distance )[0]+1
#     os = np.split(order,da)
#     fma = np.zeros_like(sma)
#     for j in range(len(os)):
#         if len(os[j]) > 2:
#             (a,b,c),cov = curve_fit(cuad, ili[os[j]], sma[os[j]])
#             fma[ili[os[j]]] = cuad(ili[os[j]], a,b,c)
#         else: fma[ili[os[j]]] = sma[ili[os[j]]]

#     return fma

def fit_wall_pixels(ili, sma, distance=100):
    order = np.argsort(sma)

    da = np.where(np.diff(sma[order]) > distance)[0] + 1
    segments = np.split(order, da)

    fma = np.empty_like(sma)

    for idx in segments:
        n = idx.size

        if n > 2:
            x = ili[idx]
            y = sma[idx]
            a, b, c = np.polyfit(x, y, 2)
            fma[x] = a*x*x + b*x + c
        else:
            fma[ili[idx]] = sma[idx]

    return fma


def wall_d_y(y, y0s, x0s):
    m = (x0s[1] - x0s[0]) / (y0s[1] - y0s[0])
    return m * (y - y0s[0]) + x0s[0]


# def ice_boundary(ili, smi, sma, wall_distace, calib, dis_bar=True, method='SLSQP'):
    # wall_d = np.ones_like(ili) * wall_distace #distance from grid to window in mm (from fisrt point)
    # points = np.vstack((sma,ili,wall_d)).T
    # yr, zr = calib_invzl(calib, points, 0, dis_bar=dis_bar, method=method)    
    # yc, zc, dwall = yr, zr, wall_d
    
    # points = np.vstack((smi,ili,zc)).T
    # xi, yi = calib_invzl(calib, points, 2, dis_bar=dis_bar, method=method)
    # return dwall, yc, zc, xi, yi

def ice_boundary(ili, smi, sma, wall_distace, calib, dis_bar=True, method='SLSQP'):
    n = ili.size

    # Preallocate
    wall_d = np.empty(n, dtype=np.float32)
    wall_d.fill(wall_distance)

    points = np.empty((n, 3), dtype=np.float64)
    points[:, 0], points[:, 1], points[:, 2] = sma, ili, wall_d

    yr, zr = calib_invzl(calib, points, 0, dis_bar, method)
    yc, zc = yr, zr

    # Second transform (reuse points)
    points[:, 0], points[:, 2] = smi, zc

    xi, yi = calib_invzl(calib, points, 2, dis_bar, method)

    return wall_d, yc, zc, xi, yi

def ice_boundary_fast(ili, smi, sma, wall_distace, calib, dis_bar=True, method='SLSQP'):
    n = ili.size

    # Preallocate
    wall_d = np.empty(n, dtype=np.float32)
    wall_d.fill(wall_distance)

    points = np.empty((n, 3), dtype=np.float64)
    points[:, 0], points[:, 1], points[:, 2] = sma, ili, wall_d

    yr, zr = calib_invzl_fast(calib, points, 0, dis_bar, method)
    yc, zc = yr, zr

    # Second transform (reuse points)
    points[:, 0], points[:, 2] = smi, zc

    xi, yi = calib_invzl_fast(calib, points, 2, dis_bar, method)

    return wall_d, yc, zc, xi, yi




def initial_frames(times, fps=30):
    dis = [0]
    for l in range(1,len(times)):
        dis.append( (int(times[l][:2]) - int(times[0][:2]))*60**2 + 
                   (int(times[l][3:5]) - int(times[0][3:5]))*60 + int(times[l][6:8]) - int(times[0][6:8]) )
    dis = np.array(dis)
    return dis*fps

def frames_reconstruction(start, interval, times, len_vid):
    ifram = initial_frames(times)
    end = ifram[-1] + len_vid[-1]
    
    pos = np.arange(start, end, interval*30)
    len_vid = np.array(len_vid)-2
    
    iv = np.sum( (pos - np.expand_dims(ifram,axis=1)) >= 0, axis=0 )-1
    ev = np.sum( (pos+60 - np.expand_dims(ifram,axis=1)) >= 0, axis=0 )-1
    ov = np.sum( (pos+60 - np.expand_dims(ifram+len_vid,axis=1)) >= 0, axis=0 )
    
    prob1 = np.where((ev-iv)!=0)[0]
    prob2 = np.where((ov-iv)!=0)[0]
    
    if len(prob1) == 0 and len(prob2) == 0:
        print('All good')
        return pos - ifram[iv], iv, pos
    else:    
        print('Error, Issues at intervals {:}'.format(prob2))
        print('Distance to next video: {:}'.format(ifram[ov[prob2]] - pos[prob2]) )
        print()
        # print( ifram[ev[prob1]] - pos[prob1] )
        return pos - ifram[iv], iv, pos


def count_missing_frames_with_video_lengths( brightness, video_lengths, lims, fps=30, threshold=None, divisor=None, missing_cycles=None,
                                            blink_period_sec=60.0, min_on_frames=1, tolerance = 5):
    """
    Estimate missing frames between videos using LED blink timing.

    Parameters
    ----------
    brightness : array-like
        Mean brightness per frame (concatenated videos).
    video_lengths : list[int]
        Length (in frames) of each video, in order.
    lims : [int,int] 
        First and last recorded frames to consider (exclusive).
    fps : float
        Frames per second.
    threshold : float or None
        Brightness threshold for LED ON detection.
    divisors : int or None
       Divisors of the blink interval (e.g. [2, 3] → half, third).
    missing_cycles: list or None
        List should have the same number of elements as cut (or len(video_lenghts)-2 ). Each element is the amount of cycles skipped in each cut.
    blink_period_sec : float
        LED blink period in seconds.
    min_on_frames : int
        Minimum consecutive ON frames to count as a blink.
     tolerance : inte
            Allowed deviation from expected blink period (in frames).
    Returns
    -------
    total_missing_frames : int
        Total missing frames across all cuts.
    missing_per_cut : list[int]
        Missing frames between each video pair.
    blink_frames : np.ndarray
        Global frame indices where blinks were detected.
    """

    brightness = np.asarray(brightness)

    if threshold is None:
        low = np.percentile(brightness, 20)
        high = np.percentile(brightness, 80)
        threshold = (low + high) / 2

    # Detect LED ON frames
    led_on = brightness > threshold

    # Rising edges → blink start
    edges = np.diff(led_on.astype(int)) == 1
    candidate_frames = np.where(edges)[0] + 1

    # Filter noise
    blink_frames = []
    for f in candidate_frames:
        if np.sum(led_on[f:f + min_on_frames]) >= min_on_frames:
            blink_frames.append(f)

    blink_frames = np.array(blink_frames)

    # Expected blink spacing
    expected_frames = int(round(blink_period_sec * fps))

    # Compute video boundaries
    video_starts = np.cumsum(video_lengths[:-1])
    video_ends = np.cumsum(video_lengths[1:])

    missing_per_cut = []
    
    for i in range(len(video_lengths) - 2):
        # Last blink in video i
        blinks_current = blink_frames[
            (blink_frames >= video_starts[i]) &
            (blink_frames < video_ends[i])
        ]
    
        # First blink in video i+1
        blinks_next = blink_frames[
            (blink_frames >= video_starts[i + 1]) &
            (blink_frames < video_ends[i + 1])
        ]
    
        if len(blinks_current) == 0 or len(blinks_next) == 0:
            missing_per_cut.append(0)
            continue
    
        last_blink = blinks_current[-1]
        first_blink = blinks_next[0]
    
        gap = first_blink - last_blink
        is_missing = int( np.abs(gap - expected_frames) > tolerance ) 
        missing = max(0, (expected_frames - gap) * is_missing )
    
        missing_per_cut.append(int(missing))
        
    missing_per_cut = np.array(missing_per_cut) 
        
    if missing_cycles:
        missing_cycle_frames = np.array(missing_cycles) * expected_frames
        missing_per_cut = missing_per_cut + missing_cycle_frames
        
    # ---------- Build cumulative missing before each video ---------
    cumulative_missing = np.zeros(len(video_lengths)-1, dtype=int)
    for i in range(1, len(video_lengths)-1):
        cumulative_missing[i] = cumulative_missing[i - 1] + missing_per_cut[i - 1]
        
    # ---------- Divisor frame computation ----------
    divisor_frames = None
    if divisor:
        divisor_frames = []
    
        for k in range(len(blink_frames)):
            
            blink_curent = blink_frames[k]
            
            for l in range(divisor):
                ideal_time = blink_curent + round(expected_frames * l / divisor)

                vid_blink = np.searchsorted(video_ends, blink_curent, side="right")
                vid_current = np.searchsorted(video_ends, ideal_time, side="right")
                
                # check if doesn't go after the recorded time
                if vid_current < len(video_ends):
                    cum_difference = cumulative_missing[vid_current] - cumulative_missing[vid_blink]
    
                    # check if it is recorded
                    if cum_difference > 0:
                        
                        frames_to_next_video = video_ends[vid_blink] - blink_curent + missing_per_cut[vid_blink]
                        is_recorded = frames_to_next_video < round(expected_frames * l / divisor)
    
                        if is_recorded:
                            recorded_frame = ideal_time - cum_difference                        
                            divisor_frames.append( recorded_frame )
                    
                    else:
                        recorded_frame = ideal_time - cum_difference                        
                        divisor_frames.append( recorded_frame )
            
        divisor_frames = np.array(divisor_frames)
    
    start,end = lims
    filt_blink = (blink_frames > start) & (blink_frames < end)  
    blink_frames = blink_frames[filt_blink]

    filt_div = (divisor_frames > start) & (divisor_frames < end)  
    divisor_frames = divisor_frames[filt_div]
    
    # ---------- Calculate times ----------
    start_time = start / fps
    
    divisor_vid = np.searchsorted(video_ends, divisor_frames, side="right")
    divisor_missing = cumulative_missing[divisor_vid]
    divisor_time = (divisor_frames+divisor_missing) / fps    

    return missing_per_cut, blink_frames, divisor_frames, divisor_time-start_time

def is_reconstructable(frames, video_lengths, N_reconstruction, times, shift=0, fps=30 ):
    """
    Checks whether I can do the full reconstruction of at all intended times within the same video.

    Parameters
    ----------
    frames: list[int]
        Frames at which each reconstruction starts.
    video_lengths : list[int]
        Length (in frames) of each video, in order.
    N_reconstruction: int
        Number of frames to use for the reconstruction.
    times: list
        times corresponding to each frame
    shift: int
        Shifts the starting point of each reconstruction        
    fps: float
        Fps of video recording

    Returns
    -------
    n_frames: list[int]
        The shifted frames.
    frames_start_vid: list[ind]
        The video where each n_frame belongs.
    n_times: list[int]
        The shifted times.
    """
    video_starts = np.cumsum(video_lengths[:-1])
    video_ends = np.cumsum(video_lengths[1:])

    frames_start_vid = np.searchsorted(video_ends, frames + shift, side="right")
    frames_end_vid = np.searchsorted(video_ends, frames + N + shift, side="right")
    
    changes_vid = frames_end_vid - frames_start_vid
    
    if np.sum(changes_vid) > 0:
        ind = np.where( changes_vid>0 )[0]
        overshoot = frames[ind]+N+shift -  video_ends[frames_start_vid[ind]] 

        print(f'With shift {shift}, issue(s) at reconstruction n° {ind} (located at frame {frames[ind]+shift}.')
        print(f'Number of frames in next video (or in no video): {overshoot}')
        
        #Calculate minimum a maximum posible shifts (from reference 0 shift)
        vid_start_distance =  np.min(frames - video_starts[frames_start_vid])
        vid_end_distance = np.min(video_ends[frames_end_vid] - frames+N)
        
        print(f'Minimum possible shift: {-vid_start_distance}. Maximum possible shift: {vid_end_distance}.' )
        
    else: 
        #Calculate minimum a maximum posible shifts (from reference 0 shift)
        vid_start_distance =  np.min(frames - video_starts[frames_start_vid])
        vid_end_distance = np.min(video_ends[frames_end_vid] - frames+N)
        
        print('No Issues')
        print(f'Minimum possible shift: {-vid_start_distance}. Maximum possible shift: {vid_end_distance}.' )
    
    print()
    
    n_frames = frames + shift
    n_times = times + shift / fps
    
    return n_frames, frames_start_vid, n_times
    

def pixel_filtering(xi, xc, ili, extra_tol=10, filter_1=720, filter_2=22, filter_3=25, top_removal=False, print_sd=False):
    """
    Filters out the incorrect points of the surface recontruction. It consists of 3 filters:
        1. (Optional) A final at the top part of the ice. Removes the points above the top surface of the ice
        2. If the reconstructed surface is too noisy, it is completely discarded.
        3. Points that are too separetad from the mean value are also discarded.
    Discarded/Reemoved point are replaced with NaN's in the returned array

    Parameters
    ----------
    xi : 1D-array
        x position of the ice.
    xc : 1D-array
        x position of the wall.
    ili : 1D-arrat
        Vertical pixel position.
    extra_tol : float, optional
        How much bigger the mean distance between the ice and the wall should be to stop considering it a part of the ice. The default is 10.
    filter_1 : float, optional
        Vertical pixel position where to stop looking for the surface of the ice. The default is 720.
    filter_2 : float, optional
        Threshold of standard deviation of the ice x position (after fisrt filter was apply if relevant) for the second filter (2.). The default is 22.
    filter_3 : float, optional
        Distance threshold to mean of the ice x position (after fisrt filter was apply if relevant) for the third filter (3.). The default is 25.
    top_removal : bool, optional
        Set to True to apply the first filter (1.). The default is False.
    print_sd : bool, optional
        Prints the standard deviation of the ice x position (after fisrt filter was apply if relevant). The default is False.

    Returns
    -------
    xii : 1D-array
        Copy of xi, with the masked points replaced with NaN's.
    """
        
    xii = np.copy(xi)

    if top_removal:
        uplim = np.median(xi - xc) + extra_tol #110
        ifil = np.where( (np.abs(xi - xc) > uplim) * (ili < filter_1) )[0]
        fil1 = np.zeros_like(xii, dtype=bool)
        if len(ifil)>0:
            fil1[:ifil[-1]+1] = True
            xii[fil1] = np.nan
    
    fil2 = np.nanstd(xii) > filter_2

    if print_sd: print(np.nanstd(xii))

    if fil2:
        xii[:] = np.nan
    else:
        fil3 = np.abs(xii - np.nanmean(xii)) > filter_3 
        xii[fil3] = np.nan
    
    return xii

#%%
# 30 fps

path = '/Volumes/Ice blocks/Scan water channel/26-03-05/'

data = np.load(path+'calibration_data.npz')
angle_xy, angle_yz, angle_xz = float(data['arr_3']), float(data['arr_4']), float(data['arr_5'])

cal_up = data['arr_0']
cal_do = data['arr_1']
wall_distance = float(data['arr_2'])


dvids = [cv2.VideoCapture( path + 'Camera down/DSC_9997.MOV'), # starts 4542
         cv2.VideoCapture( path + 'Camera down/DSC_9998.MOV')  # ends 19312
         ]

uvids = [cv2.VideoCapture( path + 'Camera up/DSC_0015.MOV'), # starts 4482
         cv2.VideoCapture( path + 'Camera up/DSC_0016.MOV')  # ends 17485
         ]

dlens = [0]+[int(dvids[i].get(cv2.CAP_PROP_FRAME_COUNT)) for i in range(len(dvids))]
ulens = [0]+[int(uvids[i].get(cv2.CAP_PROP_FRAME_COUNT)) for i in range(len(uvids))]

#fps 30, (supposedly 29.97)
#%%
# Visualize some frames, useful for finding led position
l = 1

for i in range(15900,15910,1):
    dvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)      
    
    # im = np.array( dvids[l].read()[1] )[:,:,::-1]
    im = np.array( dvids[l].read()[1] )[85:130,480:530,::-1]
    # im = np.array( dvids[l].read()[1] )[:,1500:2700,::-1]
    # im = grayscale_im(im)
    im = im[:,:,0]

    plt.figure()
    plt.imshow(im, cmap='gray') #, vmax=20) 
    plt.title(i)
    plt.show()
    print(i, np.mean(im), np.median(im))

# for i in range(15870,15880,1):
#     uvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)
    
#     # im = np.array( uvids[l].read()[1] )[:,:,::-1]
#     im = np.array( uvids[l].read()[1] ) [1970:2010,415:460,::-1] 
#     # im = np.array( uvids[l].read()[1] ) [:,1400:2600,::-1] 
#     # im = grayscale_im(im)
#     im = im[:,:,0]

#     plt.figure()
#     plt.imshow(im, cmap='gray') #, vmax=20) 
#     plt.title(i)
#     plt.show()
#     print(i, np.mean(im), np.median(im))



#%%
# Led finding
d_led = [85,130,480,530]
u_led = [1970,2010,415,460]

u_mes, d_mes = [],[]

for l in range(len(dvids)):
    dvids[l].set(cv2.CAP_PROP_POS_FRAMES, 0)
    for j in tqdm(range(dlens[l+1])):
        im = np.array( dvids[l].read()[1] )[d_led[0]:d_led[1],d_led[2]:d_led[3]] 
        # im = grayscale_im(im)
        im = im[:,:,2]
        d_mes.append(np.mean(im))

for l in range(len(uvids)):
    uvids[l].set(cv2.CAP_PROP_POS_FRAMES, 0)
    for j in tqdm(range(ulens[l+1])):
        im = np.array( uvids[l].read()[1] )[u_led[0]:u_led[1],u_led[2]:u_led[3]] 
        # im = grayscale_im(im)
        im = im[:,:,2]
        u_mes.append(np.mean(im))

np.savez(path+'led_blink.npz', u_mes=u_mes, d_mes=d_mes)

#%%

blink = np.load(path+'led_blink.npz')
u_mes,d_mes = blink['u_mes'], blink['d_mes']

threshold = [100]
plt.figure()
plt.plot( u_mes , '.-')
plt.plot( d_mes, '.-')
plt.hlines( threshold, 0, np.max([len(d_mes),len(u_mes)]), color='g' )
plt.grid()
plt.show()

#%%
# Finding frame where to start reconstruction (from led blinking)
N = 60
shift = 0
endd = 47707 # np.sum(dlens) # 83090 + 0 # np.sum(dlens)
endu = 47665 # np.sum(ulens) # 83062 + 0 # np.sum(ulens)

d_cuts, u_cuts = np.cumsum(dlens), np.cumsum(ulens)

d_missing, d_led, d_frame, d_time = count_missing_frames_with_video_lengths(d_mes, dlens[:], [4542, endd], fps=30, 
                                                                            threshold=100, divisor=2, missing_cycles=None )
u_missing, u_led, u_frame, u_time = count_missing_frames_with_video_lengths(u_mes, ulens[:], [4482, endu], fps=30, 
                                                                            threshold=100, divisor=2 )
print('Down')
d_frame, d_vid, d_time = is_reconstructable(d_frame, dlens, N, d_time, shift=shift, fps=30 )

print('Up')
u_frame, u_vid, u_time = is_reconstructable(u_frame, ulens, N, u_time, shift=shift, fps=30 )


h1,h2,h3 = 90,100,110
time_unit = 1 #60 for min, 1 for seconds

plt.figure()
plt.plot( d_mes, '.-', label='brightness' )
plt.vlines( d_led,0,h1, color='r', label='led' )
plt.vlines( d_frame,0,h2, color='g', alpha=0.4, label='record' )
plt.vlines( d_cuts,0,h3,colors='k',alpha=0.5, label='video cut' )
plt.grid()
plt.title('d')
plt.xlabel('frame')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot( u_mes, '.-', label='brightness'  )
plt.vlines( u_led,0,h1, color='r', label='led' )
plt.vlines( u_frame,0,h2, color='g', alpha=0.4, label='record'  )
plt.vlines( u_cuts,0,h3,colors='k',alpha=0.5, label='video cut' )
plt.grid()
plt.title('u')
plt.xlabel('frame')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot( d_time/time_unit, '.-' , label='down')
plt.plot( u_time/time_unit, '.-' , label='up' )
plt.xlabel('record')
plt.legend()
if time_unit == 60: plt.ylabel('time (min)')
elif time_unit == 1: plt.ylabel('time (seg)')
else: plt.ylabel('time (other units)')
plt.grid()
plt.show()


#%%

# d_frame = np.arange(3158, np.sum( dlens ) - dlens[-1] + 6258+1, 1 )
# d_vid = np.searchsorted( np.cumsum(dlens), d_frame) - 1
# d_missing = np.array( [0,0,0,0,812] )

# d_pframe = d_frame+d_missing[d_vid]
# d_times = (d_pframe ) / 30

# d_time = (d_times - d_times[0]) % 30 < 0.01

# alog = np.arange(3158, 3158+len(d_times) )

# plt.figure()
# # plt.plot( np.diff( d_frame ), '.-' )
# # plt.plot( np.diff( d_pframe ), '.-' )
# plt.plot( alog, d_times  / 60 , '.-' )
# plt.plot( alog[d_time], d_times[d_time] / 60 , '.' )
# plt.vlines( np.cumsum(dlens), -1,35, colors='gray' )
# plt.grid()
# plt.title('d')
# plt.show()



# u_frame = np.arange(3198, np.sum( ulens ) - ulens[-1] + 6139+1, 1 )
# u_vid = np.searchsorted( np.cumsum(ulens), u_frame) - 1
# u_missing = np.array( [0,0,0,937] )

# u_pframe = u_frame+u_missing[u_vid]
# u_times = (u_pframe ) / 30

# u_time = np.logical_or( (u_times - u_times[0]) % 30 < 0.01, (u_times - u_times[0]) % 30 > 29.99 )

# alog = np.arange(3198, 3198+len(u_times) )

# plt.figure()
# # plt.plot( np.diff( u_frame ), '.-' )
# # plt.plot( np.diff( u_pframe ), '.-' )
# plt.plot( alog, u_times  / 60 , '.-' )
# plt.plot( alog[u_time], u_times[u_time] / 60 , '.' )
# plt.vlines( np.cumsum(ulens), -1,35, colors='gray' )
# plt.grid()
# plt.title('u')
# plt.show()

# N = 60
# d_cuts, u_cuts = np.cumsum(dlens), np.cumsum(ulens)

# d_frame = d_frame[d_time]
# d_vid = np.searchsorted( np.cumsum(dlens), d_frame) - 1
# d_time = (d_times - d_times[0])[d_time]


# u_frame = u_frame[u_time]
# u_vid = np.searchsorted( np.cumsum(ulens), u_frame) - 1
# u_time = (u_times - u_times[0])[u_time]


#%%
# testing camera down laser recognition
i = -28395 + 46976
disp = 55
l = 1
first_graph = 0
second_graph = 1

lims = [None,None,1300,2450]
filter_2 = 22
filter_3 = 35 #np.nanstd(xii) * 3.


dvids[l].set(cv2.CAP_PROP_POS_FRAMES, i+disp)
imo = np.array( dvids[l].read()[1] )[:,:,::-1]

ny, nx, _ = imo.shape
top,bot = int(lims[0] or 0), ny-int(lims[1] or ny) 

t1 = time()
im = grayscale_im(imo[lims[0]:lims[1],lims[2]:lims[3]])
# ili, smi, sma = laser_edges(im, sigma=20)
ili, smi, sma = laser_edges_fast(im, sigma=20)
if lims[2]: smi,sma = smi+lims[2], sma+lims[2]

fma = fit_wall_pixels(ili, sma)
ili = ili+top

# xc, yc, zc, xi, yi = ice_boundary(ili, smi, fma, wall_distance, cal_do, method='lm')
xc, yc, zc, xi, yi = ice_boundary_fast(ili, smi, fma, wall_distance, cal_do, method='lm')

xii = pixel_filtering(xi, xc, ili, top_removal=False, filter_2=filter_2, filter_3=filter_3, print_sd=False)

xc, yc, zc = np.pad(xc,[top,bot],constant_values=np.nan), np.pad(yc,[top,bot],constant_values=np.nan), np.pad(zc,[top,bot],constant_values=np.nan)
xii, xi, yi = np.pad(xii,[top,bot],constant_values=np.nan), np.pad(xi,[top,bot],constant_values=np.nan), np.pad(yi,[top,bot],constant_values=np.nan)
zi = zc

print(np.sum(np.isnan(xii)))

if first_graph:
    plt.figure()
    plt.imshow(imo, cmap='gray')
    plt.plot(smi, ili,'b-',alpha=0.5)
    plt.plot(sma, ili,'r-',alpha=0.5)
    plt.plot(fma, ili,'y--',alpha=0.5)
    plt.plot(smi[np.isnan(xii)[top:ny-bot]], ili[np.isnan(xii)[top:ny-bot]],'m.',alpha=0.5)
    plt.show()

if second_graph:
    plt.figure()
    plt.imshow( grayscale_im(imo), cmap='gray', vmax = 15 )
    plt.plot(smi, ili,'b-',alpha=0.5)
    plt.plot(sma, ili,'r-',alpha=0.5)
    plt.plot(fma, ili,'y--',alpha=0.5)
    plt.plot(smi[np.isnan(xii)[top:ny-bot]], ili[np.isnan(xii)[top:ny-bot]],'m.',alpha=0.5)
    plt.show()

#%%
# testing camera up laser recognition
i = -0 + 4687
disp = 2
l = 0
first_graph = 0
second_graph = 1

lims = [100,None,1250,2500]
extra_tol = 15
filter_1 = 800 #1000
filter_2 = 35
filter_3 = 35 #np.nanstd(xii) * 3.


uvids[l].set(cv2.CAP_PROP_POS_FRAMES, i+disp)
imo = np.array( uvids[l].read()[1] )[:,:,::-1]
ny, nx, _ = imo.shape
top,bot = int(lims[0] or 0), ny-int(lims[1] or ny) 

im = grayscale_im(imo[lims[0]:lims[1],lims[2]:lims[3]])
# ili, smi, sma = laser_edges(im, sigma=20)
ili, smi, sma = laser_edges_fast(im, sigma=20)
if lims[2]: smi,sma = smi+lims[2], sma+lims[2]
fma = fit_wall_pixels(ili, sma)
ili = ili+top

# xc, yc, zc, xi, yi = ice_boundary(ili, smi, fma, wall_distance, cal_up, method='lm')
xc, yc, zc, xi, yi = ice_boundary_fast(ili, smi, fma, wall_distance, cal_up, method='lm')
xii = pixel_filtering(xi, xc, ili, top_removal=True, extra_tol=extra_tol, filter_1=filter_1, filter_2=filter_2, filter_3=filter_3)


xc, yc, zc = np.pad(xc,[top,bot],constant_values=np.nan), np.pad(yc,[top,bot],constant_values=np.nan), np.pad(zc,[top,bot],constant_values=np.nan)
xii, xi, yi = np.pad(xii,[top,bot],constant_values=np.nan), np.pad(xi,[top,bot],constant_values=np.nan), np.pad(yi,[top,bot],constant_values=np.nan)
zi = zc
    
    
print(np.sum(np.isnan(xii)))

if first_graph:
    plt.figure()
    plt.imshow(imo, cmap='gray')
    # plt.imshow(im, cmap='gray')
    plt.plot(smi, ili,'b-',alpha=0.5)
    plt.plot(sma, ili,'r-',alpha=0.5)
    plt.plot(fma, ili,'y--',alpha=0.5)
    plt.plot(smi[np.isnan(xii)[top:ny-bot]], ili[np.isnan(xii)[top:ny-bot]],'m.',alpha=0.5)
    plt.show()

if second_graph:
    plt.figure()
    plt.imshow( grayscale_im(imo), cmap='gray', vmax = 30 )
    plt.plot(smi, ili,'b-',alpha=0.5)
    plt.plot(sma, ili,'r-',alpha=0.5)
    plt.plot(fma, ili,'y--',alpha=0.5)
    plt.plot(smi[np.isnan(xii)[top:ny-bot]], ili[np.isnan(xii)[top:ny-bot]],'m.',alpha=0.5)
    plt.show()

#%%
# reconstruction
t1 = time()

ny,nx, _ = np.shape(dvids[0].read()[1])
with h5py.File(path + 'reconstructed_profile.hdf5', 'w') as f:

    # Down
    nt = len(d_frame)
    gdo = f.create_group('Down')

    gdo_t = gdo.create_dataset('time', (nt,), dtype='f') 
    gdo_x = gdo.create_dataset('x', (nt,ny*N), dtype='f') 
    gdo_y = gdo.create_dataset('y', (nt,ny*N), dtype='f') 
    gdo_z = gdo.create_dataset('z', (nt,ny*N), dtype='f') 
    
    lims = [None,None,1300,2450] 
    filter_2 = 22
    filter_3 = 35 

    top,bot = int(lims[0] or 0), ny-int(lims[1] or ny) 
    gdo_t[:] = d_time
    for i in tqdm(range(nt),disable=False):
        
        vid = dvids[d_vid[i]]
        ini = d_frame[i] - d_cuts[d_vid[i]]
        vid.set(cv2.CAP_PROP_POS_FRAMES, ini)

        for j in tqdm(range(N),disable=True):
            
            imo = np.array( vid.read()[1] )[:,:,::-1]
            
            im = grayscale_im(imo[lims[0]:lims[1],lims[2]:lims[3]])
            # ili, smi, sma = laser_edges(im, sigma=20)
            ili, smi, sma = laser_edges_fast(im, sigma=20)
            if lims[2]: smi,sma = smi+lims[2], sma+lims[2]
            fma = fit_wall_pixels(ili, sma)
            ili = ili+top
            
            # xc, yc, zc, xi, yi = ice_boundary(ili, smi, fma, wall_distance, cal_do, method='lm')
            xc, yc, zc, xi, yi = ice_boundary_fast(ili, smi, fma, wall_distance, cal_do, method='lm')
            xii = pixel_filtering(xi, xc, ili, top_removal=False, filter_2=filter_2, filter_3=filter_3)

            xii, yi, zc = np.pad(xii,[top,bot],constant_values=np.nan), np.pad(yi,[top,bot],constant_values=np.nan), np.pad(zc,[top,bot],constant_values=np.nan)
            zi = zc

            gdo_x[i,j*ny:(j+1)*ny] = xii
            gdo_y[i,j*ny:(j+1)*ny] = yi
            gdo_z[i,j*ny:(j+1)*ny] = zi    

            
    # Up
    nt = len(u_frame)
    gup = f.create_group('Up')

    gup_t = gup.create_dataset('time', (nt,), dtype='f') 
    gup_x = gup.create_dataset('x', (nt,ny*N), dtype='f') 
    gup_y = gup.create_dataset('y', (nt,ny*N), dtype='f') 
    gup_z = gup.create_dataset('z', (nt,ny*N), dtype='f') 
    
    lims = [100,None,1250,2500]
    extra_tol = 15
    filter_1 = 800
    filter_2 = 35
    filter_3 = 35

    top,bot = int(lims[0] or 0), ny-int(lims[1] or ny) 
    gup_t[:] = u_time
    for i in tqdm(range(nt)):
        
        vid = uvids[u_vid[i]]
        ini = u_frame[i] - u_cuts[u_vid[i]]
        vid.set(cv2.CAP_PROP_POS_FRAMES, ini)

        for j in range(N):
            
            imo = np.array( vid.read()[1] )[:,:,::-1]

            im = grayscale_im(imo[lims[0]:lims[1],lims[2]:lims[3]])
            # ili, smi, sma = laser_edges(im, sigma=20)
            ili, smi, sma = laser_edges_fast(im, sigma=20)
            if lims[2]: smi,sma = smi+lims[2], sma+lims[2]
            fma = fit_wall_pixels(ili, sma)
            ili = ili+top
            
            # xc, yc, zc, xi, yi = ice_boundary(ili, smi, fma, wall_distance, cal_up, method='lm')
            xc, yc, zc, xi, yi = ice_boundary_fast(ili, smi, fma, wall_distance, cal_up, method='lm')
            xii = pixel_filtering(xi, xc, ili, top_removal=True, extra_tol=extra_tol, filter_1=filter_1, filter_2=filter_2, filter_3=filter_3)

            xii, yi, zc = np.pad(xii,[top,bot],constant_values=np.nan), np.pad(yi,[top,bot],constant_values=np.nan), np.pad(zc,[top,bot],constant_values=np.nan)
            zi = zc

            gup_x[i,j*ny:(j+1)*ny] = xii
            gup_y[i,j*ny:(j+1)*ny] = yi
            gup_z[i,j*ny:(j+1)*ny] = zi    

t2 = time()
print(t2-t1)

#%%
# quick visualization of reconstruction
with h5py.File(path + 'reconstructed_profile.hdf5', 'r') as f:
    
    top_t = f['Up/time'][:]
    top_x = f['Up/x'][:]
    top_y = f['Up/y'][:]
    top_z = f['Up/z'][:]

    dow_t = f['Down/time'][:]
    dow_x = f['Down/x'][:]
    dow_y = f['Down/y'][:]
    dow_z = f['Down/z'][:]
#%

#%%


#%%
# =============================================================================
# Back iamge
# =============================================================================

def pick_points_on_figure(im, fig_num, max_picks=1):    
    picked_points_x, picked_points_y = [],[]

    fig, ax = plt.subplots()
    ax.set_title(f"Figure {fig_num}: Pick {max_picks} points")
    ax.imshow(im, cmap='gray')

    def on_pick(event):

        picked_x = event.xdata
        picked_y = event.ydata
        
        picked_points_x.append(picked_x)
        picked_points_y.append(picked_y)

        # if (picked_x, picked_y) not in picked_points:
        idx = len(picked_points_x)
        ax.plot(picked_x, picked_y, 'ro')

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        if idx >= max_picks:
            fig.canvas.mpl_disconnect(cid)
            plt.pause(1)
            plt.close(fig)


    cid = fig.canvas.mpl_connect('button_press_event', on_pick)

    while plt.fignum_exists(fig.number):
        plt.pause(0.1)  # pause briefly to process events            

    # cx,cy = picked_points_x[0],picked_points_y[0]
    # r1 = np.sqrt( (cx-picked_points_x[1])**2 + (cy-picked_points_y[1])**2 )
    # r2 = np.sqrt( (cx-picked_points_x[2])**2 + (cy-picked_points_y[2])**2 )
    return picked_points_x, picked_points_y 

def circle_fit(points, vals):
    cx,cy,r = vals[0],vals[1],vals[2]
    distc = (points[:,0] - cx)**2 + (points[:,1] - cy)**2
    distr = distc - r**2
    return distr

def circle_filter(points, im, rad_width=35):
    fun = lambda vals: circle_fit(points, vals)
    ll = least_squares(fun, [500,500,500])
    
    yind,xind = np.indices(np.shape(im))
    rind = np.sqrt( (xind-ll.x[0])**2 + (yind-ll.x[1])**2 ) #
    # oind = np.arctan2( yind-ll.x[1], xind-ll.x[0] )
    fil = (rind>ll.x[2]-rad_width)*(rind<ll.x[2]+rad_width)

    return fil, ll.x

def angle_dot(im, fil, fit_val, angs_arr, ang_ini, c_ini , threshold=0.3, sizes=[700,3000], ang_tol=12, ecc_thres=0.65): #min_dist=10000 ):

    cx,cy,r = fit_val
    imt = np.copy(im)
    imt[fil] = 0
            
    angs_rotated = (angs_arr - ang_ini + 1*np.pi) % (2 * np.pi) - np.pi
    fila = ( angs_rotated < np.pi/ang_tol) * ( angs_rotated > -np.pi/ang_tol)
    imt[~fila] = 0
    
    la = label(imt>threshold)
    dict_prop = regionprops_table(la, properties=['area','centroid','eccentricity']) #to make it with idst t previous
    area_fil = (sizes[0]<dict_prop['area'])&(dict_prop['area']<sizes[1])
    ecc_fil = dict_prop['eccentricity'] < ecc_thres
    dist = (dict_prop['centroid-0']-c_ini[0])**2 + (dict_prop['centroid-1']-c_ini[1])**2  

    dotc = np.array([np.nan, np.nan])    
    
    dot = np.argmin( (1-area_fil*1)*50 + (1-ecc_fil*1)*50 + dist.argsort().argsort() )
    dotc[0] = dict_prop['centroid-0'][ dot ]
    dotc[1] = dict_prop['centroid-1'][ dot ]

    # dist_fil = dist < min_dist
    # fil_tot = area_fil&dist_fil
    # if np.sum(fil_tot) > 0:
    #     dotc[0] = dict_prop['centroid-0'][ fil_tot ][0]
    #     dotc[1] = dict_prop['centroid-1'][ fil_tot ][0]
    
    # dotc = np.array([np.nan, np.nan])
    # for prop in regionprops(la):
    #     if sizes[0] < prop.area < sizes[1]:
    #         dotc = np.array(prop.centroid)
    #         break

    ang = np.arctan2( -(dotc[0]-cy), dotc[1]-cx  )

    return ang

def circmedian(angs):
    pdists = angs[np.newaxis, :] - angs[:, np.newaxis]
    pdists = (pdists + np.pi) % (2 * np.pi) - np.pi
    pdists = np.abs(pdists).sum(1)
    return angs[np.argmin(pdists)]

# def get_median_angles(angles,stops,restarts,l):
#     angmed = []
#     for i in range(len(stops[l])):
#         if i == 0: 
#             angmed.append( circmedian( angles[ : stops[l][0] ] ) )
#         elif i == len(stops[l])-1: 
#             angmed.append( circmedian( angles[ restarts[l][-2] : ] ) )
#         else: 
#             angmed.append( circmedian( angles[ restarts[l][i-1] : stops[l][i]  ] ) )
#     return np.array(angmed)
    
def get_median_angles(angles,stops,restarts):
    angmed = []
    for i in range(len(stops)+1):
        if i == 0: 
            angmed.append( circmedian( angles[ : stops[0] ] ) )
        elif i == len(stops): 
            angmed.append( circmedian( angles[ restarts[-1] : ] ) )
        else: 
            angmed.append( circmedian( angles[ restarts[i-1] : stops[i]  ] ) )
    return np.array(angmed)
    

#%%

# at 24 fps

path = '/Volumes/Ice blocks/Scan water channel/26-03-05/' # this one has 60 fps

bvids = [cv2.VideoCapture( path + 'Camera back/_DSC5173.MOV'), #starts 1034 
         cv2.VideoCapture( path + 'Camera back/_DSC5174.MOV'), #ends 17048
          ]

blens = [0]+[int(bvids[i].get(cv2.CAP_PROP_FRAME_COUNT)) for i in range(len(bvids))]

#%%
# Find led to get times

l = 0
for i in range(2760,2761,1):
    bvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)

    im = np.array(bvids[l].read()[1])[:,:,::-1] #[:1900,1050:2900,::-1]
    # im = np.array(bvids[l].read()[1])[390:420,1750:1780,::-1] #[150:2100,750:2800,::-1]
    # im = np.array(bvids[l].read()[1])[:,:,::-1] #[1540:1595,880:930,::-1] #[150:2100,750:2800,::-1]
    im = im[:,:,1]
    
    plt.figure()
    plt.imshow(im, vmax=10)
    # plt.title('R, {}, {:.2f}'.format(i,np.mean(im[:,:,0])))
    # plt.title('{}, {:.2f}'.format(i,np.mean(im[:,:,0])))
    plt.title(i)
    plt.show()    

    print( i,'\t{:.2f}\t'.format(np.mean(im)), np.median(im) )
    
    # plt.figure()
    # plt.imshow(im[:,:,1])
    # plt.title('G, {}, {:.2f}'.format(i,np.mean(im[:,:,1])))
    # plt.show()    
    # plt.figure()
    # plt.imshow(im[:,:,2])
    # plt.title('B, {}, {:.2f}'.format(i,np.mean(im[:,:,2])))
    # plt.show()    

#%%

b_led = [390,420,1750,1780]

b_mes = []

start = 0
end = 17048

for l in range(len(bvids)):
    bvids[l].set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    starting, ending = 0, blens[l+1]
    if l == len(bvids)-1: ending = end
    if l == 0: starting = start

    for j in tqdm(range(starting, ending)):

        im = np.array( bvids[l].read()[1] )[b_led[0]:b_led[1],b_led[2]:b_led[3]] 
        im = im[:,:,2]

        b_mes.append(np.mean(im))

b_mes = np.array(b_mes)

#%%

threshold = 100
plt.figure()
plt.plot( b_mes , '.-')
plt.hlines( threshold, 0, len(b_mes), color='g' )
plt.grid()
plt.show()

missing_frames, _,_,_ = count_missing_frames_with_video_lengths( b_mes, blens, [start,blens[1]+end], fps=24, threshold=threshold, divisor=1 )

missing_frames
#%%

#calculate time using missing_frames

# np.save(path+'/back_times.npy', temp)

#%%
# Finding angle of handle

l=1
# for i in range(4000,20000,1000):
for i in range(9709, 9630-1, -10):
# for i in [12325]:
    bvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)
    # im = np.array(bvids[l].read()[1])[:,:,::-1] 
    # im = np.array(bvids[l].read()[1]) [50:1000,400:1450,::-1]
    im = np.array(bvids[l].read()[1])[50:1050,400:1500,1]
    g = gaussian(im,20)
    g[g>0.1] = 0
    g = normalize(g)
    
    plt.figure()
    plt.imshow( g , vmax= .5 )
    plt.title(i)
    plt.show()    
    
# 0: 3000, 15000
# 1: 12000, 15000
# 2: 12000, 18000
# 3: 0, 3000

#%%
# l = 0
blims = [50,1050,400,1500]
# blims = [50,1100,350,1400]

pos = []

# for i in [7000,13000,19000,22000,28000,37000,40000]:
for i,l in zip([5000,10000,15000,25000,30000,0,5000],[0,0,0,0,0,1,1]):
    bvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)
    im = np.array(bvids[l].read()[1])[blims[0]:blims[1],blims[2]:blims[3],::-1]
    im = normalize( gaussian( im[:,:,0], 2) )

    pos.append(pick_points_on_figure(im, i))
    
pos = (np.array(pos))[:,:,0]
fil, fit_val = circle_filter(pos, im)

fit_val
#%%

starts = [0,0]
ends = [blens[1], 17048]
hand_lim = [50,1050,400,1500]

t1 = time()

cxs, cys = [], []
for l in range(len(starts)):
    bvids[l].set(cv2.CAP_PROP_POS_FRAMES, starts[l])
    n_frames = ends[l] - starts[l]
    
    for i in tqdm(range(n_frames)):
        
        im = np.array(bvids[l].read()[1])[hand_lim[0]:hand_lim[1], hand_lim[2]:hand_lim[3], 2]
        im[30:100,880:980] = 0
        b_im = im > 200
        yb, xb = np.where(b_im)
        cx, cy = np.mean(xb), np.mean(yb)
        
        cxs.append(cx)
        cys.append(cy)
        
cxs, cys = np.array(cxs), np.array(cys) 

#%%
mfr_1 = np.array( [8554,8624,8704,8984,9224,11454,12265,12275,12285,12315,12345,12375,12385,12405,12415,12435,12465,12485,12505,12515,12535,12555,
       12575,12595,12625,12655,12675,12685,12715,12775,12825,12985,13195,13645,14195,14945,15950,16850,16900,16910,16940,16960,16980,17000,
       17010,17030,17040,17060,17080,17090,17110,17130,17140,17160,17170,17190,17210,17230,17240,17260,17280,17290,17310,17330,17350,17370,
       17390,17410,17440,17480,17530,17600,17900,18350,19600,20200,21600,21750,21800,21850,21860,21890,21900,21920,21940,21960,21970,21990,
       22000,22030,22050,22070,22090,22110,22130,22140,22160,22180,22200,22220,22250,22270,22290,22300,22320,22340,22350,22370,22390,22410,
       22430,22450,22460,22480,22500,22520,22550,22600,22620,22770] )
mcx_1 = [236 ,374 ,425 ,452 ,452 ,455  ,447  ,482  ,839  ,836  ,244  ,222  ,623  ,633  ,996  ,390  ,149  ,677  ,671  ,996  ,674  ,122  ,
       182  ,333  ,931  ,836  ,138  ,214  ,214  ,222  ,228  ,228  ,238  ,238  ,236  ,236  ,225  ,246  ,225  ,571  ,999  ,571  ,130  ,176  ,
       398  ,428  ,985  ,801  ,214  ,238  ,401  ,514  ,999  ,779  ,301  ,108  ,287  ,384  ,920  ,939  ,436  ,122  ,249  ,412  ,950  ,926  ,
       512  ,228  ,200  ,187  ,203  ,195  ,192  ,192  ,192  ,200  ,192  ,190  ,198  ,182  ,106  ,119  ,839  ,847  ,547  ,119  ,398  ,395  ,
       731  ,907  ,926  ,368  ,154  ,704  ,782  ,961  ,412  ,195  ,228  ,709  ,936  ,709  ,222  ,146  ,368  ,420  ,1009 ,693  ,333  ,133  ,
       265  ,276  ,899  ,882  ,547  ,255  ,265  ,271  ,271,  279  ]
mcy_1 = [166 ,907 ,961 ,964 ,972 ,956  ,964  ,953  ,872  ,136  ,130  ,785  ,964  ,975  ,377  ,63   ,693  ,934  ,964  ,569  ,33   ,339  ,
       764  ,918  ,777  ,122  ,260  ,812  ,877  ,853  ,864  ,853  ,837  ,850  ,858  ,866  ,858  ,853  ,845  ,972  ,415  ,68   ,304  ,764  ,
       958  ,961  ,682  ,149  ,204  ,807  ,923  ,961  ,555  ,139  ,82   ,550  ,912  ,948  ,810  ,277  ,76   ,428  ,874  ,948  ,739  ,277  ,   
       71   ,195  ,220  ,228  ,231  ,222  ,214  ,220  ,217  ,225  ,217  ,220  ,225  ,231  ,409  ,631  ,899  ,158  ,76   ,406  ,923  ,939  ,
       915  ,820  ,217  ,95   ,658  ,934  ,915  ,306  ,68   ,225  ,834  ,926  ,277  ,76   ,193  ,580  ,920  ,956  ,588  ,71   ,95   ,417  ,
       874  ,896  ,788  ,212  ,55   ,160  ,160  ,155  ,155  ,152  ]


mfr_2 = np.array( [0   ,700 ,1500,2500,3500,4500,4800,4850,4870,4880,4900,4920,4950,4960,4970,4980,5000,5020,5050,5070,5090,5110,5130,5160,5180,5200,5220,
         5240,5260,5270,5290,5310,5330,5350,5370,5390,5410,5430,5450,5470,5480,5500,5520,5540,5560,5580,5600,5620,5640,5660,5680,5700,5720,5740,
         5750,5780,5800,6500,7500,8500,9550,9560,9570,9590,9630,9700,9800,9930,10000,10360,10400,10420,10440,10470,10480,10510,10560,10600,10620,
         10640,10690,10760 ]) #,10800,11000,12000,13000,15000  ] )

mcx_2 = [271 ,263 ,263 ,271 ,263 ,268 ,265 ,265 ,260 ,122 ,328 ,731 ,728 ,993 ,966 ,363 ,238 ,504 ,493 ,991 ,585 ,138 ,587 ,809 ,871 ,311 ,114 ,
         539 ,555 ,963 ,704 ,146 ,117 ,344 ,338 ,982 ,731 ,136 ,490 ,525 ,999 ,512 ,125 ,463 ,587 ,991, 625 ,141 ,274 ,604 ,988 ,834 ,255 ,171 ,
         217 ,211 ,222 ,195 ,222 ,214 ,200 ,179 ,393 ,509 ,517 ,493 ,512 ,541 ,522  ,980  ,414  ,360  ,677  ,701  ,963  ,274  ,709  ,817  ,836  ,
         807  ,798  ,804  ] #,798  ,785  ,801  ,798  ,796  ]

mcy_2 = [158 ,155 ,155 ,160 ,158 ,155 ,160 ,158 ,166 ,539 ,904 ,934 ,939 ,501 ,314 ,101 ,839 ,948 ,937 ,585 ,47  ,433 ,950 ,883 ,190 ,112 ,661 , 
         950 ,934 ,672 ,120 ,323 ,425 ,923 ,926 ,631 ,98  ,420 ,948 ,958 ,371 ,49  ,555 ,937 ,958 ,539 ,71  ,347 ,880 ,948 ,615 ,166 ,160 ,701 ,
         796 ,801 ,818 ,812 ,796 ,807 ,804 ,791 ,920 ,950 ,948 ,945 ,958 ,942 ,953  ,358  ,66   ,885  ,931  ,931  ,333  ,133  ,904  ,885  ,144  ,
         141  ,149  ,141  ] #,168  ,155  ,179  ,174  ,160  ]


plt.figure()
plt.imshow( im )
plt.plot(cxs, cys, 'r.')
plt.plot(mcx_1, mcy_1, 'g.')
plt.plot(mcx_2, mcy_2, 'g.')

# plt.plot(cys, 'r.-')
# plt.plot(mfr_1,mcy_1, 'b.-')
# plt.plot(mfr_2+22773,mcy_2, 'b.-')
# plt.plot(cxs, 'g.-')
# plt.plot(mfr_1,mcx_1, 'y.-')


plt.show()

#%%

def circle_center(val):
    div = 1e5
    xte = (cxs - val[1]) 
    yte = (cys - val[2])     
    
    return np.nansum( np.abs( xte**2 / div + yte**2 / div - val[0]**2 / div ) )
    
ll = least_squares(circle_center, [450,570,550])

def angles(cxs, cys, vals ):
    return np.arctan2( -(cys-vals[2]), cxs-vals[1] )
    
ang = angles(cxs, cys, ll.x)

plt.figure()
plt.plot( ll.x[0] * np.cos(np.linspace(0,2*np.pi,10000)) +ll.x[1] , ll.x[0] * np.sin(np.linspace(0,2*np.pi,10000)) +ll.x[2], '-' )
plt.plot(ll.x[0] * np.cos(-ang) +ll.x[1] , ll.x[0] * np.sin(-ang) +ll.x[2], '.')
plt.plot(cxs, cys, 'r.')
plt.axis('equal')
plt.show()

#%%

all_fr = np.arange(len(cxs))
all_cx, all_cy = np.copy(cxs), np.copy(cys)
all_cx[mfr_1], all_cy[mfr_1] = mcx_1, mcy_1
all_cx[mfr_2+blens[1]], all_cy[mfr_2+blens[1]] = mcx_2, mcy_2

all_cx[32411], all_cy[32411] = np.nan, np.nan

# all_fr, all_cx, all_cy = all_fr[~np.isnan(all_cx)], all_cx[~np.isnan(all_cx)], all_cy[~np.isnan(all_cy)]

all_ang = angles(all_cx, all_cy, ll.x )

plt.figure()

# plt.plot( np.unwrap(ang), '.-' )
plt.plot(all_fr, np.unwrap(all_ang), '.' )
# plt.plot(all_fr, (all_ang), '.-' )

plt.grid()
plt.show()




#%%

all_frames = np.cumsum(blens)

m_frame = [0,48]
btime = np.empty(0)
for i in range(len(starts)):
    timeb = np.arange( starts[i], ends[i]  )
    timeb = ( timeb-timeb[0] + m_frame[i] + len(btime) ) / 24
    btime = np.concatenate( (btime,timeb) )

fil = np.isnan(ang)

ang_f, tim_f = ang[~fil], btime[~fil] 
uang_f = np.unwrap(ang_f)

backpos = np.unwrap(uang_f - uang_f[0]) * 2.5/(2*np.pi)


plt.figure()
# plt.plot( btime, '.-' )
# plt.plot( tim_f, ang_f, '-' )
plt.plot( tim_f, uang_f, '-' )
plt.plot( tim_f, backpos, '-' )
plt.grid()
plt.show()


np.savez(path+'back_position.npz', time=tim_f, position=backpos)



#%%



all_frames = np.cumsum(blens)

m_frame = [0,57]
btime = np.empty(0)
for i in range(len(starts)):
    timeb = np.arange( starts[i], ends[i]  )
    timeb = ( timeb-timeb[0] + m_frame[i] + len(btime) ) / 24
    btime = np.concatenate( (btime,timeb) )

fil = np.isnan(all_ang)

ang_f, tim_f = all_ang[~fil], btime[~fil] 
uang_f = np.unwrap(ang_f)

backpos = np.unwrap(uang_f - uang_f[100]) * 2.5/(2*np.pi)


plt.figure()
# plt.plot( btime, '.-' )
# plt.plot( tim_f, ang_f, '-' )
plt.plot( tim_f, uang_f, '.-' )
plt.plot( tim_f, backpos, '.-' )
plt.grid()
plt.show()


np.savez(path+'back_position.npz', time=tim_f, position=backpos)




#%%





#%%
# fit_val = [520.38439706, 523.98264292, 444.35461342]

l=0
# for i in range(0,34000,3000):
for i in [13000]:
    bvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)
    # im = np.array(bvids[l].read()[1])[:,:,::-1] #[:1900,1050:2900,::-1]
    # im = np.array(bvids[l].read()[1]) [50:1050,500:1600,::-1]
    im = np.array(bvids[l].read()[1])[blims[0]:blims[1],blims[2]:blims[3],2]
    
    plt.figure()
    plt.imshow(im * fil)
    plt.title(i)
    plt.show()    
#%%

start = 3886
end = 10060
# end = 44632 # , np.sum(blens)

starts = [3886,0]
ends = [blens[1],10060]

nfil = ~fil

t1 = time()

ratio = 0.7
bvids[0].set(cv2.CAP_PROP_POS_FRAMES, starts[0])
im = np.array(bvids[0].read()[1])[blims[0]:blims[1],blims[2]:blims[3],::-1]
# im = im[:,:,2]    
# im = normalize( gaussian( im[:,:,1], 4) )
im = normalize( gaussian( im[:,:,1]* ratio + im[:,:,2]* (1-ratio) , 3) )

cx,cy,r = fit_val
ny,nx = np.shape(im)
xxx,yyy = np.meshgrid( np.arange(nx), np.arange(ny) ) 
angs = np.arctan2( -(yyy-cy), xxx-cx )

angl = angle_dot(im, ~fil, fit_val, angs, 0, [840,330] , threshold=0.3, sizes=[80,350], ang_tol=1/2, ecc_thres=0.7 )
cxa, cya = r*np.cos(angl)+cx, cy-r*np.sin(angl)

print(angl, cxa,cya)

tot_angles, vidi = [], []

finish = False
for l in range(len(bvids)):
    counter = 0
    start,end = starts[l], ends[l]
    
    N_total = end-start
    angles = np.zeros(blens[l+1])
    bvids[l].set(cv2.CAP_PROP_POS_FRAMES, start )

    # for i in tqdm(range(blens[l+1])):
    for i in tqdm(range(N_total)):
        # counter = np.searchsorted(restarts[l], i)
        counter += 1
        if counter == end: 
            finish=True
            break
        
        im = np.array(bvids[l].read()[1])[blims[0]:blims[1],blims[2]:blims[3],::-1]
        im = normalize( gaussian( im[:,:,1]* ratio + im[:,:,2]* (1-ratio) , 3) )
        
        ang_ini = angl
        threshold= 0.3
        sizes=[80,350]
        ecc_thres = 0.8 #1
        min_dist = 20000
        ang_tol = 8
        
        if angl > 0: threshold = 0.2
        
        if l == 0:
            if -2.1 < angl < -1.5: threshold = 0.35   #works for everything in vid 1

        elif l == 1:
            if -2.1 < angl < -1.5: threshold = 0.3   #try second vid
            if i in [63141, 63142, 63143]: threshold = 0.2 # 63141, 63142, 63143
            if i in [63144, 63145, 63146, 63147, 63148]: threshold = 0.25 # 63144, 63145, 63146, 63147, 63148

        
        
        # angl = angle_dot(im, nfil, fit_val, angs, angl, [cya,cxa] , threshold=0.3, sizes=[80,350], ang_tol=8, ecc_thres=0.7 )
        angl = angle_dot(im, ~fil, fit_val, angs, angl, [cya,cxa],  threshold=threshold, sizes=sizes, ang_tol=ang_tol, ecc_thres=ecc_thres  )

        angles[i] = angl
        cxa, cya = r*np.cos(angl)+cx, cy-r*np.sin(angl)
        
        vidi.append(l)

    tot_angles.append( angles )
    
    if finish: break
    

tot_angles = np.hstack(tot_angles)
vidi = np.array(vidi).astype('int')

frames_b = np.hstack([[0],np.cumsum(missing_frames)])
times_b = ( np.arange(len(vidi)) + frames_b[vidi] ) / 60

# cuts = np.cumsum(ends-starts)
# bvideo = np.searchsorted(cuts, np.arange(68576), side='right')

t2 = time()
print(t2-t1)

#%%
# # np.save(path+'/angles(not_added).npy', tot_angles)
# np.save(path+'back_angles.npy', tot_angles)
# np.save(path+'back_times.npy', times_b)


# #%%
# ttt = np.load( path+'back_times.npy' )
# aaa = np.load( path+'back_angles.npy' )
# aab = aaa[aaa!=0]

# backpos = np.unwrap(aab - aab[0]) * 2.5/(2*np.pi)

# plt.figure()
# # plt.plot(aaa==0)
# # plt.plot( np.unwrap(aab - aab[0]) / (2*np.pi) )#* 2.5/(2*np.pi) )
# plt.plot(ttt, backpos, '.-' )
# # plt.plot( aab )#* 2.5/(2*np.pi) )
# plt.grid()
# plt.show()

# np.savez(path+'back_position.npz', time=ttt, position=backpos)

#%%
l = 0
i0 = 7064
bvids[l].set(cv2.CAP_PROP_POS_FRAMES, i0)
im = np.array(bvids[l].read()[1])[blims[0]:blims[1],blims[2]:blims[3],::-1]
# im = im[:,:,1]    
# im = normalize( gaussian( im[:,:,1], 4) )
ratio = 0.7
im = normalize( gaussian( im[:,:,1]* ratio + im[:,:,2]* (1-ratio) , 3) )

cx,cy,r = fit_val
ny,nx = np.shape(im)
xxx,yyy = np.meshgrid( np.arange(nx), np.arange(ny) ) 
angs = np.arctan2( -(yyy-cy), xxx-cx )

angl = angle_dot(im, ~fil, fit_val, angs, 0, [0,0],  threshold=0.3, sizes=[80,350], ang_tol=1/2, ecc_thres=0.8 )
cxa, cya = r*np.cos(angl)+cx, cy-r*np.sin(angl) 
print(angl)

cx,cy,r = fit_val
imt = np.copy(im)
imt[~fil] = 0
        
angs_arr = angs
ang_ini = 0
ang_tol = 12
c_ini = [0,0]
sizes=[80,350]
ecc_thres=0.8
threshold=0.3

angs_rotated = (angs_arr - ang_ini + 1*np.pi) % (2 * np.pi) - np.pi
fila = ( angs_rotated < np.pi/ang_tol) * ( angs_rotated > -np.pi/ang_tol)
imt[~fila] = 0

la = label(imt>threshold)
dict_prop = regionprops_table(la, properties=['area','centroid','eccentricity']) #to make it with idst t previous
area_fil = (sizes[0]<dict_prop['area'])&(dict_prop['area']<sizes[1])
ecc_fil = dict_prop['eccentricity'] < ecc_thres
dist = (dict_prop['centroid-0']-c_ini[0])**2 + (dict_prop['centroid-1']-c_ini[1])**2  

plt.figure()
# plt.imshow( im )
plt.imshow( imt )
plt.plot( cxa,cya, 'r.' )
plt.title(i0)
plt.show()

# plt.figure()
# plt.imshow( la )
# plt.plot( cxa,cya, 'r.' )
# plt.colorbar()
# plt.title(i0)
# plt.show()

#%%
# i0 = 3886 + 5800 + 1000 + 3200 + 490 + 3950 + 400 + 3960 + 370 + 3540 + 780 + 5510
i0 = 2500 + 5690
l = 1

ratio = 0.7
bvids[l].set(cv2.CAP_PROP_POS_FRAMES, i0)
im = np.array(bvids[l].read()[1])[blims[0]:blims[1],blims[2]:blims[3],::-1]
# im = normalize( gaussian( im[:,:,1], 4) )
im = normalize( gaussian( im[:,:,1]* ratio + im[:,:,2]* (1-ratio) , 3) )

cx,cy,r = fit_val
ny,nx = np.shape(im)
xxx,yyy = np.meshgrid( np.arange(nx), np.arange(ny) ) 
angs = np.arctan2( -(yyy-cy), xxx-cx )
# angl = angle_dot(im, ~fil, fit_val, angs, 0, [840,330],  threshold=0.3, sizes=[80,350], ang_tol=1/2, ecc_thres=0.8 )
angl = angle_dot(im, ~fil, fit_val, angs, 0, [740,120],  threshold=0.3, sizes=[80,350], ang_tol=1/2, ecc_thres=0.8 )
cxa, cya = r*np.cos(angl)+cx, cy-r*np.sin(angl) 

print(angl)
# plt.figure()
# plt.imshow( im )
# plt.plot( cxa,cya, 'r.' )
# plt.title(i0)
# plt.show()
# # %%

detailed = 1

cccc = []
bvids[l].set(cv2.CAP_PROP_POS_FRAMES, i0)    
# for i in range(16161,16260):
# for i in [i0+1]:
for i in tqdm(range(0,1870,1), disable=False ):

    # bvids[l].set(cv2.CAP_PROP_POS_FRAMES, i0 + i)    
    im = np.array(bvids[l].read()[1])[blims[0]:blims[1],blims[2]:blims[3],::-1]
    # im = im[:,:,2]    
    # im = normalize( gaussian( im[:,:,1], 4) )
    im = normalize( gaussian( im[:,:,1]* ratio + im[:,:,2]* (1-ratio) , 3) )
        
    if 1-detailed: 

        ang_ini = angl
        threshold= 0.3
        sizes=[80,350]
        ecc_thres = 0.8 #1
        min_dist = 20000
        ang_tol = 8               

        # if i0+i in [9740]: ecc_thres = 0.78 # 9740
        # if i0+i in [9742]: sizes = [80,450] # 9742
        # if i0+i in [9758,9759,9760]: ang_tol = 8 # 9758,9759,9760
        
        angl = angle_dot(im, ~fil, fit_val, angs, angl, [cya,cxa],  threshold=threshold, sizes=sizes, ang_tol=ang_tol, ecc_thres=ecc_thres  )

        cxa, cya = r*np.cos(angl)+cx, cy-r*np.sin(angl) 
        cccc.append(angl)
    
    if detailed: 
        ang_ini = angl
        threshold = 0.3
        # sizes = [80,350]
        sizes = [210,500]
        ecc_thres = 0.8 #1
        min_dist = 20000
        ang_tol = 8
        
        # starting 3886 + 5800
        # if i in [71,72,73,74]: sizes, ecc_thres = [80,500], 0.94
        # elif i in [90,91,92,93,94,95,96]: sizes, ecc_thres = [230,500], 0.9
        # elif i in [200,201]: sizes, ecc_thres = [270,500], 0.7
        # elif i in [244,245]: sizes, ecc_thres = [120,400], 0.9
        # elif i in [246,247]: sizes, ecc_thres = [120,600], 0.92
        # elif i in [295,296]: sizes, ecc_thres = [200,600], 0.92
        # elif i in [307]: sizes, ecc_thres = [240,600], 0.92
        # elif i in [341,342]: sizes, ecc_thres = [300,600], 0.92
        # elif i in [343,344,345,346]: sizes, ecc_thres = [401,600], 0.92
        # elif i in [372,373,374,375]: sizes, ecc_thres = [401,600], 0.89
        # elif i in [376,377]: sizes, ecc_thres = [340,600], 0.89
        # elif i in [404,405]: sizes, ecc_thres = [270,600], 0.91
        # elif i in [406,407]: sizes, ecc_thres = [285,360], 0.93
        # elif i in [451,452]: sizes, ecc_thres = [320,600], 0.9
        # elif i in [466]: sizes, ecc_thres = [320,550], 0.9
        # elif i in [479,480]: sizes, ecc_thres = [320,550], 0.9
        # elif i in [495,496,497]: sizes, ecc_thres = [230,550], 0.9
        # elif i in [526,527,528]: sizes, ecc_thres = [230,550], 0.9
        # elif i in [573]: sizes, ecc_thres = [230,550], 0.9
        # elif i in [590,591]: sizes, ecc_thres = [240,550], 0.9
        # elif i in [650,651]: sizes, ecc_thres = [300,560], 0.9
        # elif i in [682]: sizes, ecc_thres = [290,560], 0.9
        # elif i in [683,684,685,686]: sizes, ecc_thres = [200,250], 0.9
        # elif i in [743,744]: sizes, ecc_thres = [450,600], 0.9
        # elif i in [772,773,774]: sizes, ecc_thres = [300,500], 0.9
        # elif i in [787,788]: sizes, ecc_thres = [200,500], 0.75
        # elif i in [789]: sizes, ecc_thres = [210,500], 0.9
        # elif i in [828]: sizes, ecc_thres = [300,400], 0.9
        # elif i in [829]: sizes, ecc_thres = [400,500], 0.9
        # elif i in [880,881]: sizes, ecc_thres = [280,500], 0.9
        # elif i in [912,913]: sizes, ecc_thres = [300,500], 0.9
        # elif i in [931,932,933,934,935,935,936,937,938,939,940]: sizes, ecc_thres = [200,500], 0.9
        
        # #starting 3886 + 5800 + 1000 + 3200
        # if i in [159,160,161,175,192,219,267,268,269,270,300,301,318,349,379,380,433,435,464,465,466]: sizes, ecc_thres = [270,500], 0.94
        # elif i in [189]: sizes, ecc_thres = [420,460], 0.9
        # elif i in [190]: sizes, ecc_thres = [420,560], 0.92
        # elif i in [191]: sizes, ecc_thres = [1140,1150], 0.95
        # elif i in [204]: sizes, ecc_thres = [600,700], 0.9
        # elif i in [220,221]: sizes, ecc_thres = [200,450], 0.9
        # elif i in [235,236,319]: sizes, ecc_thres = [290,400], 0.91
        # elif i in [351]: sizes, ecc_thres = [400,550], 0.83
        # elif i in [381]: threshold, sizes = 0.32, [200,400]
        # elif i in [398]: sizes, ecc_thres = [283,400], 0.94
        # elif i in [399]: sizes, ecc_thres = [200,280], 0.94
        # elif i in [413,414]: sizes, ecc_thres = [100,300], 0.94
        # elif i in [434]: sizes, ecc_thres = [380,390], 0.94
    
        # # #starting 3886 + 5800 + 1000 + 3200 + 490 + 3950
        # if i in [39,40,41,116,145,195,196,228,282,382]: sizes, ecc_thres = [250,500], 0.94
        # elif i in [72,75]: sizes, ecc_thres = [300,450], 0.9
        # elif i in [86,87,88]: sizes, ecc_thres = [300,600], 0.9
        # elif i in [148]: sizes, ecc_thres = [400,500], 0.9
        # elif i in [229]: sizes, ecc_thres = [351,360], 0.9
        # elif i in [283,284]: sizes, ecc_thres = [200,400], 0.73
        # elif i in [383]: sizes, ecc_thres = [234,400], 0.9
        
        # # # #starting 3886 + 5800 + 1000 + 3200 + 490 + 3950 + 400 + 3960
        # if i in [86,87,88,156,252,253,294,340]: sizes, ecc_thres = [270,500], 0.94
        # elif i in [127,128,129,130,207,296,307,309,308,310]: sizes, ecc_thres = [400,600], 0.9
        # elif i in [143]: threshold, sizes = 0.32, [200,400]
        # elif i in [158,159,172]: sizes, ecc_thres = [200,240], 0.9
        # elif i in [190]: sizes, ecc_thres = [200,400], 0.9
        # elif i in [338]: sizes, ecc_thres = [250,260], 0.9
        # elif i in [339]: sizes, ecc_thres = [270,280], 0.9
        
        # # #starting 3886 + 5800 + 1000 + 3200 + 490 + 3950 + 400 + 3960 + 370 + 3540 + 780 + 5510
        # if i in [57,222,265,364,377,378,412]: sizes, ecc_thres = [270,500], 0.94
        # elif i in [20]: sizes, ecc_thres = [240,500], 0.9
        # elif i in [38,39]: sizes, ecc_thres = [240,380], 0.9
        # elif i in [121,122]: sizes, ecc_thres = [210,240], 0.9
        # elif i in [177,178]: sizes, ecc_thres = [450,520], 0.9
        # elif i in [223]: threshold = 0.25
        # elif i in [266]: sizes, ecc_thres = [400,600], 0.9
        # elif i in [311,413]: sizes, ecc_thres = [260,600], 0.9
        # elif i in [314]: sizes, ecc_thres = [250,600], 0.9
        # elif i in [315]: sizes, ecc_thres = [220,230], 0.9
        
        # # # #starting second video: 2500
        # if i in [82,83,150,151,307,308]: sizes, ecc_thres = [270,500], 0.94
        # elif i in [108]: sizes, ecc_thres = [420,450], 0.9
        # elif i in [109,181,182,260,261,353]: sizes, ecc_thres = [400,600], 0.92
        # elif i in [110]: sizes, ecc_thres = [700,800], 0.92
        # elif i in [111]: sizes, ecc_thres = [560,600], 0.92
        # elif i in [226,227]: sizes, ecc_thres = [241,251], 0.92
        # elif i in [239]: sizes, ecc_thres = [200,300], 0.92

        # # #starting second video: 2500 + 5690
        if i in [51,52]: sizes, ecc_thres = [270,500], 0.94
        # elif i in [108]: sizes, ecc_thres = [420,450], 0.9

        
        cx,cy,r = fit_val
        imt = np.copy(im)
        imt[~fil] = 0
        angs_rotated = (angs - ang_ini + 1*np.pi) % (2 * np.pi) - np.pi
        fila = ( angs_rotated < np.pi/ang_tol) * ( angs_rotated > -np.pi/ang_tol)
        imt[~fila] = 0
        la = label(imt>threshold)    
        dotc = np.array([np.nan, np.nan])    
        
        dict_prop = regionprops_table(la, properties=['area','centroid','eccentricity'])
        area_fil = (sizes[0]<dict_prop['area'])&(dict_prop['area']<sizes[1])
        ecc_fil = dict_prop['eccentricity'] < ecc_thres
        dist = (dict_prop['centroid-0']-cya)**2 + (dict_prop['centroid-1']-cxa)**2  
        
        # if i in [51,52,53]: print('\n',i,dict_prop)
        
        wight = (1-area_fil*1)*50 + (1-ecc_fil*1)*50 + dist.argsort().argsort()
        
        dot = np.argmin( wight )
            
        dotc[0] = dict_prop['centroid-0'][ dot ]
        dotc[1] = dict_prop['centroid-1'][ dot ]
    
        angl = np.arctan2( -(dotc[0]-cy), dotc[1]-cx  )
        cxa, cya = r*np.cos(angl)+cx, cy-r*np.sin(angl) 
        cccc.append(angl)
            
        
        # if i == 353 :
        # if i%400 == 0:
        # if i >= 90:
        if (i >= 90) & (i%100==0):
            plt.figure()
            plt.imshow( la )
            # plt.imshow( im )
            plt.imshow( imt )
            # plt.plot( dotc[1], dotc[0], 'r.' )
            plt.plot( cxa, cya, 'r.' )
            plt.title(i)
            plt.show()

#%%


plt.figure()
# plt.plot( angles[3900:9000-1], '.-' )
plt.plot( cccc,'.-')

# plt.plot( np.unwrap(cccc),'.-')
# plt.plot( np.unwrap(tot_angles[16160:16260][1:]) ,'.-')

plt.grid()
plt.show()

np.save(path+'back_angles_1(15).npy', np.array(cccc))


#%%

angs1, angs2 = [], []
for i in range(1,13): angs1.append( np.load( path+f'back_angles({i}).npy' ) )
for i in range(13,16): angs2.append( np.load( path+f'back_angles_1({i}).npy' ) )

angs1, angs2 = np.hstack( angs1 ), np.hstack( angs2 )
t1, t2 = np.arange( len(angs1) ) / 24,  np.arange( len(angs2) ) / 24
t2 = t2 + t1[-1] + 55/24

t_tot = np.hstack( [t1,t2] )
ang_tot = np.hstack( [angs1, angs2] )

backpos = (np.unwrap( ang_tot ) - ang_tot[0] ) /(2*np.pi) * 2.5

plt.figure()
plt.plot( t1, angs1 )
plt.plot( t2, angs2 )
# plt.plot( t_tot, ang_tot, '--')
plt.plot( t_tot, backpos )
plt.show()

# np.savez(path+'back_position.npz', time=t_tot, position=backpos)

#%%
l = 0
i0 = 9856 

bvids[l].set(cv2.CAP_PROP_POS_FRAMES, i0)
im = np.array(bvids[l].read()[1])[blims[0]:blims[1],blims[2]:blims[3],::-1]
# im = normalize( gaussian( im[:,:,1], 4) )
im = im[:,:,1]* 0.7 + im[:,:,2]* 0.3

plt.figure()
plt.imshow( im )
plt.show() 

im2 = np.array(bvids[l].read()[1])[blims[0]:blims[1],blims[2]:blims[3],::-1]
# im2 = normalize( gaussian( im2[:,:,1], 4) )
im2 = im2[:,:,1]* 0.7 + im2[:,:,2]* 0.3

plt.figure()
plt.imshow( im2 )
plt.show() 

# plt.figure()
# plt.imshow( im2 - im )
# plt.show() 




#%%
# All together

tot_angles = np.load(path+'/angles(not_added).npy')

plt.figure()
# plt.plot(tot_angles[:],'.-')
plt.plot( np.unwrap(tot_angles[:]),'.-')
plt.grid()
plt.show()



start = 1034
tot_angles = np.load(path+'/angles(not_added).npy')
times = np.load(path+'/back_times.npy')

backpos = np.unwrap(tot_angles - tot_angles[start+100]) * 2.5/(2*np.pi)

plt.figure()
plt.plot( times, backpos, '.-' )
# plt.vlines(np.cumsum(blens), 0, 2500, color='gray')
plt.xlabel('Time (s)')
plt.ylabel('Back position (mm)')
plt.grid()
plt.show()


np.savez(path+'back_position.npz', time=times, position=backpos)

#%%


#%%


#%%


#%%

path = '/Volumes/Ice blocks/Scan water channel/25-08-07/' # this one has 30 fps

bvid_cal = cv2.VideoCapture( path + 'Camera back/DSC_0435.MOV')

bvids = [cv2.VideoCapture( path + 'Camera back/DSC_0436.MOV'), #starts 3486 
         cv2.VideoCapture( path + 'Camera back/DSC_0437.MOV'), 
         cv2.VideoCapture( path + 'Camera back/DSC_0438.MOV'), 
         cv2.VideoCapture( path + 'Camera back/DSC_0439.MOV'), 
         cv2.VideoCapture( path + 'Camera back/DSC_0440.MOV'), 
         cv2.VideoCapture( path + 'Camera back/DSC_0441.MOV'), 
         cv2.VideoCapture( path + 'Camera back/DSC_0442.MOV'), 
         cv2.VideoCapture( path + 'Camera back/DSC_0443.MOV'), #ends 5180
          ]

blens = [0]+[int(bvids[i].get(cv2.CAP_PROP_FRAME_COUNT)) for i in range(len(bvids))]
#%%

l = 0
for i in range(4924,4925,1):
    bvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)

    im = np.array(bvids[l].read()[1])[:,:,::-1] #[:1900,1050:2900,::-1]
    # im = np.array(bvids[l].read()[1])[1540:1600,880:930,::-1] #[150:2100,750:2800,::-1]
    # im = np.array(bvids[l].read()[1])[:,:,::-1] #[1540:1595,880:930,::-1] #[150:2100,750:2800,::-1]
    im = im[:,:,1]
    
    plt.figure()
    # plt.imshow(im)
    plt.imshow(im, vmax=40)
    plt.title(i)
    plt.show()    

#     # print( i,'\t{:.2f}\t'.format(np.mean(im)), np.median(im) )
#%%
# calibration
from scipy.ndimage import rotate

i = 222 #723
bvid_cal.set(cv2.CAP_PROP_POS_FRAMES, i)
im = np.array(bvid_cal.read()[1])[:,:,::-1] 
imr = rotate(im, -2.)

pixels = [3392, 3207, 3024, 2845, 2669, 2493, 2321, 2150, 1982, 1816, 1655, 1495]
dist = [150,140,130,120,110,100,90,80,70,60,50,40]

linr = linregress(pixels, dist)
slo, ori = linr[0], linr[1]

plt.figure()
# plt.imshow(imr[:,:,1], vmin=40, vmax=100)
plt.imshow(imr)
plt.plot( pixels, [1000]*12, 'r.' )
plt.title(i)
plt.show()


plt.figure()
plt.plot( pixels, dist, '.' )
plt.plot( np.linspace(800,3500,100), np.linspace(800,3500,100)*slo+ori, '--' )
plt.show()

#%%

start = 3486
inds = np.arange(start, 50156, 1000)
vid_ind = np.searchsorted(np.cumsum(blens), inds) - 1
missing_frames_vid = np.array( [0,0,0,0,0,0,0,1305] )

ind_in_vid = inds - np.cumsum(blens)[vid_ind]
totframes = inds + missing_frames_vid[vid_ind]

for i in [46]:
    l = vid_ind[i]
    frame = ind_in_vid[i]
    tot_frame = frame + np.cumsum(blens)[l]

    print(l, frame, tot_frame )
    bvids[l].set(cv2.CAP_PROP_POS_FRAMES, frame)
    im = np.array(bvids[l].read()[1])[:,:,::-1] 
    imr = rotate(im, -2.)
    imr = imr[:,:,1]
    
    plt.figure()
    plt.imshow(imr, vmax=200)
    plt.title( frame )
    plt.show()    
#%%
fps = 25
# inex = [3486, 4486, 5486, 6486, 7486, ]

pixpos = [3433, 3430, 3275, 2998, 2998, 2996, 2792, 2789, 2790, 2750, 2640, 2640, np.nan, 2645, 2535, 2530, 2345, 2355, 2190,
          2172, 2170, 2171, 1882, np.nan, 1882, 1885, 1886, 1887, 1764, 1763, 1765, 1761, np.nan, 1582, 1581, 1583, np.nan, 1581,
          1465, 1464, 1465, 1475, 1267, 1276, 1270, 1215, 1220,  ]
pixpos = np.array(pixpos)


mov_v = np.array( [0,0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4, 5,5,5, 6,6,6, 7,7,7 ] )
mov_f = np.array( [3486, 4924, 5225, 5983, 6398, 1788, 2184, 5056, 5376, 2760, 2984, 5035, 5415, 637 , 916 , 4335, 4758, 
                   3467, 3706, 1535, 1903, 6577, 135 , 3598, 3939, 2876, 3030, 5180 ])
mov_ff = mov_f + missing_frames_vid[mov_v] + np.cumsum(blens)[mov_v]

mov_p = np.array( [3432, 3432, 3275, 3275, 2997, 2997, 2790, 2790, 2640, 2640, 2532, 2532, 2350, 2350, 2172, 2172, 1884, 
                   1884, 1764, 1764, 1582, 1582, 1464, 1464, 1270, 1270, 1215, 1215 ])


times = (mov_ff - mov_ff[0] ) / fps / 60
backpos = mov_p*slo+ori
backpos = -(backpos - backpos[0])

plt.figure()
# plt.plot( totframes, pixpos, '.-' )
# plt.plot( mov_ff, mov_p, '.-' )
# plt.vlines(np.cumsum(blens), 1100,3500, colors='gray')

# plt.plot( totframes / fps / 60, pixpos*slo+ori, '.-' )
# plt.plot( mov_ff / fps / 60, mov_p*slo+ori, '.-' )
# plt.vlines(np.cumsum(blens) / fps/ 60, 15,160, colors='gray')

plt.plot( times, backpos, '.-' )

plt.grid()
plt.show()



np.savez(path+'back_position.npz', time=times, position=backpos)


#%%
l = 7
frame = 3030

bvids[l].set(cv2.CAP_PROP_POS_FRAMES, frame)
im = np.array(bvids[l].read()[1])[:,:,::-1] 
imr = rotate(im, -2.)
imr = imr[:,:,1]

plt.figure()
plt.imshow(imr, vmax=200)
plt.title( frame )
plt.show()    

vvv = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4, 5,5,5, 6,6,6, 7,7 ]
fff = [4924, 5225, 5983, 6398, 1788, 2184, 5056, 5376, 2760, 2984, 5035, 5415, 637 , 916 , 4335, 4758, 3467, 3706, 1535, 1903, 6577, 135 , 3598, 3939, 2876, 3030 ]

ppp = [3457, 3291, 3296, 3021, 3020, 2827, 2825, 2675, 2660, 2546, 2542, 2360, 2358, 2190, 2180, 1898, 1897, 1798, 1775, 1596, 1602, 1482, 1481, 1284, 1285, 1234 ]

#%%
# =============================================================================
# Check back_position.npz
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt


plt.figure()

path = '/Volumes/Ice blocks/Scan water channel/25-10-24/'
bp = np.load(path+'back_position.npz')
print(bp)
plt.plot( bp['time'], bp['position'], '.-', label=path[-9:-1] )

path = '/Volumes/Ice blocks/Scan water channel/25-11-17/'
bp = np.load(path+'back_position.npz')
print(bp)
plt.plot( bp['time'], bp['position'], '.-', label=path[-9:-1] )

path = '/Volumes/Ice blocks/Scan water channel/25-12-12/'
bp = np.load(path+'back_position.npz')
print(bp)
plt.plot( bp['time'], bp['position'], '.-', label=path[-9:-1] )

path = '/Volumes/Ice blocks/Scan water channel/26-01-27/'
bp = np.load(path+'back_position.npz')
print(bp)
plt.plot( bp['time'], bp['position'], '.-', label=path[-9:-1] )

path = '/Volumes/Ice blocks/Scan water channel/26-02-17/'
bp = np.load(path+'back_position.npz')
print(bp)
plt.plot( bp['time'], bp['position'], '.-', label=path[-9:-1] )

plt.legend()
# plt.show()

# plt.figure()

path = '/Volumes/Ice blocks/Scan water channel/25-09-29/'
bp = np.load(path+'back_position.npz')
print(bp)
# plt.plot( bp['arr_0'], bp['arr_1'], '.-' )
plt.plot( bp['time'], bp['position'], '.-' )

plt.show()

#%%

path = '/Volumes/Ice blocks/Scan water channel/25-09-29/'
bp = np.load(path+'back_plate_position.npz')

np.savez(path+'back_position.npz', time=bp['arr_0'], position=bp['arr_1'])



