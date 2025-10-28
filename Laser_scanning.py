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


def calib_invzl(v, points, coord, x0s=[[50,300]], dis_bar=False, method='SLSQP'):
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

def cuad(x,a,b,c):
    return a * x**2 + b * x + c 

def fit_wall_pixels(ili, sma, distance=100):
    order = np.argsort(sma)
    da = np.where( np.diff(sma[order]) > distance )[0]+1
    os = np.split(order,da)
    fma = np.zeros_like(sma)
    for j in range(len(os)):
        if len(os[j]) > 2:
            (a,b,c),cov = curve_fit(cuad, ili[os[j]], sma[os[j]])
            fma[ili[os[j]]] = cuad(ili[os[j]], a,b,c)
        else: fma[ili[os[j]]] = sma[ili[os[j]]]

    return fma


def wall_d_y(y, y0s, x0s):
    m = (x0s[1] - x0s[0]) / (y0s[1] - y0s[0])
    return m * (y - y0s[0]) + x0s[0]

# def ice_boundary(ili, smi, sma, x0s, y0s, wall_distace, dis_bar=True, method='SLSQP'):
#     wall_d = np.ones_like(ili) * wall_distace #distance from grid to window in mm (from fisrt point)
#     points = np.vstack((sma,ili,wall_d)).T
#     yr, zr = calib_invzl(cal_up, points, 0, dis_bar=dis_bar, method=method)
    
#     # # dwall = wall_d_y(yr, [294,-390], [-7.5,-3])
#     # dwall = wall_d_y(yr, x0s, y0s)
#     # points = np.vstack((sma,ili,dwall)).T
#     # yc, zc = calib_invzl(cal_up, points, 0, dis_bar=dis_bar)
#     yc, zc, dwall = yr, zr, wall_d
    
#     points = np.vstack((smi,ili,zc)).T
#     xi, yi = calib_invzl(cal_up, points, 2, dis_bar=dis_bar, method=method)
    
#     return dwall, yc, zc, xi, yi

def ice_boundary(ili, smi, sma, wall_distace, calib, dis_bar=True, method='SLSQP'):
    wall_d = np.ones_like(ili) * wall_distace #distance from grid to window in mm (from fisrt point)
    points = np.vstack((sma,ili,wall_d)).T
    yr, zr = calib_invzl(calib, points, 0, dis_bar=dis_bar, method=method)    
    yc, zc, dwall = yr, zr, wall_d
    
    points = np.vstack((smi,ili,zc)).T
    xi, yi = calib_invzl(calib, points, 2, dis_bar=dis_bar, method=method)
    
    return dwall, yc, zc, xi, yi


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

def frame_position(f_mes, lens, lims, divisor=2, threshold=10, minute_interval=1798, n_recons=60):
    """
    Returns the frames where to start the recontruction of each profile

    Parameters
    ----------
    f_mes : array
        Mean brighness of image at LED position for each frame .
    lens : list
        List of lengths (in frames) of each video.
    lims : [start,end]
        starting and ending frames of experiment.
    divisor : int, optional
        How many profiles to be reconstructed per minute. The default is 2.
    threshold : float, optional
        Intensity threshold for f_mes. If f_mes>threshold then the led in that frame is considered lit. The default is 10.
    minute_interval : int, optional
        Number of frames per minute (more specifically frames between LED blinks). The default is 1798.
    n_recons : int, optional
        Number of frames to use for surface reconstruction. The default is 60.

    Returns
    -------
    all_frames : array
        Frames where to start reconstruction.
    vid: array
        Video in which all_frames are found.
    frames_led : array
        Frames where LED turns on.
    """
    start,end = lims[0], lims[1]
    lens = lens[1:]
    frames_led = np.where(np.diff((f_mes>threshold)*1.)>0.5)[0]+1
    
    cuts = np.cumsum(lens)
    gaps = np.where(np.abs(np.diff(frames_led) - minute_interval) > 2)[0] #cut in video between this blink and next blink
        
    missing_frames = minute_interval - np.diff(frames_led)[gaps]
    missing_pos = np.searchsorted(cuts, frames_led[gaps])
    
    tcuts = np.copy(cuts)
    for i in range(len(missing_frames)): tcuts[cuts>cuts[missing_pos[i]]] += missing_frames[i]
    
    tvid = np.zeros(cuts[-1]+np.sum(missing_frames))
    for i in range(len(tcuts)-1): tvid[tcuts[i]:tcuts[i+1]] = i+1
    for i in range(len(missing_frames)): tvid[tcuts[missing_pos[i]]:tcuts[missing_pos[i]]+missing_frames[i]] = np.nan
    
    tframes_led = np.copy(frames_led)
    for i in range(len(missing_frames)): tframes_led[frames_led>cuts[missing_pos[i]]] += missing_frames[i]
    
    dist_frames = minute_interval/divisor
    intermediate_frames = []
    for i in range(1,divisor):
        intermediate_frames.append( tframes_led + int(dist_frames*i) )
    
    tall_frames = np.sort( np.hstack((tframes_led, np.hstack(intermediate_frames) )) )
    all_frames = np.copy(tall_frames)
    for i in range(len(missing_frames)): all_frames[tall_frames>tcuts[missing_pos[i]]] += -missing_frames[i]
    
    fil = (all_frames>=start) * (all_frames<=end+np.sum(missing_frames))
    all_frames = all_frames[fil]
    
    if np.isnan( np.sum(tvid[tall_frames]) ) or np.isnan( np.sum(tvid[tall_frames+n_recons]) ):
        print('One or more profiles cannot be fully reconstructed')
        plt.figure()
        plt.vlines(tall_frames,0,len(lens)-1, colors='red',label='Start frame',alpha=0.5)
        plt.vlines(tall_frames+n_recons,0,len(lens)-1, colors='green',label='End frame',alpha=0.5)
        plt.plot( np.arange(len(tvid)), tvid,'.-' )
        plt.legend()
        plt.show()
    else: print('No issues')        
        
    return all_frames, tvid[all_frames].astype(int), frames_led


#%%
# 30 fps

path = '/Volumes/Ice blocks/Scan water channel/25-09-29/'

data = np.load(path+'calibration_data.npz')
angle_xy, angle_yz, angle_xz = float(data['arr_3']), float(data['arr_4']), float(data['arr_5'])

cal_up = data['arr_0']
cal_do = data['arr_1']
wall_distance = float(data['arr_2'])


dvids = [cv2.VideoCapture( path + 'Camera down/DSC_6033.MOV'), # starts 3317
         cv2.VideoCapture( path + 'Camera down/DSC_6034.MOV'), # 
         cv2.VideoCapture( path + 'Camera down/DSC_6035.MOV'), #
         cv2.VideoCapture( path + 'Camera down/DSC_6036.MOV'), #
         cv2.VideoCapture( path + 'Camera down/DSC_6037.MOV') # ends in last frame
         ]

uvids = [cv2.VideoCapture( path + 'Camera up/DSC_9559.MOV'), # starts 3381
         cv2.VideoCapture( path + 'Camera up/DSC_9560.MOV'), #
         cv2.VideoCapture( path + 'Camera up/DSC_9561.MOV'), #
         cv2.VideoCapture( path + 'Camera up/DSC_9562.MOV') # ends in last frame
         ]

dlens = [0]+[int(dvids[i].get(cv2.CAP_PROP_FRAME_COUNT)) for i in range(len(dvids))]
ulens = [0]+[int(uvids[i].get(cv2.CAP_PROP_FRAME_COUNT)) for i in range(len(uvids))]

#fps 30, (supposedly 29.97)
#%%
l = 0

for i in range(3315,3318,2):
    dvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)      
    im = np.array( dvids[l].read()[1] )[:,:,::-1]
    im = grayscale_im(im)
    plt.figure()
    plt.imshow(im, cmap='gray') 
    plt.title(i)
    plt.show()

# for i in range(3380,3386,2):
#     uvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)
#     im = np.array( uvids[l].read()[1] )[:,:,::-1]
#     # im = grayscale_im(im)
#     plt.figure()
#     plt.imshow(im, cmap='gray') 
#     plt.title(i)
#     plt.show()



#%%
# Led finding
d_led = [122,158,2656,2692] 
u_led = [1895,1933,2891,2926] 

u_mes, d_mes = [],[]

for l in range(len(dvids)):
    dvids[l].set(cv2.CAP_PROP_POS_FRAMES, 0)
    for j in tqdm(range(dlens[l])):
        im = np.array( dvids[l].read()[1] )[d_led[0]:d_led[1],d_led[2]:d_led[3]] 
        im = grayscale_im(im)
        d_mes.append(np.mean(im))

for l in range(len(uvids)):
    uvids[l].set(cv2.CAP_PROP_POS_FRAMES, 0)
    for j in tqdm(range(ulens[l])):
        im = np.array( uvids[l].read()[1] )[u_led[0]:u_led[1],u_led[2]:u_led[3]] 
        im = grayscale_im(im)
        u_mes.append(np.mean(im))

np.savez(path+'led_blink.npz', u_mes=u_mes, d_mes=d_mes)

#%%

blink = np.load(path+'led_blink.npz')
u_mes,d_mes = blink['u_mes'], blink['d_mes']

plt.figure()
plt.plot( u_mes, '.-')
plt.plot( d_mes, '.-')
plt.show()

#%%

N = 60
shift = 64
end = np.min([np.sum(ulens),np.sum(dlens)])
d_frame, d_vid, d_leds = frame_position(d_mes, dlens, [3317,end], n_recons=N)
u_frame, u_vid, u_leds = frame_position(u_mes, ulens, [3381,end], n_recons=N)
d_cuts, u_cuts = np.cumsum(dlens), np.cumsum(ulens)

d_frame, u_frame = d_frame+shift, u_frame+shift
d_vid, u_vid = np.searchsorted(d_cuts[1:], d_frame), np.searchsorted(u_cuts[1:], u_frame)

d_Nvid, u_Nvid = np.searchsorted(d_cuts[1:], d_frame+N),  np.searchsorted(u_cuts[1:], u_frame+N)
if np.sum(u_Nvid-u_vid) > 0:
    ind = np.where( (u_Nvid-u_vid)>0 )[0]
    overshoot = u_frame[ind]+N - u_cuts[u_vid[ind]+1]     
    print(f'Issue with u at {ind}. Number of frames in next video: {overshoot}')
if np.sum(d_Nvid-d_vid) > 0:
    ind = np.where( (d_Nvid-d_vid)>0 )[0]
    overshoot = d_frame[ind]+N - d_cuts[d_vid[ind]+1]
    print(f'Issue with d at {ind}. Number of frames in next video: {overshoot}')

print(end, d_frame[-1]+N,  u_frame[-1]+N)

plt.figure()
plt.plot( u_mes, '.-')
plt.vlines( u_frame,0,70,colors='red')
# plt.vlines( u_frame+N,0,70,colors='magenta')
plt.vlines( u_cuts,0,70,colors='k',alpha=0.5)
# plt.scatter( u_frame, [50]*len(u_frame), c=u_vid )
plt.show()
plt.figure()
plt.plot( d_mes, '.-')
plt.vlines( d_frame,0,70,colors='red')
# plt.vlines( u_frame+N,0,70,colors='magenta')
plt.vlines( d_cuts,0,70,colors='k',alpha=0.5)
# plt.scatter( d_frame, [50]*len(d_frame), c=d_vid )
plt.show()

#%%

t1 = time()

ny,nx, _ = np.shape(dvids[0].read()[1])
ice_x, ice_y, ice_z = np.zeros((len(d_frame),ny*2*N)), np.zeros((len(d_frame),ny*2*N)), np.zeros((len(d_frame),ny*2*N)) 

for i in tqdm(range(len(d_frame))):
# for i in tqdm(range(1)):
    
    vid = dvids[d_vid[i]]
    ini = d_frame[i] - d_cuts[d_vid[i]]
    vid.set(cv2.CAP_PROP_POS_FRAMES, ini)

    for j in range(N):
        
        im = np.array( vid.read()[1] )[:,1100:2400]
        im = grayscale_im(im)
        im[2090:] = 0
        ili, smi, sma = laser_edges(im, sigma=10)
        smi, sma = smi+1100, sma+1100 
        fma = fit_wall_pixels(ili, sma)
    
        # xc, yc, zc, xi, yi = ice_boundary(ili, smi, fma, wall_distance, cal_do, method='lm')
        xc, yc, zc, xi, yi = ice_boundary(ili, smi, sma, wall_distance, cal_do, method='lm')
        zi = zc
        
        dxi, dyi, dzi = np.copy(xi), np.copy(yi), np.copy(zi)
        
        # fil2_d = np.ones_like(xi, dtype=bool) * np.nanstd(dxi) > 22
        fil2_d = np.nanstd(dxi) > 23
        if fil2_d:
            dxi, dyi, dzi = np.nan, np.nan, np.nan    
        else:
            fil3_d = np.abs(dxi - np.nanmedian(dxi)) > 25
            dxi[fil3_d], dyi[fil3_d], dzi[fil3_d] = np.nan, np.nan, np.nan
        
        ice_x[i][2*j*ny:(2*j+1)*ny] = dxi
        ice_y[i][2*j*ny:(2*j+1)*ny] = dyi
        ice_z[i][2*j*ny:(2*j+1)*ny] = dzi
        
    vid = uvids[u_vid[i]]
    ini = u_frame[i] - u_cuts[u_vid[i]]
    vid.set(cv2.CAP_PROP_POS_FRAMES, ini)

    for j in range(N):
        
        im = np.array( vid.read()[1] )[:,1400:2600]
        im = grayscale_im(im)
        ili, smi, sma = laser_edges(im, sigma=10)
        smi, sma = smi+1400, sma+1400 
        fma = fit_wall_pixels(ili, sma)

        # xc, yc, zc, xi, yi = ice_boundary(ili, smi, fma, wall_distance, cal_up, method='lm')
        xc, yc, zc, xi, yi = ice_boundary(ili, smi, sma, wall_distance, cal_up, method='lm')
        zi = zc
        
        uxi, uyi, uzi = np.copy(xi), np.copy(yi), np.copy(zi)
        # uxi[:130] = np.nan
        
        uplim = np.median(xi - xc) + 10 #110
        ifil = np.where( ((xi - xc) > uplim) * (ili<650) )[0]
        fil1 = np.zeros_like(xi, dtype=bool)
        if len(ifil)>0:
            fil1[:ifil[-1]+1] = True        
            uxi[fil1], uyi[fil1], uzi[fil1] = np.nan, np.nan, np.nan
        
        # fil2_u = np.ones_like(xi, dtype=bool) * np.nanstd(uxi) > 22
        fil2_u = np.nanstd(uxi) > 23
        if fil2_u:
            uxi, uyi, uzi = np.nan, np.nan, np.nan    
        else:
            fil3_u = np.abs(uxi - np.nanmedian(uxi)) > 25
            uxi[fil3_u], uyi[fil3_u], uzi[fil3_u] = np.nan, np.nan, np.nan

        ice_x[i][(2*j+1)*ny:(2*j+2)*ny] = uxi
        ice_y[i][(2*j+1)*ny:(2*j+2)*ny] = uyi
        ice_z[i][(2*j+1)*ny:(2*j+2)*ny] = uzi
    
# u_vid[i], u_vids[u_vid[i]].get_data(0)

np.save(path+'ice_x_0.npy', ice_x)
np.save(path+'ice_y_0.npy', ice_y)
np.save(path+'ice_z_0.npy', ice_z)

t2 = time()
print(t2-t1)
#%%

ice_x = np.load(path+'ice_x_0.npy')
ice_y = np.load(path+'ice_y_0.npy')
ice_z = np.load(path+'ice_z_0.npy')


#%%
# ax = plt.figure().add_subplot(projection='3d')

# i = 15
# ax.plot( ice_z[i], ice_x[i], ice_y[i], 'b.', label=i, markersize=1, alpha=0.5 )
# i = 63
# ax.plot( ice_z[i], ice_x[i], ice_y[i], 'r.', label=i, markersize=1, alpha=0.5 )
 
# ax.set_xlabel('z (mm)')
# ax.set_ylabel('x (mm)')
# ax.set_zlabel('y (mm)')
# ax.set_box_aspect([1,1,1])
# # plt.legend()
# plt.show()

# i = 0
# plt.figure()
# for j in range(18,20):
#     plt.plot( ice_x[i][j*ny*2:(j+1)*ny*2], ice_y[i][j*ny*2:(j+1)*ny*2], '.', label=j )
# plt.legend()
# # plt.axis('equal')
# plt.grid()
# plt.show()

i = 60
plt.figure( figsize=(5,10))
ss=  plt.scatter( ice_z[i], ice_y[i], c=ice_x[i], s=1 )
plt.axis('equal')
cbar = plt.colorbar(ss, location='top')
# plt.xlim(-50,150)
plt.xlabel(r'$x$ (mm)')
plt.ylabel(r'$y$ (mm)')
plt.title(r'$h$ (mm)',fontsize=12, pad=50)
plt.savefig('./Documents/prof4.png',dpi=200, bbox_inches='tight')
plt.show()


#%%

# i = 15000 #430
# l = 0

# t0=time()
# dvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)
# imo = np.array( dvids[l].read()[1] )[:,:,::-1]
# t1 = time()
# # im = grayscale_im(imo[:2090,1100:2400])
# im = grayscale_im(imo[:,1100:2400])
# im[2090:] = 0
# ili, smi, sma = laser_edges(im, sigma=10)
# smi,sma = smi+1100, sma+1100
# fma = fit_wall_pixels(ili, sma)
# xc, yc, zc, xi, yi = ice_boundary(ili, smi, fma, wall_distance, cal_do, method='lm')
# zi = zc
# t2 = time()

# t3 = time()
# xii = np.copy(xi)

# fil2 = np.nanstd(xii) > 22
# if fil2:
#     xii[:] = np.nan
# else:
#     fil3 = np.abs(xii - np.nanmean(xii)) > 25 #np.nanstd(xii) * 3.
#     xii[fil3] = np.nan
# t4 = time()

# print(t2-t1, t4-t3, t1-t0)


# plt.figure()
# # plt.imshow(imo, cmap='gray')
# plt.imshow(im, cmap='gray')
# plt.plot(smi, ili,'b-',alpha=0.5)
# # plt.plot(sma, ili,'r-',alpha=0.5)
# plt.plot(fma, ili,'y--',alpha=0.5)
# plt.show()

i = 0 #430
algo = 10
# l = 3

t0 = time()
# uvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)
# imo = np.array( uvids[l].read()[1] )[:,:,::-1]

vid = uvids[u_vid[i]]
ini = u_frame[i] - u_cuts[u_vid[i]]
vid.set(cv2.CAP_PROP_POS_FRAMES, ini + algo)
imo = np.array( vid.read()[1] )
# im = grayscale_im(im)

t1 = time()

im = grayscale_im(imo[:,1400:2600])
ili, smi, sma = laser_edges(im, sigma=10)
smi,sma = smi+1400, sma+1400
fma = fit_wall_pixels(ili, sma)
xc, yc, zc, xi, yi = ice_boundary(ili, smi, fma, wall_distance, cal_do, method='lm')
zi = zc
t2 = time()

t3 = time()
xii = np.copy(xi)

uplim = np.median(xi - xc) + 10 #110
ifil = np.where( ((xi - xc) > uplim) * (ili<650) )[0]
fil1 = np.zeros_like(xii, dtype=bool)
if len(ifil)>0:
    fil1[:ifil[-1]+1] = True
    xii[fil1] = np.nan

fil2 = np.nanstd(xii) > 22
print(np.nanstd(xii))
if fil2:
    xii[:] = np.nan
else:
    fil3 = np.abs(xii - np.nanmean(xii)) > 25 #np.nanstd(xii) * 3.
    xii[fil3] = np.nan
t4 = time()

# print(ifil[-1], fil2)
# print(t2-t1, t4-t3, t1-t0)
print( xii )


plt.figure()
plt.imshow(imo[:,:,::-1], cmap='gray')
# plt.imshow(im, cmap='gray')
plt.plot(smi, ili,'b-',alpha=0.5)
plt.plot(sma, ili,'r-',alpha=0.5)
plt.plot(fma, ili,'y--',alpha=0.5)
plt.show()




#%%
i = 10000 #430
l = 0
# uvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)
# imo = np.array( uvids[l].read()[1] )[:,:,::-1]
dvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)
imo = np.array( dvids[l].read()[1] )[:,:,::-1]
t1 = time()
# im = grayscale_im(imo[:,1400:2600])
im = grayscale_im(imo[:2090,1100:2400])

ny,nx = np.shape(im)
g2 = gaussian(im, 10)
dg = normalize( np.gradient(g2,axis=1) )
mdg = np.median(dg)

ili = np.arange(ny)

ima, imi = np.full(len(dg), np.nan, dtype=int), np.full(len(dg), np.nan, dtype=int)
for j in range(len(dg)):
    arr1 = find_peaks( dg[j,:], height= mdg+0.1, prominence=0.1)[0]
    arr2 = find_peaks(-dg[j,:], height=-mdg+0.1, prominence=0.1)[0]    
    ima[j] = np.concatenate( (arr1[:1],[np.argmax(dg[j,:])]) )[0]
    imi[j] = np.concatenate( (arr2[:1],[np.argmin(dg[j,:])]) )[0]
smi, sma = subpixel(normalize(dg), imi, ili), subpixel(normalize(dg), ima, ili)

plt.figure()
plt.imshow(dg)
# plt.imshow(im)
plt.plot(sma, ili, '--')
plt.plot(smi, ili, '-')
plt.show()

#%%


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

def circle_filter(points, im):
    fun = lambda vals: circle_fit(points, vals)
    ll = least_squares(fun, [500,500,500])
    
    yind,xind = np.indices(np.shape(im))
    rind = np.sqrt( (xind-ll.x[0])**2 + (yind-ll.x[1])**2 ) #
    # oind = np.arctan2( yind-ll.x[1], xind-ll.x[0] )
    fil = (rind>ll.x[2]-25)*(rind<ll.x[2]+25)

    return fil, ll.x

def angle_dot(im, fil, fit_val, threshold=0.3, sizes=[700,3000] ):

    cx,cy,r = fit_val
    imt = np.copy(im)
    imt[fil] = 0
    
    la = label(imt>threshold)

    # props = regionprops_table( la, properties=('area', 'centroid') )
    # size = props['area']
    # cent = np.column_stack( (props['centroid-0'], props['centroid-1']) )
    # find = (size>sizes[0])&(size<sizes[1])
    # dotc = np.concatenate( (cent[find].flatten() , [np.nan,np.nan]) )[:2]
    # ang = np.arctan2( dotc[0]-cy, dotc[1]-cx  )
    
    dotc = np.array([np.nan, np.nan])
    for prop in regionprops(la):
        if sizes[0] < prop.area < sizes[1]:
            dotc = np.array(prop.centroid)
            break

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

# at 25 fps

path = '/Volumes/Ice blocks/Scan water channel/25-09-29/'

bvids = [cv2.VideoCapture( path + 'Camera back/DSC_0449.MOV'), #starts at 3502
         cv2.VideoCapture( path + 'Camera back/DSC_0450.MOV')] #end at 27627

blens = [0]+[int(bvids[i].get(cv2.CAP_PROP_FRAME_COUNT)) for i in range(len(bvids))]
#%%
l=1
# for i in range(27625,27630,1):
for i in [1000,6000,14000,22000]:
    bvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)
    im = np.array(bvids[l].read()[1])[:,:,::-1] #[:1900,1050:2900,::-1]
    
    plt.figure()
    plt.imshow(im)
    plt.title(i)
    plt.show()    
#%%
l = 0

pos = []
for i in [7000,13000,19000,22000,28000,37000,40000]:
    bvids[l].set(cv2.CAP_PROP_POS_FRAMES, i)
    im = np.array(bvids[l].read()[1])[:1900,1050:2900,::-1]
    im = normalize( gaussian( im[:,:,2], 5) )

    pos.append(pick_points_on_figure(im, i))
    
pos = (np.array(pos))[:,:,0]
fil, fit_val = circle_filter(pos, im)

fit_val
#%%

starts = np.array([3502, 0])
ends = np.array([blens[1]-1, 27627])
stops = [[7533,14308,21987,30126,37849,np.inf],[1895,8760,17771,np.inf]]
restarts = [[7895,14505,22228,30356,38075,np.inf],[2097,8895,17921,np.inf]]

cuts = np.cumsum(ends-starts)
bvideo = np.searchsorted(cuts, np.arange(cuts[-1]), side='right')

nfil = ~fil

t1 = time()

tot_angles = []

for l in range(len(bvids)):
    start,end = starts[l], ends[l]
    N_total = end-start
    angles = np.zeros(N_total)
    bvids[l].set(cv2.CAP_PROP_POS_FRAMES, start )
    for i in tqdm(range(N_total)):
        counter = np.searchsorted(restarts[l], i)
        
        if stops[l][counter] < i+start < restarts[l][counter]:
            bvids[l].read()
            angles[i] = np.nan
    
        else:        
            im = np.array(bvids[l].read()[1])[:1900,1050:2900,::-1]
            im = im[:,:,2]    
            angles[i] = angle_dot(im, nfil, fit_val, threshold=0.3*255, sizes=[800,3000])
    tot_angles.append( angles )

tot_angles = np.hstack(tot_angles)
cuts = np.cumsum(ends-starts)
bvideo = np.searchsorted(cuts, np.arange(68576), side='right')

t2 = time()
print(t2-t1)

np.save(path+'/angles(not_added).npy', tot_angles)

#%%
# from scipy.stats import circmean
l = 0
rotated = True

tot_angles = np.load(path+'/angles(not_added).npy')
tot_stops, tot_rest = np.array( stops[0][:-1] ), np.array( restarts[0][:-1] )
for l in range(1,len(bvids)):
    tot_stops = np.concatenate( (tot_stops,stops[l][:-1] + cuts[l-1]) )
    tot_rest = np.concatenate( (tot_rest,restarts[l][:-1] + cuts[l-1]) )
tot_stops, tot_rest = tot_stops - starts[np.searchsorted(cuts, tot_stops)], tot_rest - starts[np.searchsorted(cuts, tot_rest)] 


angt = tot_angles - 1*(np.sign(tot_angles)-1)*np.pi
angmed = get_median_angles(angt, tot_stops, tot_rest)

tot_nrots = [0,4,3,4,4,4,5,3,2]
ext_rot = np.diff(angmed) - (np.sign(np.diff(angmed))-1)*np.pi
rotation = np.cumsum(tot_nrots)*2*np.pi + np.cumsum( np.hstack((0,ext_rot)) )

frm = [0]
if rotated:
    ags = [rotation[0]]
    for i in range(len(rotation)-1):
        frm.append( tot_stops[i] )
        frm.append( tot_rest[i] )
        ags.append( rotation[i] )
        ags.append( rotation[i+1] )
    ags.append( rotation[-1] )
elif not rotated:
    ags = [angmed[0]]
    for i in range(len(rotation)-1):
        frm.append( tot_stops[i] )
        frm.append( tot_rest[i] )
        ags.append( angmed[i] )
        ags.append( angmed[i+1] )
    ags.append( angmed[-1] )
frm.append( len(tot_angles) )


frm, ags = np.array(frm), np.array(ags)

plt.figure()

plt.plot( angt, '.')
plt.plot( frm, ags, '.-' )
# plt.hlines( angmed , 0, 70000, linestyles='dashed', colors=['r','b','g','m','y'])

plt.grid()
plt.show()    

#%%
time = frm/25
time[11:] += 3

bposition = ags * 2.5 / (2*np.pi)

plt.figure()
plt.plot(time/60, bposition, '.-')
plt.grid()
plt.show()

bdat = [time, bposition]
np.savez(path+'back_plate_position.npz', *bdat)

#%%

bplate = np.load(path+'back_plate_position.npz')

time = bplate['arr_0']
bpos = bplate['arr_1']

plt.figure()
plt.plot(time/60, bpos, '.-')
plt.grid()
plt.show()



#%%


import datetime

timeList = ['0:00:00', '0:29:38', '15:39:05']
mysum = datetime.timedelta()
for i in timeList:
    (h, m, s) = i.split(':')
    d = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))
    mysum += d
print(str(mysum))



#%%




#%%
starts = [3502, 0]
ends = [blens[1], 27627]

stops = [[7533,14308,21987,30126,37849,np.inf],[1895,8760,17771,np.inf]]
restarts = [[7895,14505,22228,30356,38075,np.inf],[2097,8895,17921,np.inf]]

nfil = ~fil


l = 1


N_total = 27627
angles = np.zeros(N_total)

t1 = time()
start = 0
bvids[l].set(cv2.CAP_PROP_POS_FRAMES, start )
for i in tqdm(range(N_total)):
    counter = np.searchsorted(restarts[l], i)
    
    if stops[l][counter] < i < restarts[l][counter]:
        bvids[l].read()
        angles[i] = np.nan

    else:        
        im = np.array(bvids[l].read()[1])[:1900,1050:2900,::-1]
        im = im[:,:,2]    
        angles[i] = angle_dot(im, nfil, fit_val, threshold=0.3*255, sizes=[900,3000])

t2 = time()
t2-t1


#%%
l = 1
start = 0
i = -start+ 27550
bvids[l].set(cv2.CAP_PROP_POS_FRAMES, start+i )

im = np.array(bvids[l].read()[1])[:1900,1050:2900,::-1]
im = im[:,:,2]

ang1 = angle_dot(im, nfil, fit_val, threshold=0.3*255, sizes=[800,3000])

sizes = [800,3000]
threshold=0.3*255
cx,cy,r = fit_val
imt = np.copy(im)
imt[nfil] = 0

la = label(imt>threshold)

print( np.array( [int(prop.area) for prop in regionprops(la)] ) )

dotc = np.array([np.nan, np.nan])
for prop in regionprops(la):
    if sizes[0] < prop.area < sizes[1]:
        dotc = np.array(prop.centroid)
        break

ang = np.arctan2( -(dotc[0]-cy), dotc[1]-cx  )

print(ang *180/np.pi, ang1*180/np.pi, angles[i]*180/np.pi)


# plt.figure()
# plt.imshow( la )
# plt.plot( dotc[1], dotc[0], 'r.' )
# plt.show()

plt.figure()
plt.imshow( normalize(imt) )
plt.plot( dotc[1], dotc[0], 'r.' )
plt.show()
plt.figure()
plt.imshow( normalize(im) )
plt.show()


#%%



#%%








#%%
# =============================================================================
# Back iamge
# =============================================================================
# at 25 fps

path = '/Volumes/Ice blocks/Scan water channel/25-08-07/'
bvid1 = imageio.get_reader( path + 'Camera back/DSC_0436.MOV', 'ffmpeg') # 7272 frames, starts 3487 (for seeing 4480), 15:31:06
bvid2 = imageio.get_reader( path + 'Camera back/DSC_0437.MOV', 'ffmpeg') # 6624 frames                               , 15:35:56
bvid3 = imageio.get_reader( path + 'Camera back/DSC_0438.MOV', 'ffmpeg') # 6684 frames                               , 15:40:21
bvid4 = imageio.get_reader( path + 'Camera back/DSC_0439.MOV', 'ffmpeg') # 6876 frames                               , 15:44:49
bvid5 = imageio.get_reader( path + 'Camera back/DSC_0440.MOV', 'ffmpeg') # 6852 frames                               , 15:49:24
bvid6 = imageio.get_reader( path + 'Camera back/DSC_0441.MOV', 'ffmpeg') # 6648 frames                               , 15:53:58
bvid7 = imageio.get_reader( path + 'Camera back/DSC_0442.MOV', 'ffmpeg') # 4020 frames                               , 15:58:24
bvid8 = imageio.get_reader( path + 'Camera back/DSC_0443.MOV', 'ffmpeg') # 6924 frames, ends 5658 (for seeing 5424)  , 16:01:57


# print(bvid1.count_frames(), bvid2.count_frames(), bvid3.count_frames(), bvid4.count_frames()) 
# print(bvid5.count_frames(), bvid6.count_frames(), bvid7.count_frames(), bvid8.count_frames()) 

times = ['15:31:06', '15:35:56', '15:40:21', '15:44:49', '15:49:24', '15:53:58', '15:58:24', '16:01:57']    
# pol = np.array([7270,6622,6682,6874,6850,6646,4018,6922])
b_frames = initial_frames(times, fps=25)
b_frames
#%%

for i in range(5656,5659,1):
    im = np.array( bvid8.get_data(i) )
    # im = grayscale_im(im)
    plt.figure()
    plt.imshow(im)
    plt.title(i)
    plt.show()

#%%

def nan_argmax(peks, pprop):
    try:
        return peks[np.argmax(pprop['prominences'])]
    except ValueError:
        return np.nan


pes, aas, ies = [],[],[]

for i in tqdm(range(4480, 7271, 10)):
    im = np.array( bvid1.get_data(i) )
    im = grayscale_im(im)
    img = gaussian(im,15)
    aa = np.mean( img[1100:1300,700:3500],axis=0 )
    peks,pprop = find_peaks(-aa, prominence=30, wlen=120)
    pep = nan_argmax(peks, pprop)

    pes.append(pep)
    aas.append(aa)
    ies.append(i+b_frames[0])

for i in tqdm(range(0, 6623, 10)):
    im = np.array( bvid2.get_data(i) )
    im = grayscale_im(im)
    img = gaussian(im,15)
    aa = np.mean( img[1100:1300,700:3500],axis=0 )
    peks,pprop = find_peaks(-aa, prominence=30, wlen=120)
    pep = nan_argmax(peks, pprop)

    pes.append(pep)
    aas.append(aa)
    ies.append(i+b_frames[1])
    
for i in tqdm(range(0, 6683, 10)):
    im = np.array( bvid3.get_data(i) )
    im = grayscale_im(im)
    img = gaussian(im,15)
    aa = np.mean( img[1100:1300,700:3500],axis=0 )
    peks,pprop = find_peaks(-aa, prominence=30, wlen=120)
    pep = nan_argmax(peks, pprop)

    pes.append(pep)
    aas.append(aa)
    ies.append(i+b_frames[2])
    
for i in tqdm(range(0, 6875, 10)):
    im = np.array( bvid4.get_data(i) )
    im = grayscale_im(im)
    img = gaussian(im,15)
    aa = np.mean( img[1100:1300,700:3500],axis=0 )
    peks,pprop = find_peaks(-aa, prominence=30, wlen=120)
    pep = nan_argmax(peks, pprop)

    pes.append(pep)
    aas.append(aa)
    ies.append(i+b_frames[3])
    
for i in tqdm(range(0, 6851, 10)):
    im = np.array( bvid5.get_data(i) )
    im = grayscale_im(im)
    img = gaussian(im,15)
    aa = np.mean( img[1100:1300,700:3500],axis=0 )
    peks,pprop = find_peaks(-aa, prominence=30, wlen=120)
    pep = nan_argmax(peks, pprop)

    pes.append(pep)
    aas.append(aa)
    ies.append(i+b_frames[4])
    
for i in tqdm(range(0, 6647, 10)):
    im = np.array( bvid6.get_data(i) )
    im = grayscale_im(im)
    img = gaussian(im,15)
    aa = np.mean( img[1100:1300,700:3500],axis=0 )
    peks,pprop = find_peaks(-aa, prominence=30, wlen=120)
    pep = nan_argmax(peks, pprop)

    pes.append(pep)
    aas.append(aa)
    ies.append(i+b_frames[5])
    
for i in tqdm(range(0, 4019, 10)):
    im = np.array( bvid7.get_data(i) )
    im = grayscale_im(im)
    img = gaussian(im,15)
    aa = np.mean( img[1100:1300,700:3500],axis=0 )
    peks,pprop = find_peaks(-aa, prominence=30, wlen=120)
    pep = nan_argmax(peks, pprop)

    pes.append(pep)
    aas.append(aa)
    ies.append(i+b_frames[6])
    
for i in tqdm(range(0, 5423, 10)):
    im = np.array( bvid8.get_data(i) )
    im = grayscale_im(im)
    img = gaussian(im,15)
    aa = np.mean( img[1100:1300,700:3500],axis=0 )
    peks,pprop = find_peaks(-aa, prominence=30, wlen=120)
    pep = nan_argmax(peks, pprop)

    pes.append(pep)
    aas.append(aa)
    ies.append(i+b_frames[7])
    
pes = np.array(pes)
ies = np.array(ies)
#%%

start = 3158 + 2
interval = 30

d_times = ['16:32:30','16:41:30','16:51:14','17:00:04','17:02:56']    
d_len_vid = [16155, 17490, 15900, 4374, 6259]
d_frame, d_vid, d_fpos = frames_reconstruction(start, interval, d_times, d_len_vid)

u_times = ['17:32:00','17:43:12','17:54:14','18:02:30']    
u_len_vid = [20100, 19860, 13959, 6140]
u_frame, u_vid, u_fpos = frames_reconstruction(start+40, interval, u_times, u_len_vid)


u_frame, u_vid, u_len_vid

#%%

cosos = [2308,
         2168,
         2171,
         1880,
         1880,
         1880,
         1880,
         1679,
         1679,
         1679,
         1679,
         1529,
         1529,
         1529,
         1529,
         1529,
         1529,
         1422,
         1422,
         1422,
         1240,
         1240,
         1240,
         1065,
         1065,
         1065,
         1065,
         1065,
         784,
         784,
         784,
         784,
         784,
         784,
         784,
         784,
         665,
         665,
         665,
         665,
         665,
         665,
         594,
         484,
         484,
         484,
         484,
         484,
         484,
         369,
         369,
         369,
         369,
         369,
         172,
         172,
         172,
         172,
         172,
         172,
         128,
         128,
         128,
         128,
         128]

plt.figure()
# for i in range(10):
#     plt.plot(aas[i],'-',zorder=-1)
#     plt.plot(pes[i],aas[i][pes[i]],'k.',zorder=0)

plt.plot((ies-4480)/25, pes,'.-')
# plt.plot( (d_fpos-start)/30, [100]*len(d_fpos), '.' )
# plt.plot( (u_fpos-start-40)/30, [100]*len(u_fpos), '.' )
plt.plot( (u_fpos-start-40)/30, cosos, '.' )

plt.show()

#%%

# cal frame
path = '/Volumes/Ice blocks/Scan water channel/25-08-07/'
bcal = imageio.get_reader( path + 'Camera back/DSC_0435.MOV', 'ffmpeg') # 7272 frames, starts 3487 (for seeing 4480), 15:31:06

im = np.array( bcal.get_data(225) )
im = grayscale_im(im[700:1300,700:3500])
# im = grayscale_im(im)

plt.figure()
plt.imshow(im, cmap='gray')
plt.show()

#%%
from scipy.stats import linregress

posit = [[150,3350],
         [140,3165],
         [130,2984],
         [120,2805],
         [110,2628],
         [100,2454],
         [ 90,2282],
         [ 80,2112],
         [ 70,1943],
         [ 60,1778],
         [ 50,1616],
         [ 40,1455],
         [ 30,1296],
         [ 20,1138],
         [ 10,986]]
posit = np.array(posit)
posit[:,1] = posit[:,1]-700 

a = linregress(posit[:,1], posit[:,0])
b = np.polyfit(posit[:,1], posit[:,0], 2)
ee = np.linspace(100,2800,20)  

plt.figure()
# plt.imshow(im, cmap='gray')
# plt.plot( posit[:,1], [250]*15, '.' )

plt.plot( posit[:,1], posit[:,0], '.' )
plt.plot( ee, ee*a.slope+a.intercept,'--' )
plt.plot( ee, ee**2 * b[0]+ee*b[1]+b[2],'--' )

plt.show()
#%%

cosos = np.array(cosos)
disp = cosos**2 *b[0] + cosos * b[1] + b[2]
disp = disp[0] - disp

print(disp)

np.save(path+'back_dispalcement.npy',disp)
#%%
# =============================================================================
# aaaa
# =============================================================================
path = '/Volumes/Ice blocks/Scan water channel/25-08-07/'
ice_x = np.load(path+'ice_x.npy')
ice_y = np.load(path+'ice_y.npy')
ice_z = np.load(path+'ice_z.npy')

#%%
i = 19
plt.figure()
plt.scatter(ice_z[i], ice_y[i], c=ice_x[i], s=1, cmap='jet')
plt.axis('equal')
plt.show()

#%%

mes, sd = [],[]
for i in range(len(ice_x)):
    mes.append( np.nanmean(ice_x[i]) )
    sd.append( np.nanstd(ice_x[i]) )
#%%
fil = np.isnan(mes)
ss = linregress( np.arange(65)[~fil], (mes + disp)[~fil] )
dd = np.arange(65)

plt.figure()
plt.plot( mes, '.-' )
plt.plot( mes + disp, '.-' )
plt.plot( dd, dd*ss.slope+ss.intercept,'--' )
# plt.plot( sd, '.-')
plt.grid()
plt.show()




#%%


















#%%


















#%%















