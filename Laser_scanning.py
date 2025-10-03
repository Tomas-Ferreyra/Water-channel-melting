#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 14:46:17 2025

@author: tomasferreyrahauchar
"""

import imageio.v2 as imageio
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
from skimage.measure import label, regionprops
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

    ili = np.arange(ny)
    imi = np.argmin(dg, axis=1  ) 
    ima = np.argmax(dg, axis=1  ) 
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


#%%
# 30 fps

path = '/Volumes/Ice blocks/Scan water channel/25-08-07/'

data = np.load(path+'calibration_data.npz')
angle_xy, angle_yz, angle_xz = float(data['arr_3']), float(data['arr_4']), float(data['arr_5'])

cal_up = data['arr_0']
cal_do = data['arr_1']
wall_distance = float(data['arr_2'])


dvid1 = imageio.get_reader( path + 'Camera down/DSC_9525.MOV', 'ffmpeg') # 16155 frames, starts 3158, 16:32:30
dvid2 = imageio.get_reader( path + 'Camera down/DSC_9526.MOV', 'ffmpeg') # 17490 frames,              16:41:30
dvid3 = imageio.get_reader( path + 'Camera down/DSC_9527.MOV', 'ffmpeg') # 15900 frames,              16:51:14 
dvid4 = imageio.get_reader( path + 'Camera down/DSC_9528.MOV', 'ffmpeg') # 4374 frames,               17:00:04
dvid5 = imageio.get_reader( path + 'Camera down/DSC_9529.MOV', 'ffmpeg') # 18075 frames, ends 6259,   17:02:56

uvid1 = imageio.get_reader( path + 'Camera up/DSC_6011.MOV', 'ffmpeg') # 20100 frames, starts 3198, 17:32:00
uvid2 = imageio.get_reader( path + 'Camera up/DSC_6012.MOV', 'ffmpeg') # 19860 frames,              17:43:12
uvid3 = imageio.get_reader( path + 'Camera up/DSC_6013.MOV', 'ffmpeg') # 13959 frames,              17:54:14
uvid4 = imageio.get_reader( path + 'Camera up/DSC_6014.MOV', 'ffmpeg') # 18270 frames, ends 6140,   18:02:30

# dvid1.count_frames(), dvid2.count_frames(), dvid3.count_frames(), dvid4.count_frames(), dvid5.count_frames()
# uvid1.count_frames(), uvid2.count_frames(), uvid3.count_frames(), uvid4.count_frames()

#uvid1 -> cut at 16692, comes back 18162 (49 seconds of darkness)
#dvid2 -> cut at 497, comes back 1967 (49 seconds of darkness)
#fps 30, (supposedly 29.97)
#%%
for i in range(6258,6259,1):
    im = np.array( dvid1.get_data(i) )
    # im = grayscale_im(im)
    plt.figure()
    plt.imshow(im) 
    plt.title(i)
    plt.show()

#%%

start = 3158 + 2
interval = 30

d_times = ['16:32:30','16:41:30','16:51:14','17:00:04','17:02:56']    
d_len_vid = [16155, 17490, 15900, 4374, 6259]
d_frame, d_vid, _ = frames_reconstruction(start, interval, d_times, d_len_vid)

u_times = ['17:32:00','17:43:12','17:54:14','18:02:30']    
u_len_vid = [20100, 19860, 13959, 6140]
u_frame, u_vid, _ = frames_reconstruction(start+40, interval, u_times, u_len_vid)

alignment = np.zeros_like(u_vid)
alignment[15:19], alignment[19:34], alignment[34:41] = -45, 15, -15 
alignment[41:52], alignment[52:57], alignment[57:] = -15, -15, -39

u_frame = u_frame + alignment
#%%

t1 = time()

wall_distance = -5.56566902681778
N = 60


d_vids = {0:dvid1, 1:dvid2, 2:dvid3, 3:dvid4, 4:dvid5}
u_vids = {0:uvid1, 1:uvid2, 2:uvid3, 3:uvid4}

ny,nx, _ = np.shape(dvid1.get_data(0))
ice_x, ice_y, ice_z = np.zeros((len(d_frame),ny*2*N)), np.zeros((len(d_frame),ny*2*N)), np.zeros((len(d_frame),ny*2*N)) 

for i in tqdm(range(len(d_frame))):
# for i in tqdm(range(4)):
# for i in tqdm([14,15,16,17]):
    
    # problem with 57 -> is when recordings cut off becuase of the 30 min
    # problems with 15,16 -> they are when laser was off
    if (i == 57) or (i == 16):
        ice_x[i] = np.nan
        ice_y[i] = np.nan
        ice_z[i] = np.nan
        continue
    
    ini = d_frame[i]

    for j,l in enumerate(range(ini, ini+N)):
    
        im = np.array( d_vids[d_vid[i]].get_data(l) )[:,1250:2300]
        im = grayscale_im(im)
        ili, smi, sma = laser_edges(im, sigma=10)
        smi, sma = smi+1250, sma+1250 
        fma = fit_wall_pixels(ili, sma)
    
        xc, yc, zc, xi, yi = ice_boundary(ili, smi, fma, wall_distance, cal_do, method='lm')
        zi = zc
        
        dxi, dyi, dzi = np.copy(xi), np.copy(yi), np.copy(zi)
        
        # fil2_d = np.ones_like(xi, dtype=bool) * np.nanstd(dxi) > 22
        fil2_d = np.nanstd(dxi) > 22
        if fil2_d:
            dxi, dyi, dzi = np.nan, np.nan, np.nan    
        else:
            fil3_d = np.abs(dxi - np.nanmedian(dxi)) > 25
            dxi[fil3_d], dyi[fil3_d], dzi[fil3_d] = np.nan, np.nan, np.nan
        
        ice_x[i][2*j*ny:(2*j+1)*ny] = dxi
        ice_y[i][2*j*ny:(2*j+1)*ny] = dyi
        ice_z[i][2*j*ny:(2*j+1)*ny] = dzi
        
        
    ini = u_frame[i]

    for j,l in enumerate(range(ini, ini+N)):
        
        im = np.array( u_vids[u_vid[i]].get_data(l) )[:,1250:2330]
        im = grayscale_im(im)
        ili, smi, sma = laser_edges(im, sigma=10)
        smi, sma = smi+1250, sma+1250 
        fma = fit_wall_pixels(ili, sma)

        xc, yc, zc, xi, yi = ice_boundary(ili, smi, fma, wall_distance, cal_up, method='lm')
        zi = zc
        
        uxi, uyi, uzi = np.copy(xi), np.copy(yi), np.copy(zi)
        uxi[:130] = np.nan
        
        uplim = np.median(xi - xc) + 10 #110
        ifil = np.where( ((xi - xc) > uplim) * (ili<600) )[0]
        fil1 = np.zeros_like(xi, dtype=bool)
        if len(ifil)>0:
            fil1[:ifil[-1]+1] = True        
            uxi[fil1], uyi[fil1], uzi[fil1] = np.nan, np.nan, np.nan
        
        # fil2_u = np.ones_like(xi, dtype=bool) * np.nanstd(uxi) > 22
        fil2_u = np.nanstd(uxi) > 22
        if fil2_u:
            uxi, uyi, uzi = np.nan, np.nan, np.nan    
        else:
            fil3_u = np.abs(uxi - np.nanmedian(uxi)) > 25
            uxi[fil3_u], uyi[fil3_u], uzi[fil3_u] = np.nan, np.nan, np.nan

        ice_x[i][(2*j+1)*ny:(2*j+2)*ny] = uxi
        ice_y[i][(2*j+1)*ny:(2*j+2)*ny] = uyi
        ice_z[i][(2*j+1)*ny:(2*j+2)*ny] = uzi
    
# u_vid[i], u_vids[u_vid[i]].get_data(0)

np.save('/Volumes/Ice blocks/Scan water channel/25-08-07/ice_x.npy', ice_x)
np.save('/Volumes/Ice blocks/Scan water channel/25-08-07/ice_y.npy', ice_y)
np.save('/Volumes/Ice blocks/Scan water channel/25-08-07/ice_z.npy', ice_z)

t2 = time()
print(t2-t1)
#%%

ice_x = np.load('/Volumes/Ice blocks/Scan water channel/25-08-07/ice_x.npy')
ice_y = np.load('/Volumes/Ice blocks/Scan water channel/25-08-07/ice_y.npy')
ice_z = np.load('/Volumes/Ice blocks/Scan water channel/25-08-07/ice_z.npy')


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

plt.figure()
ss=  plt.scatter( ice_z[0], ice_y[0], c=ice_x[0], s=1 )
plt.axis('equal')
plt.colorbar(ss)
plt.show()


#%%

plt.figure()

i = 3280 #430
imo = np.array( dvid5.get_data(i) )
t1 = time()
im = grayscale_im(imo[:,1250:2300])
ili, smi, sma = laser_edges(im, sigma=10)
smi,sma = smi+1250, sma+1250
fma = fit_wall_pixels(ili, sma)
xc, yc, zc, xi, yi = ice_boundary(ili, smi, fma, -5.56566902681778, cal_do, method='lm')
zi = zc
t2 = time()

t3 = time()
xii = np.copy(xi)

fil2 = np.nanstd(xii) > 22
if fil2:
    xii[:] = np.nan
else:
    fil3 = np.abs(xii - np.nanmean(xii)) > 25 #np.nanstd(xii) * 3.
    xii[fil3] = np.nan
t4 = time()

print(t2-t1, t4-t3)

plt.plot(zc, yc, '.-')


# plt.figure()
# plt.imshow(imo, cmap='gray')
# plt.plot(smi, ili,'b-',alpha=0.5)
# # plt.plot(sma, ili,'r-',alpha=0.5)
# plt.plot(fma, ili,'y--',alpha=0.5)
# plt.show()

# plt.figure()
# plt.plot(xii, yi,'.-', markersize=10)
# plt.plot(xi, yi,'.', markersize=3)
# # plt.axis('equal')
# plt.show()


# from scipy.stats import kurtosis


# for i in 3200 + np.arange(15,45,1):
for i in 3200 + np.array([-39,-4]):
    imo = np.array( uvid4.get_data(i) )
    t1 = time()
    im = grayscale_im(imo[:,1250:2330])
    ili, smi, sma = laser_edges(im, sigma=10)
    smi,sma = smi+1250, sma+1250
    fma = fit_wall_pixels(ili, sma)
    xc, yc, zc, xi, yi = ice_boundary(ili, smi, fma, -5.56566902681778, cal_up, method='lm')
    zi = zc
    t2 = time()
    
    
    t3 = time()
    xii = np.copy(xi)
    xii[:130] = np.nan
    
    uplim = np.median(xi - xc) + 10 #110
    ifil = np.where( ((xi - xc) > uplim) * (ili<600) )[0]
    fil1 = np.zeros_like(xii, dtype=bool)
    if len(ifil)>0:
        fil1[:ifil[-1]+1] = True
        xii[fil1] = np.nan
    
    fil2 = np.nanstd(xii) > 22
    if fil2:
        xii[:] = np.nan
    else:
        fil3 = np.abs(xii - np.nanmedian(xii)) > 25 #np.nanstd(xii) * 3.
        xii[fil3] = np.nan
    t4 = time()
    
    plt.plot(zc, yc, '.--', label=i-3200)
# plt.plot(zi, yi, '.--')

# print(t2-t1, t4-t3)
    
    # plt.figure()
    # plt.imshow(imo) #, cmap='gray')
    # plt.plot(smi, ili,'b-',alpha=0.5)
    # plt.plot(sma, ili,'r-',alpha=0.5)
    # plt.plot(fma, ili,'y--',alpha=0.5)
    # plt.show()

# plt.figure()
# plt.plot(xii, yi,'.-', markersize=10)
# plt.plot(xi, yi,'.', markersize=3)
# # plt.plot(xc, yc,'.', markersize=3)
# plt.show()

plt.legend()
plt.show()


#%%


a = np.array([1,2,3,3])
b = np.array([1,2])

# a[:b[-1:]+1]
np.size(b[-1:]+1)

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















