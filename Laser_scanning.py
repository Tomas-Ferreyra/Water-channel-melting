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
        return eq1**2+eq2**2
    
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
        mini = minimize(funci, x0, method=method)
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
        (a,b,c),cov = curve_fit(cuad, ili[os[j]], sma[os[j]])
        fma[ili[os[j]]] = cuad(ili[os[j]], a,b,c)
    
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

    

#%%
# 30 fps

# cal_do = np.array([ 2.64549425e+00, -3.99891836e-02,  4.61459294e+00,  1.23603665e+02,  9.28098368e-04,  3.38779738e-05, -2.29887133e-05,  3.57383747e-01,
#                    -6.07051697e+00, -3.87935959e-01,  1.87544749e+02,  6.84006030e-06, -2.94275290e-05,  5.62181348e-04,  1.80430389e-04,  7.62471773e-06,
#                    -3.24476172e-04, -5.05999352e-08, -2.19517311e-08, -3.35309119e-08])

# cal_up = np.array([ 2.63988133e+00, -2.58954213e-02,  4.57771951e+00,  1.71317494e+02,  9.11008193e-04,  2.78947580e-05,  5.37167048e-06,  1.70425401e-01,
#                    -6.03239571e+00, -3.27384103e-01,  1.85175687e+03, -1.17408633e-05, -3.03129810e-05,  5.71964284e-04,  1.88649976e-04,  1.18542528e-05,
#                    -3.25451713e-04, -3.18051528e-08, -8.08787246e-09,  3.14982785e-08])


cal_do = np.array([ 2.85929551e+00, -2.15384113e-02,  4.22331053e+00,  1.32362855e+03, 9.57344726e-04,  1.70426598e-05, -4.18446710e-05,  3.70903866e-01,
                    -6.06846060e+00, -3.87914915e-01,  1.87593532e+02,  3.01258117e-06, -2.56008634e-05,  5.60267140e-04,  1.75146199e-04,  9.10059318e-06,
                    -3.25217711e-04,  4.97555994e-08, -1.75545859e-08, -2.54218391e-08])
 
cal_up = np.array([ 2.87019324e+00, -3.28273855e-03,  4.18727807e+00,  1.37131050e+03, 7.09707209e-04,  9.87310806e-06,  4.13837579e-05,  2.10699779e-01,
                    -6.03161482e+00, -3.69764818e-01,  1.98191198e+03, -1.26120666e-04, -2.89736408e-05,  5.93548674e-04,  1.92803621e-04,  1.26672913e-05,
                    -3.25175568e-04, -1.37520884e-07, -1.22594341e-08,  2.60493380e-08])


dvid1 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera down/DSC_9525.MOV', 'ffmpeg') # 16155 frames, starts 3158
dvid2 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera down/DSC_9526.MOV', 'ffmpeg') # 17490 frames
dvid3 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera down/DSC_9527.MOV', 'ffmpeg') # 15900 frames
dvid4 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera down/DSC_9528.MOV', 'ffmpeg') # 4374 frames
dvid5 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera down/DSC_9529.MOV', 'ffmpeg') # 18075 frames, ends 6259

uvid1 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera up/DSC_6011.MOV', 'ffmpeg') # 20100 frames, starts 3198
uvid2 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera up/DSC_6012.MOV', 'ffmpeg') # 19860 frames
uvid3 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera up/DSC_6013.MOV', 'ffmpeg') # 13959 frames
uvid4 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera up/DSC_6014.MOV', 'ffmpeg') # 18270 frames, ends 6140

# dvid1.count_frames(), dvid2.count_frames(), dvid3.count_frames(), dvid4.count_frames(), dvid5.count_frames()
# uvid1.count_frames(), uvid2.count_frames(), uvid3.count_frames(), uvid4.count_frames()

#uvid1 -> cut at 16692, comes back 18162 (49 seconds of darkness)
#uvid1 -> cut at 497, comes back 1967 (49 seconds of darkness)
#%%

for i in range(6258,6259,1):
    im = np.array( dvid1.get_data(i) )
    # im = grayscale_im(im)
    plt.figure()
    plt.imshow(im)
    plt.title(i)
    plt.show()

#%%
t1 = time()

N = 60
ini = 10000

ny,nx, _ = np.shape(dvid1.get_data(0))
dxis, dyis, dzis = np.zeros((N,ny)), np.zeros((N,ny)), np.zeros((N,ny))
dxcs, dycs, dzcs = np.zeros((N,ny)), np.zeros((N,ny)), np.zeros((N,ny))
dsmi, dsma = np.zeros((N,ny)), np.zeros((N,ny))

t2s, t4s = [], []
for j,i in enumerate(tqdm(range(ini, ini+N,1))):
    im = np.array( dvid1.get_data(i) )[:,1250:2300]
    
    t1 = time()
    im = grayscale_im(im)
    ili, smi, sma = laser_edges(im, sigma=10)
    smi, sma = smi+1250, sma+1250 
    fma = fit_wall_pixels(ili, sma)

    dxcs[j], dycs[j], dzcs[j], dxis[j], dyis[j] = ice_boundary(ili, smi, fma, -5.56566902681778, cal_do, method='SLSQP')
    dzis[j] = dzcs[j]
    dsmi[j], dsma[j] = smi,sma

    
uxis, uyis, uzis = np.zeros((N,ny)), np.zeros((N,ny)), np.zeros((N,ny))
uxcs, uycs, uzcs = np.zeros((N,ny)), np.zeros((N,ny)), np.zeros((N,ny))    
usmi, usma = np.zeros((N,ny)), np.zeros((N,ny))

for j,i in enumerate(tqdm(range(ini+40, ini+40+N,1))):
    im = np.array( uvid1.get_data(i) )[:,1250:2330]
    
    t1 = time()
    im = grayscale_im(im)
    ili, smi, sma = laser_edges(im, sigma=10)
    smi, sma = smi+1250, sma+1250 
    fma = fit_wall_pixels(ili, sma)

    uxcs[j], uycs[j], uzcs[j], uxis[j], uyis[j] = ice_boundary(ili, smi, fma, -5.56566902681778, cal_up, method='SLSQP')
    uzis[j] = uzcs[j]
    usmi[j], usma[j] = smi,sma
    

t2 = time()
print(t2-t1)

#%%
# first filter xis - xcs > 110 (if index<600)
# second filter, remove if np.std(np.gradient(xis[i])) > 4 

# i = 209
# fil = np.where( (xis[i] - xcs[i] > 110))[0]
# filfil = fil<600
# ifil = np.max(fil[filfil]) 

# plt.figure()
# # plt.plot(zcs[i], ycs[i], '.-')
# # plt.plot(zis[i], yis[i], '.-')
# plt.plot( xis[i] - xcs[i], '.-')
# plt.plot( (xis[i] - xcs[i])[:ifil], '.-')
# plt.show()

# # print(np.std(np.gradient(xis[i])))
# # print(np.std(np.gradient(yis[i])))
# # print(np.std(np.gradient(zis[i])))
# # print(np.std(np.gradient(xcs[i])))
# # print(np.std(np.gradient(ycs[i])))
# # print(np.std(np.gradient(zcs[i])))

# plt.figure()
# plt.plot( np.std(np.gradient(xis,axis=1),axis=1), '.-' )
# # plt.plot( np.std(np.gradient(yis,axis=1),axis=1), '.-' )
# # plt.plot( np.std(np.gradient(zis,axis=1),axis=1), '.-' )
# # plt.plot( np.std(np.gradient(xis,axis=1),axis=1), '.-' )
# plt.grid()
# plt.show()

ax = plt.figure().add_subplot(projection='3d')
for i in range(60):
    # if i == 29: continue
# for i in [9,19,29]:
    # ax.plot( uzcs[i], uxcs[i], uycs[i], 'r-' )
    ax.plot( uzis[i], uxis[i], uyis[i], '-', label=i )
    
    # ax.plot( dzcs[i], dxcs[i], dycs[i], 'r-' )
    ax.plot( dzis[i], dxis[i], dyis[i], '-', label=i )
    
    # i_filt = np.where((np.gradient(xis[i])>-50) * (np.gradient(xis[i])<-3))[0][-1]
    # filt = ili <= i_filt
    # ax.plot( zis[i][filt], xis[i][filt], yis[i][filt], 'b-', label=i )
    
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.set_box_aspect([1,1,1])
# plt.legend()
plt.show()

# i = 30

# # i_filt = np.where((np.gradient(smi)>-50) * (np.gradient(smi)<-5))[0][-1]
# # # i_filt = np.where((np.gradient(xis[i])>-50) * (np.gradient(xis[i])<-3))[0][-1]
# # filt = ili <= i_filt

# plt.figure()
# # plt.plot( np.gradient(xis[i]), '.-')
# # plt.plot( np.gradient(xis[i])[filt], '.-')
# # plt.plot( (xis[i]), '.-')
# # plt.plot( (xis[i])[filt], '.-')
# plt.plot( zcs[i], ycs[i], '.-')
# plt.xlabel('z (mm)')
# plt.ylabel('y (mm)')
# plt.axis('equal')
# plt.grid()
# plt.show()


#%%


def calib_ls(v, points, coord, x0s=[[50,300]], dis_bar=False, method='trf'):

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

def ice_boundary_ls(ili, smi, sma, wall_distace, calib, dis_bar=True, method='trf'):
    wall_d = np.ones_like(ili) * wall_distace #distance from grid to window in mm (from fisrt point)
    points = np.vstack((sma,ili,wall_d)).T
    # yr, zr = calib_invzl(calib, points, 0, dis_bar=dis_bar, method=method)    
    yr, zr = calib_ls(calib, points, 0, dis_bar=dis_bar, method=method)    
    yc, zc, dwall = yr, zr, wall_d
    
    points = np.vstack((smi,ili,zc)).T
    # xi, yi = calib_invzl(calib, points, 2, dis_bar=dis_bar, method=method)
    xi, yi = calib_ls(calib, points, 2, dis_bar=dis_bar, method=method)
    
    return dwall, yc, zc, xi, yi





#%%
i = 10000 + 36
imo = np.array( dvid1.get_data(i) )
t1 = time()
im = grayscale_im(imo[:,1250:2300])
ili, smi, sma = laser_edges(im, sigma=10)
smi,sma = smi+1250, sma+1250
fma = fit_wall_pixels(ili, sma)
t2 = time()
xcm, ycm, zcm, xim, yim = ice_boundary(ili, smi, sma, -5.56566902681778, cal_do, method='SLSQP')
t3 = time()
xcl, ycl, zcl, xil, yil = ice_boundary_ls(ili, smi, sma, -5.56566902681778, cal_do, method='lm')
t4 = time()
zim, zil = zcm, zcl

print(t3-t2,t4-t3)

plt.figure()
plt.imshow(imo, cmap='gray')
plt.plot(smi, ili,'b-',alpha=0.5)
plt.plot(sma, ili,'r-',alpha=0.5)
plt.plot(fma, ili,'y--',alpha=0.5)
plt.show()

plt.figure()
plt.plot(xim, yim,'.-')
plt.plot(xcm, ycm,'.-')
plt.plot(xil, yil,'.-')
plt.plot(xcl, ycl,'.-')
plt.axis('equal')
plt.show()


# i = 10000 + 8
# imo = np.array( uvid1.get_data(i) )
# im = grayscale_im(imo[:,1250:2330])
# ili, smi, sma = laser_edges(im, sigma=10)

# xc, yc, zc, xi, yi = ice_boundary(ili, smi, sma, -5.56566902681778, cal_up, method='SLSQP')
# zi = zc
# z2,y2 = zc, yc

# # print(sma[800], smi[800])
# # print( np.std(np.gradient(sma)), np.std(np.gradient(smi)) )

# plt.figure()
# plt.imshow(imo) #, cmap='gray')
# plt.plot(smi+1250, ili,'b-',alpha=0.5)
# plt.plot(sma+1250, ili,'r-',alpha=0.5)
# plt.show()



# plt.figure()
# plt.plot(z1, y1, '.-')
# plt.plot(z2, y2, '.-')
# plt.title(j)
# plt.grid()
# plt.show()

#%%


a = np.array([1,2,3,3])
b = np.array([2,1,4,6])
c = np.array([5,2,1,2])

np.min((a,b,c),axis=0)


#%%
# if std(dp) > 10 (std of the gradient of sma) then don't use that frame 
# (maybe if its divided one part is usable, but I will avoid the trouble of dividing, I should have ennough frames to cover for that) 

# mes,sds = [],[]
mis,mas = [],[]
for i in tqdm(range(10000,10500,1)):
    im = np.array( dvid1.get_data(i) )[:,1250:2300]
    # im = np.array( uvid1.get_data(i) )[:,1250:2300]
    im = grayscale_im(im)
    ili, smi, sma = laser_edges(im, sigma=10)
    
    # g2 = normalize( gaussian(im, 10) )
    # dg = np.gradient(g2,axis=1) 
    
    mas.append(np.std(np.gradient(sma)))
    mis.append(np.std(np.gradient(smi)))
    
    

plt.figure()
plt.plot(mas,'.-')
plt.grid()
plt.show()
plt.figure()
plt.plot(mis,'.-')
plt.grid()
plt.show()

#%%



# ax = plt.figure().add_subplot(projection='3d')
# for i in range(60):
#     # if i == 29: continue
# # for i in [9,19,29]:
#     ax.plot( zcs[i], xcs[i], ycs[i], 'r-' )
#     # ax.plot( zis[i], xis[i], yis[i], '-', label=i )
    
# ax.set_xlabel('z (mm)')
# ax.set_ylabel('x (mm)')
# ax.set_zlabel('y (mm)')
# # plt.legend()
# # ax.set_box_aspect([1,1,1])
# plt.show()

def cuad(x, a,b,c):
    return a * x**2 + b * x + c
def lini(x, m,o):
    return m * x + o

# i = 0
# aaa, bbb = np.hstack( (dzcs[i], uzcs[i]) ), np.hstack( (dycs[i], uycs[i]) )
# (a,b,c), cov = curve_fit(cuad, bbb, aaa)

aes, bes, ces = [],[],[] 
plt.figure()
for i in range(0,29,1):
    
    # aaa, bbb = np.hstack( (dzcs[i][::-1], uzcs[i]) ), np.hstack( (dycs[i][::-1], uycs[i]) )
    # (a,b,c), cov = curve_fit(cuad, bbb, aaa)

    (a,b,c), cov = curve_fit(cuad, ili, dsma[i])
    (m,o), cov = curve_fit(lini, ili, dsma[i])

    plt.plot( dsma[i], ili, '.-' )     
    plt.plot( cuad(ili,a,b,c), ili, '--')
    plt.plot( lini(ili,m,o), ili, '-')
    
    # plt.plot( usma[i], ili, '.-' )
        
    # plt.plot( dzcs[i], dycs[i], '.', label=i )    # plt.plot( zis[i], xis[i], '.-' )
    # plt.plot( uzcs[i], uycs[i], '.', label=i )    # plt.plot( zis[i], xis[i], '.-' )
    
    # plt.plot( aaa, bbb, '.-' )
    # plt.plot( cuad(bbb,a,b,c), bbb, '-' )
    # aes.append(a)
    # bes.append(b)
    # ces.append(c)

# plt.axis('equal')
plt.grid()
# plt.legend()
plt.show()

# plt.figure()
# plt.plot(aes, '.')
# plt.grid()
# plt.show()
# plt.figure()
# plt.plot(bes, '.')
# plt.grid()
# plt.show()


#%%



#%%
# =============================================================================
# Back iamge
# =============================================================================
# at 25 fps

bvid1 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera back/DSC_0436.MOV', 'ffmpeg') # 7272 frames, starts 3487 (for seeing 4480 )
bvid2 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera back/DSC_0437.MOV', 'ffmpeg') # 6624 frames
bvid3 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera back/DSC_0438.MOV', 'ffmpeg') # 6684 frames
bvid4 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera back/DSC_0439.MOV', 'ffmpeg') # 6876 frames
bvid5 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera back/DSC_0440.MOV', 'ffmpeg') # 6852 frames
bvid6 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera back/DSC_0441.MOV', 'ffmpeg') # 6648 frames
bvid7 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera back/DSC_0442.MOV', 'ffmpeg') # 4020 frames
bvid8 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera back/DSC_0443.MOV', 'ffmpeg') # 6924 frames, ends 5658 (for seeing 5424)


# print(bvid1.count_frames(), bvid2.count_frames(), bvid3.count_frames(), bvid4.count_frames()) 
# print(bvid5.count_frames(), bvid6.count_frames(), bvid7.count_frames(), bvid8.count_frames()) 

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


pes, aas = [],[] 
for i in tqdm(range(4480, 7271, 10)):
# for i in [5350,5360,5370]: #[5000,5460]:
    im = np.array( bvid1.get_data(i) )
    im = grayscale_im(im)
    img = gaussian(im,15)
    aa = np.mean( img[1100:1300,700:3500],axis=0 )
    peks,pprop = find_peaks(-aa, prominence=30, wlen=120)
    pep = nan_argmax(peks, pprop)

    pes.append(pep)
    aas.append(aa)
    
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
    # plt.figure()
    # plt.plot(aa, '-')
    # plt.plot(peks,aa[peks],'k.')
    # plt.plot(pep,aa[pep],'r.')
    # plt.show()
    
    

#%%

plt.figure()
# for i in range(10):
#     plt.plot(aas[i],'-',zorder=-1)
#     plt.plot(pes[i],aas[i][pes[i]],'k.',zorder=0)


plt.plot(pes,'.-')

plt.show()



#%%







#%%

ii,jj = [],[]
for i,j in tqdm(enumerate(range(10000, 10060,1))):
    ii.append(i)
    jj.append(j)
    
print( np.vstack((ii, jj)).T )



#%%










