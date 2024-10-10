import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from sklearn.preprocessing import minmax_scale
from scipy.stats import spearmanr
from scipy import stats
import scipy as sp
import scipy.stats
import matplotlib as mpl
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('TkAgg')

basemap_fontsize = 40
colorbar_fontsize = 40
title_fontsize = 55
tick_fontsize = 45

# 0, 0.05, 0.1, 0.15, 0.2
cor_mask = np.fromfile('/home/geunhyeong/frp/data/mask/cor.NN_FWI_Anom_mask005.gdat', np.float32)
cor_mask = cor_mask.reshape(90,180)
cor_mask[cor_mask<0.05] = -999
cor_mask[cor_mask>=0.05] = 0

rh_mask = np.fromfile('/home/geunhyeong/frp/data/mask/nn_rh_mask.gdat', np.float32)
rh_mask = rh_mask.reshape(90,180)
rh_mask[rh_mask==1] = -999

prcp_mask = np.fromfile('/home/geunhyeong/frp/data/mask/fwi_prcp_mask.gdat', np.float32)
prcp_mask = prcp_mask.reshape(90,180)
prcp_mask[prcp_mask==1] = -999

mask = cor_mask + rh_mask + prcp_mask
# mask = cor_mask
mask[mask<0] = 1
data_mask = mask.copy()

mask = np.ma.masked_equal(mask, 0)

data_mask = data_mask.reshape(1,90,180)
data_mask = np.repeat(data_mask,7305,axis=0)
data_mask[data_mask==1] = -999


##########################################################
x=45
y=45

#x_start0 = 0.57*45/x + 0.50*45/x
x_start0 = 0.31
y_start1 = 0.95*35/y
y_start0 = y_start1 - 0.32*1*35/y - 0.1*35
y_start0 = 0.95

x_size = 0.45*45/x
y_size = 0.25*35/y + 0.04*35/y
y_size = 0.29


fig = plt.figure(figsize=(x,y))

ax0 = [x_start0,y_start0,x_size,y_size] #(a)

plt.axes(ax0)

cor_mask = np.ma.masked_equal(cor_mask, -999)

x1, y1 = np.meshgrid(np.arange(180,540,2), np.arange(-90,92,2))

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5,zorder=5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(cor_mask,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower', clim=[-0.3,0.31],zorder=1)
plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2],zorder=4)

plt.title('(a) Grid points selected for bin-averaged FRP calculation',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)


## set mask ---------------------------------------------------------------------------------------

# nn & fwi cor diff [ 0.05, 0.075, 0.1 ]
nn_mask = np.fromfile('/home/geunhyeong/frp/data/NN_Cor_Origin_Anom.gdat', np.float32)
nn_mask = nn_mask.reshape(90,180)

fwi_mask = np.fromfile('/home/geunhyeong/frp/data/FWI_Cor_Origin_Anom.gdat', np.float32)
fwi_mask = fwi_mask.reshape(90,180)
fwi_mask = np.where(np.isnan(fwi_mask) == True, 0, fwi_mask)

# diff 0.05
cor_mask = nn_mask - fwi_mask
cor_mask[cor_mask<0.05] = -999
cor_mask[cor_mask>=0.05] = 0

# rh, prcp mask
rh_mask = np.fromfile('/home/geunhyeong/frp/data/mask/nn_rh_mask.gdat', np.float32)
rh_mask = rh_mask.reshape(90,180)
rh_mask[rh_mask==1] = -999 

prcp_mask = np.fromfile('/home/geunhyeong/frp/data/mask/fwi_prcp_mask.gdat', np.float32)
prcp_mask = prcp_mask.reshape(90,180)
prcp_mask[prcp_mask==1] = -999

# data_mask
mask = cor_mask + rh_mask + prcp_mask
mask[mask<0] = 1
data_mask = mask.copy()

mask = np.ma.masked_equal(mask, 0)

data_mask = data_mask.reshape(1,90,180)
data_mask = np.repeat(data_mask,7305,axis=0)
data_mask[data_mask==1] = -999


## Load Data --------------------------------------------------------------------------------------

# frp
frp = np.fromfile('/home/geunhyeong/frp/data/ncl_2001_2020_ver2.gdat', np.float32)
frp = frp.reshape(-1,90,180)

frp = np.ma.masked_equal(frp, -999)
frp[frp==0] = 1
frp = np.log(frp)
frp = frp + data_mask
frp[frp<-200] = -999
frp = np.ma.masked_equal(frp, -999)

# nn frp
nn_frp = np.zeros((1,90,180))
for i in np.arange(1,6):
    dat = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred'+str(i)+'.gdat', np.float32) # 1461,90,180
    nn_frp = np.append(nn_frp, dat.reshape(-1,90,180), axis=0)
nn_frp = nn_frp[1:] # 7305,90,180
  
nn_frp = nn_frp + data_mask
nn_frp[nn_frp<-200] = -999
nn_frp = np.ma.masked_equal(nn_frp, -999)

# fwi frp
fwi_frp = np.zeros((1,90,180))
for i in np.arange(1,6):
    dat = np.fromfile('/home/geunhyeong/frp/output/FWI_result/fwi_pred'+str(i)+'.gdat', np.float32) # 1461,90,180
    fwi_frp = np.append(fwi_frp, dat.reshape(-1,90,180), axis=0)
fwi_frp = fwi_frp[1:]
  
fwi_frp = fwi_frp + data_mask
fwi_frp[fwi_frp<-200] = -999
fwi_frp = np.ma.masked_equal(fwi_frp, -999)

# RH
rh = np.fromfile('/home/geunhyeong/frp/data/rh_2001_2020_ver2.gdat', np.float32)
rh = rh.reshape(-1,90,180)
rh = rh + data_mask
rh[rh<-200] = -999
rh = np.ma.masked_equal(rh, -999)

# Prcp
prcp = np.fromfile('/home/geunhyeong/frp/data/prcp_2001_2020_ver2.gdat', np.float32)
prcp = prcp.reshape(-1,90,180)
prcp = prcp + data_mask
prcp[prcp<-200] = -999
prcp = np.ma.masked_equal(prcp, -999)

## ------------------------------------------------------------------------------------------------

frp = frp.compressed()
nn_frp = nn_frp.compressed()
fwi_frp = fwi_frp.compressed()
rh = rh.compressed()
prcp = prcp.compressed()


# rh_bins
rh_bins = [(rh >= i) & (rh < i+10) for i in range(0, 100, 10)]

rh_pdf = np.array([frp[indices].shape[0] for indices in rh_bins])
rh_pdf = rh_pdf / np.sum(rh_pdf) * 100
rh_pdf = np.round(rh_pdf, 2)

rh_obs_bin = np.array([np.mean(frp[indices]) for indices in rh_bins])
rh_nn_bin  = np.array([np.mean(nn_frp[indices]) for indices in rh_bins])
rh_fwi_bin = np.array([np.mean(fwi_frp[indices]) for indices in rh_bins])

# RMSE
rh_nn_rmse  = np.array([mean_squared_error(frp[indices], nn_frp[indices], squared=False) for indices in rh_bins])
rh_fwi_rmse = np.array([mean_squared_error(frp[indices], fwi_frp[indices], squared=False) for indices in rh_bins])


rh_obs_bin_dif = np.zeros(9)
rh_nn_bin_dif  = np.zeros(9)
rh_fwi_bin_dif = np.zeros(9)

for i in range(9):
    rh_obs_bin_dif[i] = rh_obs_bin[i+1] - rh_obs_bin[i]
    rh_nn_bin_dif[i] = rh_nn_bin[i+1] - rh_nn_bin[i]
    rh_fwi_bin_dif[i] = rh_fwi_bin[i+1] - rh_fwi_bin[i]

## ------------------------------------------------------------------------------------------------
# OBS
obs_rh_fits = []

for indices in rh_bins:
    model = LinearRegression()
    model.fit(rh[indices].reshape(-1,1), frp[indices].reshape(-1,1))
    obs_rh_fits.append(model)

obs_rh_coef = np.array([fit.coef_ for fit in obs_rh_fits]).reshape(-1)

# NN
nn_rh_fits = []

for indices in rh_bins:
    model = LinearRegression()
    model.fit(rh[indices].reshape(-1,1), nn_frp[indices].reshape(-1,1))
    nn_rh_fits.append(model)

nn_rh_coef = np.array([fit.coef_ for fit in nn_rh_fits]).reshape(-1)

# FWI
fwi_rh_fits = []

for indices in rh_bins:
    model = LinearRegression()
    model.fit(rh[indices].reshape(-1,1), fwi_frp[indices].reshape(-1,1))
    fwi_rh_fits.append(model)

fwi_rh_coef = np.array([fit.coef_ for fit in fwi_rh_fits]).reshape(-1)


rh_x = np.array([5,15,25,35,45,55,65,75,85,95])
rh_x2 = np.array([10,20,30,40,50,60,70,80,90])


## figure -----------------------------------------------------------------------------------------

x_start1 = 0.05
x_start2 = 0.57

y_start1 = 0.95*35/y
y_start2 = y_start1 - 0.32*1*35/y
y_start3 = y_start1 - 0.32*2*35/y
y_start4 = y_start1 - 0.32*3*35/y

x_size = 0.45
y_size = 0.25*35/y


#      x위치,y위치,x크기,y크기
ax1 = [x_start1,y_start1,x_size,y_size] #(a)
ax2 = [x_start2,y_start1,x_size,y_size] #(d)

ax3 = [x_start1,y_start2,x_size,y_size] #(b)
ax4 = [x_start2,y_start2,x_size,y_size] #(e)

ax5 = [x_start1,y_start3,x_size,y_size] #(c)
ax6 = [x_start2,y_start3,x_size,y_size] #(f)


# ax1
plt.axes(ax1)

tick_label = ['5', '15', '25', '35', '45', '55', '65', '75', '85', '95']
plt.bar(rh_x, rh_obs_bin, color='k', width=5, tick_label=tick_label)
plt.ylabel('Total OBS FRP',fontsize=50)
plt.ylim(0.0,5)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(visible=True, axis='y', linewidth=2)
plt.title('(b) OBS',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax2
plt.axes(ax2)

tick_label = ['15-5', '25-15', '35-25', '45-35', '55-45', '65-55', '75-65', '85-75', '95-85']
plt.bar(rh_x2,rh_obs_bin_dif,color='b',width=5,tick_label=tick_label)
plt.ylabel('OBS FRP Difference',fontsize=50)
plt.ylim(-0.8,0.8)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(visible=True, axis='y', linewidth=2)
plt.title('(c) OBS Bin Difference',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax3
plt.axes(ax3)

tick_label = ['5', '15', '25', '35', '45', '55', '65', '75', '85', '95']
plt.bar(rh_x,rh_nn_bin,color='k',width=5,tick_label=tick_label)
plt.ylabel('Total NN FRP',fontsize=50)
plt.ylim(0.0,5)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(visible=True, axis='y', linewidth=2)
plt.title('(d) FFNNs',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax4
plt.axes(ax4)

tick_label = ['15-5', '25-15', '35-25', '45-35', '55-45', '65-55', '75-65', '85-75', '95-85']
plt.bar(rh_x2,rh_nn_bin_dif,color='b',width=5,tick_label=tick_label)
plt.ylabel('NN FRP Difference',fontsize=50)
plt.ylim(-0.8,0.8)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(visible=True, axis='y', linewidth=2)
plt.title('(e) FFNNs Bin Difference',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax5
plt.axes(ax5)

tick_label = ['5', '15', '25', '35', '45', '55', '65', '75', '85', '95']
plt.bar(rh_x, rh_fwi_bin, color='k',width=5,tick_label=tick_label)
plt.xlabel('Total RH(%)',fontsize=50,labelpad=25)
plt.ylabel('Total FWI FRP',fontsize=50)
plt.ylim(0.0,5)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(visible=True, axis='y', linewidth=2)
plt.title('(f) FWI-based model',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax6
plt.axes(ax6)

tick_label = ['15-5', '25-15', '35-25', '45-35', '55-45', '65-55', '75-65', '85-75', '95-85']
plt.bar(rh_x2, rh_fwi_bin_dif, color='b',width=5,tick_label=tick_label)
plt.xlabel('Total RH(%)',fontsize=50,labelpad=25)
plt.ylabel('FWI FRP Difference',fontsize=50)
plt.ylim(-0.8,0.8)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(visible=True, axis='y', linewidth=2)
plt.title('(g) FWI-based model Bin Difference',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

plt.savefig('/home/geunhyeong/frp/fig/png/github/Fig.6.png',format='png',dpi=300,bbox_inches='tight')

