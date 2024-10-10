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
# import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys

import matplotlib
matplotlib.use('TkAgg')

basemap_fontsize = 40
colorbar_fontsize = 40
title_fontsize = 50

# 0, 0.05, 0.1, 0.15, 0.2
cor_mask = np.fromfile('/home/geunhyeong/frp/data/mask/cor.NN_FWI_Anom_mask005.gdat', np.float32)
cor_mask = cor_mask.reshape(90,180)
cor_mask[cor_mask<0.05] = -999
cor_mask[cor_mask>=0.05] = 0

#cor_mask = np.ma.masked_where(cor_mask<0.0 ,cor_mask)

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
nn_frp1 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred1.gdat', np.float32).reshape(-1,90,180)
nn_frp2 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred2.gdat', np.float32).reshape(-1,90,180)
nn_frp3 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred3.gdat', np.float32).reshape(-1,90,180)
nn_frp4 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred4.gdat', np.float32).reshape(-1,90,180)
nn_frp5 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred5.gdat', np.float32).reshape(-1,90,180)

nn_frp = np.concatenate((nn_frp1, nn_frp2, nn_frp3, nn_frp4, nn_frp5), axis=0)

nn_frp = nn_frp + data_mask
nn_frp[nn_frp<-200] = -999
nn_frp = np.ma.masked_equal(nn_frp, -999)

# fwi frp
fwi_frp1 = np.fromfile('/home/geunhyeong/frp/output/FWI_result/fwi_pred1.gdat', np.float32).reshape(-1,90,180)
fwi_frp2 = np.fromfile('/home/geunhyeong/frp/output/FWI_result/fwi_pred2.gdat', np.float32).reshape(-1,90,180)
fwi_frp3 = np.fromfile('/home/geunhyeong/frp/output/FWI_result/fwi_pred3.gdat', np.float32).reshape(-1,90,180)
fwi_frp4 = np.fromfile('/home/geunhyeong/frp/output/FWI_result/fwi_pred4.gdat', np.float32).reshape(-1,90,180)
fwi_frp5 = np.fromfile('/home/geunhyeong/frp/output/FWI_result/fwi_pred5.gdat', np.float32).reshape(-1,90,180)

fwi_frp = np.concatenate((fwi_frp1, fwi_frp2, fwi_frp3, fwi_frp4, fwi_frp5), axis=0)

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


frp = frp.compressed()
nn_frp = nn_frp.compressed()
fwi_frp = fwi_frp.compressed()
rh = rh.compressed()
prcp = prcp.compressed()


prcp_pdf = np.zeros(70)
prcp_obs_bin = np.zeros(70)
prcp_nn_bin = np.zeros(70)
prcp_fwi_bin = np.zeros(70)

obs_prcp_coef = np.zeros(70)
nn_prcp_coef = np.zeros(70)
fwi_prcp_coef = np.zeros(70)

prcp_x = np.linspace(0,7,70)
tick_label = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
for i in range(70):
    prcp_bin = (prcp >= i/10) & (prcp < i/10+0.1)
    
    prcp_pdf[i] = frp[prcp_bin].shape[0]
    prcp_obs_bin[i] = np.mean(frp[prcp_bin])
    prcp_nn_bin[i] = np.mean(nn_frp[prcp_bin])
    prcp_fwi_bin[i] = np.mean(fwi_frp[prcp_bin])
    
    obs_prcp_fit = LinearRegression()
    nn_prcp_fit = LinearRegression()
    fwi_prcp_fit = LinearRegression()
    obs_prcp_fit.fit(prcp[prcp_bin].reshape(-1,1), frp[prcp_bin].reshape(-1,1))
    nn_prcp_fit.fit(prcp[prcp_bin].reshape(-1,1), nn_frp[prcp_bin].reshape(-1,1))
    fwi_prcp_fit.fit(prcp[prcp_bin].reshape(-1,1), fwi_frp[prcp_bin].reshape(-1,1))
    obs_prcp_coef[i] = obs_prcp_fit.coef_
    nn_prcp_coef[i] = nn_prcp_fit.coef_
    fwi_prcp_coef[i] = fwi_prcp_fit.coef_
    


prcp_pdf = prcp_pdf/np.sum(prcp_pdf) * 100
prcp_pdf = np.round(prcp_pdf,2)

#print
prcp01 = (prcp >= 0) & (prcp < 0.1)
prcp02 = (prcp >= 0.1) & (prcp < 0.2)
prcp03 = (prcp >= 0.2) & (prcp < 0.3)
prcp04 = (prcp >= 0.3) & (prcp < 0.4)
prcp05 = (prcp >= 0.4) & (prcp < 0.5)
prcp06 = (prcp >= 0.5) & (prcp < 0.6)
prcp07 = (prcp >= 0.6) & (prcp < 0.7)
prcp08 = (prcp >= 0.7) & (prcp < 0.8)
prcp09 = (prcp >= 0.8) & (prcp < 0.9)
prcp10 = (prcp >= 0.9) & (prcp < 1.0)


# OBS
obs_prcp01_fit = LinearRegression()
obs_prcp01_fit.fit(prcp[prcp01].reshape(-1,1), frp[prcp01].reshape(-1,1))
obs_prcp02_fit = LinearRegression()
obs_prcp02_fit.fit(prcp[prcp02].reshape(-1,1), frp[prcp02].reshape(-1,1))
obs_prcp03_fit = LinearRegression()
obs_prcp03_fit.fit(prcp[prcp03].reshape(-1,1), frp[prcp03].reshape(-1,1))
obs_prcp04_fit = LinearRegression()
obs_prcp04_fit.fit(prcp[prcp04].reshape(-1,1), frp[prcp04].reshape(-1,1))
obs_prcp05_fit = LinearRegression()
obs_prcp05_fit.fit(prcp[prcp05].reshape(-1,1), frp[prcp05].reshape(-1,1))
obs_prcp06_fit = LinearRegression()
obs_prcp06_fit.fit(prcp[prcp06].reshape(-1,1), frp[prcp06].reshape(-1,1))
obs_prcp07_fit = LinearRegression()
obs_prcp07_fit.fit(prcp[prcp07].reshape(-1,1), frp[prcp07].reshape(-1,1))
obs_prcp08_fit = LinearRegression()
obs_prcp08_fit.fit(prcp[prcp08].reshape(-1,1), frp[prcp08].reshape(-1,1))
obs_prcp09_fit = LinearRegression()
obs_prcp09_fit.fit(prcp[prcp09].reshape(-1,1), frp[prcp09].reshape(-1,1))
obs_prcp10_fit = LinearRegression()
obs_prcp10_fit.fit(prcp[prcp10].reshape(-1,1), frp[prcp10].reshape(-1,1))


obs_prcp_coef = np.array([obs_prcp01_fit.coef_, obs_prcp02_fit.coef_, obs_prcp03_fit.coef_, obs_prcp04_fit.coef_,
                          obs_prcp05_fit.coef_, obs_prcp06_fit.coef_, obs_prcp07_fit.coef_, obs_prcp08_fit.coef_,
                          obs_prcp09_fit.coef_, obs_prcp10_fit.coef_])

obs_prcp_coef = obs_prcp_coef.reshape(-1)

# NN
nn_prcp01_fit = LinearRegression()
nn_prcp01_fit.fit(prcp[prcp01].reshape(-1,1), nn_frp[prcp01].reshape(-1,1))
nn_prcp02_fit = LinearRegression()
nn_prcp02_fit.fit(prcp[prcp02].reshape(-1,1), nn_frp[prcp02].reshape(-1,1))
nn_prcp03_fit = LinearRegression()
nn_prcp03_fit.fit(prcp[prcp03].reshape(-1,1), nn_frp[prcp03].reshape(-1,1))
nn_prcp04_fit = LinearRegression()
nn_prcp04_fit.fit(prcp[prcp04].reshape(-1,1), nn_frp[prcp04].reshape(-1,1))
nn_prcp05_fit = LinearRegression()
nn_prcp05_fit.fit(prcp[prcp05].reshape(-1,1), nn_frp[prcp05].reshape(-1,1))
nn_prcp06_fit = LinearRegression()
nn_prcp06_fit.fit(prcp[prcp06].reshape(-1,1), nn_frp[prcp06].reshape(-1,1))
nn_prcp07_fit = LinearRegression()
nn_prcp07_fit.fit(prcp[prcp07].reshape(-1,1), nn_frp[prcp07].reshape(-1,1))
nn_prcp08_fit = LinearRegression()
nn_prcp08_fit.fit(prcp[prcp08].reshape(-1,1), nn_frp[prcp08].reshape(-1,1))
nn_prcp09_fit = LinearRegression()
nn_prcp09_fit.fit(prcp[prcp09].reshape(-1,1), nn_frp[prcp09].reshape(-1,1))
nn_prcp10_fit = LinearRegression()
nn_prcp10_fit.fit(prcp[prcp10].reshape(-1,1), nn_frp[prcp10].reshape(-1,1))


nn_prcp_coef = np.array([nn_prcp01_fit.coef_, nn_prcp02_fit.coef_, nn_prcp03_fit.coef_, nn_prcp04_fit.coef_,
                          nn_prcp05_fit.coef_, nn_prcp06_fit.coef_, nn_prcp07_fit.coef_, nn_prcp08_fit.coef_,
                          nn_prcp09_fit.coef_, nn_prcp10_fit.coef_])

nn_prcp_coef = nn_prcp_coef.reshape(-1)

# FWI
fwi_prcp01_fit = LinearRegression()
fwi_prcp01_fit.fit(prcp[prcp01].reshape(-1,1), fwi_frp[prcp01].reshape(-1,1))
fwi_prcp02_fit = LinearRegression()
fwi_prcp02_fit.fit(prcp[prcp02].reshape(-1,1), fwi_frp[prcp02].reshape(-1,1))
fwi_prcp03_fit = LinearRegression()
fwi_prcp03_fit.fit(prcp[prcp03].reshape(-1,1), fwi_frp[prcp03].reshape(-1,1))
fwi_prcp04_fit = LinearRegression()
fwi_prcp04_fit.fit(prcp[prcp04].reshape(-1,1), fwi_frp[prcp04].reshape(-1,1))
fwi_prcp05_fit = LinearRegression()
fwi_prcp05_fit.fit(prcp[prcp05].reshape(-1,1), fwi_frp[prcp05].reshape(-1,1))
fwi_prcp06_fit = LinearRegression()
fwi_prcp06_fit.fit(prcp[prcp06].reshape(-1,1), fwi_frp[prcp06].reshape(-1,1))
fwi_prcp07_fit = LinearRegression()
fwi_prcp07_fit.fit(prcp[prcp07].reshape(-1,1), fwi_frp[prcp07].reshape(-1,1))
fwi_prcp08_fit = LinearRegression()
fwi_prcp08_fit.fit(prcp[prcp08].reshape(-1,1), fwi_frp[prcp08].reshape(-1,1))
fwi_prcp09_fit = LinearRegression()
fwi_prcp09_fit.fit(prcp[prcp09].reshape(-1,1), fwi_frp[prcp09].reshape(-1,1))
fwi_prcp10_fit = LinearRegression()
fwi_prcp10_fit.fit(prcp[prcp10].reshape(-1,1), fwi_frp[prcp10].reshape(-1,1))


fwi_prcp_coef = np.array([fwi_prcp01_fit.coef_, fwi_prcp02_fit.coef_, fwi_prcp03_fit.coef_, fwi_prcp04_fit.coef_,
                          fwi_prcp05_fit.coef_, fwi_prcp06_fit.coef_, fwi_prcp07_fit.coef_, fwi_prcp08_fit.coef_,
                          fwi_prcp09_fit.coef_, fwi_prcp10_fit.coef_])

fwi_prcp_coef = fwi_prcp_coef.reshape(-1)



obs_poly = np.polyfit(prcp_x,prcp_obs_bin,2)
obs_p = np.poly1d(obs_poly)

nn_poly = np.polyfit(prcp_x,prcp_nn_bin,2)
nn_p = np.poly1d(nn_poly)

fwi_poly = np.polyfit(prcp_x,prcp_fwi_bin,2)
fwi_p = np.poly1d(fwi_poly)

frp = np.fromfile('/home/geunhyeong/frp/data/composite/obs_frp3_composite.gdat', np.float32)
nn_frp = np.fromfile('/home/geunhyeong/frp/data/composite/nn_frp3_composite.gdat', np.float32)
fwi_frp = np.fromfile('/home/geunhyeong/frp/data/composite/fwi_frp3_composite.gdat', np.float32)

frp = np.ma.masked_equal(frp, -999)
nn_frp = np.ma.masked_equal(nn_frp, -999)
fwi_frp = np.ma.masked_equal(fwi_frp, -999)

frp = frp.reshape(90,180)
nn_frp = nn_frp.reshape(90,180)
fwi_frp = fwi_frp.reshape(90,180)

mask = np.fromfile('/home/geunhyeong/frp/data/mask/frp_mask_final.gdat', np.float32)
mask = mask.reshape(90,180)
data_mask = mask.copy()
data_mask[data_mask==1] = -999

mask = np.ma.masked_equal(mask, 0)



## figure -----------------------------------------------------------------------------------------
x_start1 = 0.05
x_start2 = 0.55

y_start1 = 0.95
y_start2 = y_start1 - 0.30*1
y_start3 = y_start1 - 0.30*2
y_start4 = y_start1 - 0.30*3

x_size = 0.45
x_size2 = 0.5
y_size = 0.25

fig = plt.figure(figsize=(45,35))

#      x위치,y위치,x크기,y크기
ax1 = [x_start1,y_start1,x_size,y_size] #(a)
ax2 = [x_start2,y_start1,x_size2,y_size] #(d)

ax3 = [x_start1,y_start2,x_size,y_size] #(b)
ax4 = [x_start2,y_start2,x_size2,y_size] #(e)

ax5 = [x_start1,y_start3,x_size,y_size] #(c)
ax6 = [x_start2,y_start3,x_size2,y_size] #(f)


basemap_fontsize = 40
colorbar_fontsize = 40
title_fontsize = 50


# ax1
plt.axes(ax1)

plt.bar(prcp_x,prcp_obs_bin,color='y',width=0.025)
plt.plot(prcp_x,obs_p(prcp_x),color='k',label='reg='+str(np.round(obs_poly[0],3)))
plt.ylabel('Total OBS FRP',fontsize=50,labelpad=25)
plt.ylim(0.0,2.0)
plt.xticks(fontsize=basemap_fontsize)
plt.yticks(fontsize=basemap_fontsize)
plt.legend(fontsize=40)
plt.grid(True,axis='y', linewidth=2)
plt.title('(a) OBS',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax2
plt.axes(ax2)

x1, y1 = np.meshgrid(np.arange(180,540,2), np.arange(-90,92,2))

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5,zorder=5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(frp,axis=0), cmap=plt.cm.Reds,extent=[180,540,92,-90], origin='lower', clim=[0,3.0],zorder=1)
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(0,3.1,0.5))
cbar.ax.set_yticklabels(np.round(np.arange(0,3.1,0.5),1),fontsize=colorbar_fontsize)

plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2],zorder=4)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)

plt.title('(b) OBS FRP (PRCP > 3)',fontsize=title_fontsize,loc='left',pad=30)

# ax3
plt.axes(ax3)

plt.bar(prcp_x,prcp_nn_bin,color='y',width=0.025)
plt.plot(prcp_x,nn_p(prcp_x),color='k',label='reg='+str(np.round(nn_poly[0],3)))
plt.ylabel('Total NN FRP',fontsize=50,labelpad=25)
plt.ylim(0.0,2.0)
plt.xticks(fontsize=basemap_fontsize)
plt.yticks(fontsize=basemap_fontsize)
plt.legend(fontsize=40)
plt.grid(True,axis='y', linewidth=2)
plt.title('(c) FFNNs',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax4
plt.axes(ax4)

x1, y1 = np.meshgrid(np.arange(180,540,2), np.arange(-90,92,2))

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5,zorder=5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(nn_frp,axis=0), cmap=plt.cm.Reds,extent=[180,540,92,-90], origin='lower', clim=[0,3.0],zorder=1)
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(0,3.1,0.5))
cbar.ax.set_yticklabels(np.round(np.arange(0,3.1,0.5),1),fontsize=colorbar_fontsize)

plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2],zorder=4)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)

plt.title('(d) FFNNs FRP (PRCP > 3)',fontsize=title_fontsize,loc='left',pad=30)

# ax5
plt.axes(ax5)

plt.bar(prcp_x,prcp_fwi_bin,color='y',width=0.025)
plt.plot(prcp_x,fwi_p(prcp_x),color='k',label='reg='+str(np.round(fwi_poly[0],3)))
plt.xlabel('Total prcp(mm/day)',fontsize=50,labelpad=50)
plt.ylabel('Total FWI FRP',fontsize=50,labelpad=25)
plt.ylim(0.0,2.0)
plt.xticks(fontsize=basemap_fontsize)
plt.yticks(fontsize=basemap_fontsize)
plt.legend(fontsize=40)
plt.grid(True,axis='y', linewidth=2)
plt.title('(e) FWI-based model',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax6
plt.axes(ax6)

x1, y1 = np.meshgrid(np.arange(180,540,2), np.arange(-90,92,2))

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5,zorder=5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(fwi_frp,axis=0), cmap=plt.cm.Reds,extent=[180,540,92,-90], origin='lower', clim=[0,3.0],zorder=1)
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(0,3.1,0.5))
cbar.ax.set_yticklabels(np.round(np.arange(0,3.1,0.5),1),fontsize=colorbar_fontsize)

plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2],zorder=4)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)

plt.title('(f) FWI-based model FRP (PRCP > 3)',fontsize=title_fontsize,loc='left',pad=30)

plt.savefig('/home/geunhyeong/frp/fig/png/github/Fig.7.png',format='png',dpi=300,bbox_inches='tight')