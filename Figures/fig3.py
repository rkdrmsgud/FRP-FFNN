import numpy as np
import pickle
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as patches
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')

mask = np.fromfile('/home/geunhyeong/frp/data/mask/frp_mask_final.gdat', np.float32) # mask value : 1
mask = mask.reshape(90,180)

data_mask = mask.copy()
data_mask[data_mask==1] = -999

mask = np.ma.masked_equal(mask, 0)

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

# 1. South America
lon1 = -64 # 시작지점의 경도 (-180~180)
lat1 = -21 # 시작지점의 위도 (-90~90)
x_len1 = 24 # 가로길이 interval : 1 degree *왠만하면 짝수로 설정할것
y_len1 = 20 # 세로길이 interval : 1 degree *왠만하면 짝수로 설정할것

# 2. South Africa
lon2 = 14 # 시작지점의 경도 (-180~180)
lat2 = -18 # 시작지점의 위도 (-90~90)
x_len2 = 22 # 가로길이 interval : 1 degree *왠만하면 짝수로 설정할것
y_len2 = 24 # 세로길이 interval : 1 degree *왠만하면 짝수로 설정할것

# 3. Siberia
lon3 = 104 # 시작지점의 경도 (-180~180)
lat3 = 48 # 시작지점의 위도 (-90~90)
x_len3 = 30 # 가로길이 interval : 1 degree *왠만하면 짝수로 설정할것
y_len3 = 12 # 세로길이 interval : 1 degree *왠만하면 짝수로 설정할것

# 4. South China
lon4 = 108 # 시작지점의 경도 (-180~180)
lat4 = 22 # 시작지점의 위도 (-90~90)
x_len4 = 12 # 가로길이 interval : 1 degree *왠만하면 짝수로 설정할것
y_len4 = 8 # 세로길이 interval : 1 degree *왠만하면 짝수로 설정할것

x_start1 = int(lon1/2 + 90)
x_end1 = int(x_start1 + x_len1/2)
y_start1 = int(lat1/2 + 45)
y_end1 = int(y_start1 + y_len1/2)

x_start2 = int(lon2/2 + 90)
x_end2 = int(x_start2 + x_len2/2)
y_start2 = int(lat2/2 + 45)
y_end2 = int(y_start2 + y_len2/2)

x_start3 = int(lon3/2 + 90)
x_end3 = int(x_start3 + x_len3/2)
y_start3 = int(lat3/2 + 45)
y_end3 = int(y_start3 + y_len3/2)

x_start4 = int(lon4/2 + 90)
x_end4 = int(x_start4 + x_len4/2)
y_start4 = int(lat4/2 + 45)
y_end4 = int(y_start4 + y_len4/2)

frp1 = frp[:,y_start1:y_end1,x_start1:x_end1]
nn_frp1 = nn_frp[:,y_start1:y_end1,x_start1:x_end1]
fwi_frp1 = fwi_frp[:,y_start1:y_end1,x_start1:x_end1]

frp2 = frp[:,y_start2:y_end2,x_start2:x_end2]
nn_frp2 = nn_frp[:,y_start2:y_end2,x_start2:x_end2]
fwi_frp2 = fwi_frp[:,y_start2:y_end2,x_start2:x_end2]

frp3 = frp[:,y_start3:y_end3,x_start3:x_end3]
nn_frp3 = nn_frp[:,y_start3:y_end3,x_start3:x_end3]
fwi_frp3 = fwi_frp[:,y_start3:y_end3,x_start3:x_end3]

frp4 = frp[:,y_start4:y_end4,x_start4:x_end4]
nn_frp4 = nn_frp[:,y_start4:y_end4,x_start4:x_end4]
fwi_frp4 = fwi_frp[:,y_start4:y_end4,x_start4:x_end4]

del frp, nn_frp, fwi_frp


frp1 = np.delete(frp1, 365*3+31+29, axis=0)
frp1 = np.delete(frp1, 365*7+31+29, axis=0)
frp1 = np.delete(frp1, 365*11+31+29, axis=0)
frp1 = np.delete(frp1, 365*15+31+29, axis=0)
frp1 = np.delete(frp1, 365*19+31+29, axis=0)

frp2 = np.delete(frp2, 365*3+31+29, axis=0)
frp2 = np.delete(frp2, 365*7+31+29, axis=0)
frp2 = np.delete(frp2, 365*11+31+29, axis=0)
frp2 = np.delete(frp2, 365*15+31+29, axis=0)
frp2 = np.delete(frp2, 365*19+31+29, axis=0)

frp3 = np.delete(frp3, 365*3+31+29, axis=0)
frp3 = np.delete(frp3, 365*7+31+29, axis=0)
frp3 = np.delete(frp3, 365*11+31+29, axis=0)
frp3 = np.delete(frp3, 365*15+31+29, axis=0)
frp3 = np.delete(frp3, 365*19+31+29, axis=0)

frp4 = np.delete(frp4, 365*3+31+29, axis=0)
frp4 = np.delete(frp4, 365*7+31+29, axis=0)
frp4 = np.delete(frp4, 365*11+31+29, axis=0)
frp4 = np.delete(frp4, 365*15+31+29, axis=0)
frp4 = np.delete(frp4, 365*19+31+29, axis=0)

nn_frp1 = np.delete(nn_frp1, 365*3+31+29, axis=0)
nn_frp1 = np.delete(nn_frp1, 365*7+31+29, axis=0)
nn_frp1 = np.delete(nn_frp1, 365*11+31+29, axis=0)
nn_frp1 = np.delete(nn_frp1, 365*15+31+29, axis=0)
nn_frp1 = np.delete(nn_frp1, 365*19+31+29, axis=0)

nn_frp2 = np.delete(nn_frp2, 365*3+31+29, axis=0)
nn_frp2 = np.delete(nn_frp2, 365*7+31+29, axis=0)
nn_frp2 = np.delete(nn_frp2, 365*11+31+29, axis=0)
nn_frp2 = np.delete(nn_frp2, 365*15+31+29, axis=0)
nn_frp2 = np.delete(nn_frp2, 365*19+31+29, axis=0)

nn_frp3 = np.delete(nn_frp3, 365*3+31+29, axis=0)
nn_frp3 = np.delete(nn_frp3, 365*7+31+29, axis=0)
nn_frp3 = np.delete(nn_frp3, 365*11+31+29, axis=0)
nn_frp3 = np.delete(nn_frp3, 365*15+31+29, axis=0)
nn_frp3 = np.delete(nn_frp3, 365*19+31+29, axis=0)

nn_frp4 = np.delete(nn_frp4, 365*3+31+29, axis=0)
nn_frp4 = np.delete(nn_frp4, 365*7+31+29, axis=0)
nn_frp4 = np.delete(nn_frp4, 365*11+31+29, axis=0)
nn_frp4 = np.delete(nn_frp4, 365*15+31+29, axis=0)
nn_frp4 = np.delete(nn_frp4, 365*19+31+29, axis=0)

fwi_frp1 = np.delete(fwi_frp1, 365*3+31+29, axis=0)
fwi_frp1 = np.delete(fwi_frp1, 365*7+31+29, axis=0)
fwi_frp1 = np.delete(fwi_frp1, 365*11+31+29, axis=0)
fwi_frp1 = np.delete(fwi_frp1, 365*15+31+29, axis=0)
fwi_frp1 = np.delete(fwi_frp1, 365*19+31+29, axis=0)

fwi_frp2 = np.delete(fwi_frp2, 365*3+31+29, axis=0)
fwi_frp2 = np.delete(fwi_frp2, 365*7+31+29, axis=0)
fwi_frp2 = np.delete(fwi_frp2, 365*11+31+29, axis=0)
fwi_frp2 = np.delete(fwi_frp2, 365*15+31+29, axis=0)
fwi_frp2 = np.delete(fwi_frp2, 365*19+31+29, axis=0)

fwi_frp3 = np.delete(fwi_frp3, 365*3+31+29, axis=0)
fwi_frp3 = np.delete(fwi_frp3, 365*7+31+29, axis=0)
fwi_frp3 = np.delete(fwi_frp3, 365*11+31+29, axis=0)
fwi_frp3 = np.delete(fwi_frp3, 365*15+31+29, axis=0)
fwi_frp3 = np.delete(fwi_frp3, 365*19+31+29, axis=0)

fwi_frp4 = np.delete(fwi_frp4, 365*3+31+29, axis=0)
fwi_frp4 = np.delete(fwi_frp4, 365*7+31+29, axis=0)
fwi_frp4 = np.delete(fwi_frp4, 365*11+31+29, axis=0)
fwi_frp4 = np.delete(fwi_frp4, 365*15+31+29, axis=0)
fwi_frp4 = np.delete(fwi_frp4, 365*19+31+29, axis=0)


frp1 = frp1.reshape(7300,-1)
frp2 = frp2.reshape(7300,-1)
frp3 = frp3.reshape(7300,-1)
frp4 = frp4.reshape(7300,-1)

nn_frp1 = nn_frp1.reshape(7300,-1)
nn_frp2 = nn_frp2.reshape(7300,-1)
nn_frp3 = nn_frp3.reshape(7300,-1)
nn_frp4 = nn_frp4.reshape(7300,-1)

fwi_frp1 = fwi_frp1.reshape(7300,-1)
fwi_frp2 = fwi_frp2.reshape(7300,-1)
fwi_frp3 = fwi_frp3.reshape(7300,-1)
fwi_frp4 = fwi_frp4.reshape(7300,-1)

frp1 = np.mean(frp1, axis=1)
frp2 = np.mean(frp2, axis=1)
frp3 = np.mean(frp3, axis=1)
frp4 = np.mean(frp4, axis=1)

nn_frp1 = np.mean(nn_frp1, axis=1)
nn_frp2 = np.mean(nn_frp2, axis=1)
nn_frp3 = np.mean(nn_frp3, axis=1)
nn_frp4 = np.mean(nn_frp4, axis=1)

fwi_frp1 = np.mean(fwi_frp1, axis=1)
fwi_frp2 = np.mean(fwi_frp2, axis=1)
fwi_frp3 = np.mean(fwi_frp3, axis=1)
fwi_frp4 = np.mean(fwi_frp4, axis=1)

#=============================================================================================
# annual mean
frp1_annual = frp1.reshape(20,365)
frp2_annual = frp2.reshape(20,365)
frp3_annual = frp3.reshape(20,365)
frp4_annual = frp4.reshape(20,365)

nn_frp1_annual = nn_frp1.reshape(20,365)
nn_frp2_annual = nn_frp2.reshape(20,365)
nn_frp3_annual = nn_frp3.reshape(20,365)
nn_frp4_annual = nn_frp4.reshape(20,365)

fwi_frp1_annual = fwi_frp1.reshape(20,365)
fwi_frp2_annual = fwi_frp2.reshape(20,365)
fwi_frp3_annual = fwi_frp3.reshape(20,365)
fwi_frp4_annual = fwi_frp4.reshape(20,365)

frp1_annual = np.mean(frp1_annual, axis=1)
frp2_annual = np.mean(frp2_annual, axis=1)
frp3_annual = np.mean(frp3_annual, axis=1)
frp4_annual = np.mean(frp4_annual, axis=1)

nn_frp1_annual = np.mean(nn_frp1_annual, axis=1)
nn_frp2_annual = np.mean(nn_frp2_annual, axis=1)
nn_frp3_annual = np.mean(nn_frp3_annual, axis=1)
nn_frp4_annual = np.mean(nn_frp4_annual, axis=1)

fwi_frp1_annual = np.mean(fwi_frp1_annual, axis=1)
fwi_frp2_annual = np.mean(fwi_frp2_annual, axis=1)
fwi_frp3_annual = np.mean(fwi_frp3_annual, axis=1)
fwi_frp4_annual = np.mean(fwi_frp4_annual, axis=1)

#=============================================================================================
# daily

frp1_daily = frp1[365*19:365*19+365]
nn_frp1_daily = nn_frp1[365*19:365*19+365]
fwi_frp1_daily = fwi_frp1[365*19:365*19+365]

frp2_daily = frp2[365*16:365*16+365]
nn_frp2_daily = nn_frp2[365*16:365*16+365]
fwi_frp2_daily = fwi_frp2[365*16:365*16+365]

frp3_daily = frp3[365*3:365*3+365]
nn_frp3_daily = nn_frp3[365*3:365*3+365]
fwi_frp3_daily = fwi_frp3[365*3:365*3+365]

frp4_daily = frp4[365*7:365*7+365]
nn_frp4_daily = nn_frp4[365*7:365*7+365]
fwi_frp4_daily = fwi_frp4[365*7:365*7+365]

frp1_daily = frp1_daily[181:365]
nn_frp1_daily = nn_frp1_daily[181:365]
fwi_frp1_daily = fwi_frp1_daily[181:365]

frp2_daily = frp2_daily[151:334]
nn_frp2_daily = nn_frp2_daily[151:334]
fwi_frp2_daily = fwi_frp2_daily[151:334]

frp3_daily = frp3_daily[90:273]
nn_frp3_daily = nn_frp3_daily[90:273]
fwi_frp3_daily = fwi_frp3_daily[90:273]

frp4_daily = frp4_daily[0:181]
nn_frp4_daily = nn_frp4_daily[0:181]
fwi_frp4_daily = fwi_frp4_daily[0:181]

day1 = np.arange(1,frp1_daily.shape[0]+1,1)
day2 = np.arange(1,frp2_daily.shape[0]+1,1)
day3 = np.arange(1,frp3_daily.shape[0]+1,1)
day4 = np.arange(1,frp4_daily.shape[0]+1,1)


title_fontsize = 55 +7
labelsize = 45+7
fontsize = 55+7
legend_fontsize = 40+7

x_start1 = 0.05
x_start2 = 0.55

y_start1 = 0.95
y_start2 = y_start1 - 0.30*1
y_start3 = y_start1 - 0.30*2
y_start4 = y_start1 - 0.30*3

x_size = 0.45
y_size = 0.25

fig = plt.figure(figsize=(50,40))
year = np.arange(2001,2021,1)
tick_label = str(year)

#      x위치,y위치,x크기,y크기
ax1 = [x_start1,y_start1,x_size,y_size] #(a)
ax2 = [x_start2,y_start1,x_size,y_size] #(d)

ax3 = [x_start1,y_start2,x_size,y_size] #(b)
ax4 = [x_start2,y_start2,x_size,y_size] #(e)

ax5 = [x_start1,y_start3,x_size,y_size] #(c)
ax6 = [x_start2,y_start3,x_size,y_size] #(f)

ax7 = [x_start1,y_start4,x_size,y_size] # 75
ax8 = [x_start2,y_start4,x_size,y_size] # 75


# ax1
plt.axes(ax1)

plt.plot(year,frp1_annual,color='k',marker='o',label='OBS',linewidth=3,markersize=10)
plt.plot(year,nn_frp1_annual,color='r',marker='o',label='FFNNs (Cor=0.71)',linewidth=3,markersize=10)
plt.plot(year,fwi_frp1_annual,color='b',marker='o',label='FWI (Cor=0.63)',linewidth=3,markersize=10)
plt.xticks([2002,2004,2006,2008,2010,2012,2014,2016,2018,2020],
           label=['2002','2004','2006','2008','2010','2012','2014','2016','2018','2020'])
plt.ylabel('FRP',fontsize=fontsize,labelpad=20)
plt.tick_params(axis='x', length=10, width=2, labelsize=labelsize)
plt.tick_params(axis='y', length=10, width=2, labelsize=labelsize)
plt.legend(fontsize=legend_fontsize,loc='upper right')
plt.title('(a) 2001 - 2020 Brazil annual FRP',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax2
plt.axes(ax2)

plt.plot(day1,frp1_daily,color='k',marker='o',label='OBS',linewidth=3,markersize=10)
plt.plot(day1,nn_frp1_daily,color='r',marker='o',label='FFNNs (Cor=0.96)',linewidth=3,markersize=10)
plt.plot(day1,fwi_frp1_daily,color='b',marker='o',label='FWI (Cor=0.81)',linewidth=3,markersize=10)
plt.xticks([15,45,75,105,135,165],
           labels=['J','A','S','O','N','D'])
plt.ylabel('FRP',fontsize=fontsize,labelpad=20)
plt.tick_params(axis='x', length=10, width=2, labelsize=labelsize)
plt.tick_params(axis='y', length=10, width=2, labelsize=labelsize)
plt.legend(fontsize=legend_fontsize)
plt.title('(e) 2019 Brazil 7 - 12 daily FRP',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax3
plt.axes(ax3)

plt.plot(year,frp2_annual,color='k',marker='o',label='OBS',linewidth=3,markersize=10)
plt.plot(year,nn_frp2_annual,color='r',marker='o',label='FFNNs (Cor=0.60)',linewidth=3,markersize=10)
plt.plot(year,fwi_frp2_annual,color='b',marker='o',label='FWI (Cor=0.50)',linewidth=3,markersize=10)
plt.xticks([2002,2004,2006,2008,2010,2012,2014,2016,2018,2020],
           label=['2002','2004','2006','2008','2010','2012','2014','2016','2018','2020'])
plt.ylabel('FRP',fontsize=fontsize,labelpad=20)
#plt.ylim(-0.23,0.146)
plt.tick_params(axis='x', length=10, width=2, labelsize=labelsize)
plt.tick_params(axis='y', length=10, width=2, labelsize=labelsize)
plt.legend(fontsize=legend_fontsize,loc='lower right')
plt.title('(b) Africa 2001 - 2020  annual FRP',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax4
plt.axes(ax4)

plt.plot(day2,frp2_daily,color='k',marker='o',label='OBS',linewidth=3,markersize=10)
plt.plot(day2,nn_frp2_daily,color='r',marker='o',label='FFNNs (Cor=0.97)',linewidth=3,markersize=10)
plt.plot(day2,fwi_frp2_daily,color='b',marker='o',label='FWI (Cor=0.93)',linewidth=3,markersize=10)
plt.xticks([15,45,75,105,135,165],
           labels=['J','J','A','S','O','N'])
plt.ylabel('FRP',fontsize=fontsize,labelpad=20)
plt.tick_params(axis='x', length=10, width=2, labelsize=labelsize)
plt.tick_params(axis='y', length=10, width=2, labelsize=labelsize)
plt.legend(fontsize=legend_fontsize)
plt.title('(f) 2016 Africa 6 - 11 daily FRP',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax5
plt.axes(ax5)

plt.plot(year,frp3_annual,color='k',marker='o',label='OBS',linewidth=3,markersize=10)
plt.plot(year,nn_frp3_annual,color='r',marker='o',label='FFNNs (Cor=0.56)',linewidth=3,markersize=10)
plt.plot(year,fwi_frp3_annual,color='b',marker='o',label='FWI (Cor=0.38)',linewidth=3,markersize=10)
plt.xticks([2002,2004,2006,2008,2010,2012,2014,2016,2018,2020],
           label=['2002','2004','2006','2008','2010','2012','2014','2016','2018','2020'])
plt.ylabel('FRP',fontsize=fontsize,labelpad=20)
#plt.ylim(-0.23,0.146)
plt.tick_params(axis='x', length=10, width=2, labelsize=labelsize)
plt.tick_params(axis='y', length=10, width=2, labelsize=labelsize)
plt.legend(fontsize=legend_fontsize,loc='upper right')
plt.title('(c) 2001 - 2020 Siberia annual FRP',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax6
plt.axes(ax6)

plt.plot(day3,frp3_daily,color='k',marker='o',label='OBS',linewidth=3,markersize=10)
plt.plot(day3,nn_frp3_daily,color='r',marker='o',label='FFNNs (Cor=0.76)',linewidth=3,markersize=10)
plt.plot(day3,fwi_frp3_daily,color='b',marker='o',label='FWI (Cor=0.40)',linewidth=3,markersize=10)
plt.xticks([15,45,75,105,135,165],
           labels=['A','M','J','J','A','S'])
plt.ylabel('FRP',fontsize=fontsize,labelpad=20)
plt.tick_params(axis='x', length=10, width=2, labelsize=labelsize)
plt.tick_params(axis='y', length=10, width=2, labelsize=labelsize)
plt.legend(fontsize=legend_fontsize)
plt.title('(g) 2003 Siberia 4 - 9 daily FRP',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax7
plt.axes(ax7)

plt.plot(year,frp4_annual,color='k',marker='o',label='OBS',linewidth=3,markersize=10)
plt.plot(year,nn_frp4_annual,color='r',marker='o',label='FFNNs (Cor=0.87)',linewidth=3,markersize=10)
plt.plot(year,fwi_frp4_annual,color='b',marker='o',label='FWI (Cor=0.77)',linewidth=3,markersize=10)
plt.xticks([2002,2004,2006,2008,2010,2012,2014,2016,2018,2020],
           label=['2002','2004','2006','2008','2010','2012','2014','2016','2018','2020'])
plt.xlabel('Year',fontsize=fontsize,labelpad=40)
plt.ylabel('FRP',fontsize=fontsize,labelpad=20)
#plt.ylim(-0.23,0.146)
plt.tick_params(axis='x', length=10, width=2, labelsize=labelsize)
plt.tick_params(axis='y', length=10, width=2, labelsize=labelsize)
plt.legend(fontsize=legend_fontsize,loc='upper right')
plt.title('(d) 2001 - 2020 South China annual FRP',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# ax8
plt.axes(ax8)

plt.plot(day4,frp4_daily,color='k',marker='o',label='OBS',linewidth=3,markersize=10)
plt.plot(day4,nn_frp4_daily,color='r',marker='o',label='FFNNs (Cor=0.86)',linewidth=3,markersize=10)
plt.plot(day4,fwi_frp4_daily,color='b',marker='o',label='FWI (Cor=0.74)',linewidth=3,markersize=10)
plt.xticks([15,45,75,105,135,165],
           labels=['J','F','M','A','M','J'])
plt.xlabel('Month',fontsize=fontsize,labelpad=40)
plt.ylabel('FRP',fontsize=fontsize,labelpad=20)
plt.tick_params(axis='x', length=10, width=2, labelsize=labelsize)
plt.tick_params(axis='y', length=10, width=2, labelsize=labelsize)
plt.legend(fontsize=legend_fontsize)
plt.title('(h) 2007 South China 1 - 6 daily FRP',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

plt.savefig('/home/geunhyeong/frp/fig/png/github/Fig.3.png',format='png',dpi=300,bbox_inches='tight')