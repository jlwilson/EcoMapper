# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 12:01:08 2018

@author: jlwilson
"""

#import os
#import sys
import numpy as np
#import scipy.ndimage as nd
import matplotlib.pyplot as plt
#import fiona as f
import scipy
import geopandas as gp
#from shapely.geometry import box as shape_box
import gdal
gdal.UseExceptions()
#import ogr
#import osr
import pandas as pd
#from IPython.display import Image
#from IPython.display import Math
#from __future__ import print_function
import statsmodels.api as sm
#import sklearn as skl

pd.set_option('display.height', 2500)
pd.set_option('display.max_rows', 2500)

#from mpl_toolkits.mplot3d import Axes3D
#from scipy.interpolate import griddata
#from pyproj import Proj, transform
from shapely.geometry import Point
#import statsmodels.api as sm
#import statsmodels
import scipy.io as sio
#from datetime import datetime, date, time
 
#%% Import the MATLAB output file
file='20160818_145705_TableRockDam-Subsur_Undul_0to150_IVER2-209_Corr_Proc_QW_Shifted_0to160ft_GIS_DFS.csv'
matlabfile = 'R:\\Projects\\PostProcessed\\TableRockLake\\EMProcessedData\\20160818_145705_TableRockDam-Subsur_Undul_0to150_IVER2-209_Corr_Proc.mat'

#Set the lower and upper bounds to remove data and the first column to import from
upper=0
lower=1900
first_column=5
#%%
mat = sio.loadmat(matlabfile)  # load mat-file
mdata = mat['emdata']  # variable in mat file
mdtype = mdata.dtype  # dtypes of structures are "unsized objects"

ndata = {n: mdata[n][0, 0] for n in mdtype.names}

# df = pd.DataFrame(np.concatenate([ndata[c] for c in columns], axis=1), columns=columns)
columns=['path', 'file', 'depthcorrected', 'omitseg', 'depthrange', 'Latitude', 'Longitude', 'Num_of_Sats', 'GPS_Speed_Knots', 'GPS_True_Heading', 'GPS_Magnetic_Variation', 'HDOP', 'C_Magnetic_Heading', 'C_True_Heading', 'Pitch_Angle', 'Roll_Angle', 'C_Inside_Temp', 'DFS_Depth_ft', 'DTB_Height_ft', 'WCdepth_ft', 'DFS_Depth_m', 'DTB_Height_m', 'WCdepth_m', 'Batt_Percent', 'Power_Watts', 'Watt_Hours', 'Batt_Volts', 'Batt_Ampers', 'Batt_State', 'Time_to_Empty', 'Current_Step', 'Dist_To_Next_m', 'Next_Speed_kn', 'Vehicle_Speed_kn', 'Motor_Speed_CMD', 'Next_Heading', 'Next_Long', 'Next_Lat', 'Next_Depth_ft', 'Depth_Goal_ft', 'Next_Depth_m', 'Depth_Goal_m', 'Vehicle_State', 'Error_State', 'Fin_Pitch_R', 'Fin_Pitch_L', 'Pitch_Goal', 'Fin_Yaw_T', 'Fin_Yaw_B', 'Yaw_Goal', 'Fin_Roll', 'DVL_Depth_ft', 'DVL_Altitude_ft', 'DVL_WCdepth_ft', 'DVL_Depth_m', 'DVL_Altitude_m', 'DVL_WCdepth_m', 'Temp_C', 'SpCond_mS_cm', 'Sal_ppt', 'Depth_feet', 'Depth_meters', 'pH', 'pH_mV', 'Turbidity_NTU', 'Chl_ug_L', 'Chl_RFU', 'BGA_PC_cells_mL', 'BGA_PC_RFU', 'ODOsat_pct', 'ODO_mg_L', 'Rhod_ug_L', 'Datenum', 'LatUncorr', 'LonUncorr', 'Dest_Waypoint', 'Num_Waypoints', 'Distance_From_Waypoint', 'Delta_Distance', 'SalComp_ppt', 'CompPressure_dbar', 'Density_kgm3', 'shifted', 'tcorr', 'created']

#%%
raw = pd.DataFrame(index=np.arange(0,len(ndata['Latitude'])),columns=columns)
for x in columns:
    if len(ndata['{}'.format(x)])==len(ndata['Latitude']):
        raw['{}'.format(x)]=ndata['{}'.format(x)]
    else:
        pass

#%%
# data['Latitude']=ndata['Latitude']
raw['DFS_Depth_ft']=raw['DFS_Depth_m']*3.28084
raw['DTB_Height_ft']=raw['DTB_Height_m']*3.28084

# raw['DFS_Depth_ft']

#%%
geometry = [Point(xy) for xy in zip(raw.Longitude, raw.Latitude)]
crs=None
raw_geo = gp.GeoDataFrame(raw, crs=crs, geometry=geometry)
raw_geo.crs = {'init' :'epsg:4326'}
raw_geo = raw_geo.to_crs({'init': 'epsg:6344'})


#%%
def getXY(pt):
    return (pt.x, pt.y)
centroidseries = raw_geo['geometry'].centroid
x,y = [list(t) for t in zip(*map(getXY, centroidseries))]

raw_geo['X']=x
raw_geo['Y']=y
raw_geo['Z']=raw_geo.DFS_Depth_ft

raw_geo['Distance']=raw_geo['X']

raw_geo['Z']=raw_geo.DFS_Depth_ft

#%%
raw_geo.iloc[1,89]

#%%
#Rotating the data to along with the Y axis for better gridding
X = sm.add_constant(raw_geo['X'])
model_line = sm.OLS(raw_geo['Y'],X, missing='drop')
results = model_line.fit()

# statsmodels.graphics.regressionplots.plot_regress_exog(results,1)

theta=-np.arctan(results.params.iloc[1])
theta

raw_geo['X_rot']=raw_geo.X*np.cos(theta)+raw_geo.Y*np.sin(theta)
raw_geo['Y_rot']=-1*raw_geo.X*np.sin(theta)+raw_geo.Y*np.cos(theta)

for x in np.arange(0,len(ndata['Latitude'])): 
    raw_geo.ix[x,raw_geo.columns.get_loc('Distance')] = 3*(raw_geo.ix[x,raw_geo.columns.get_loc('Y_rot')]-raw_geo.iloc[0,raw_geo.columns.get_loc('Y_rot')])



#%%
    # prj = raw_geo.crs
# dd = raw_geo.plot()
# mplleaflet.display(fig=dd.figure, crs=prj)
    
#%%
raw_cleaned = raw_geo.iloc[50:1650,5:]

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # plt.gca().invert_zaxis()
# Xp = raw_cleaned.X
# Yp = raw_cleaned.Y
# Zp = -raw_cleaned.Z

# # ax.scatter(Xg, Yg, Zg, color='r')

# ax.scatter(Xp, Yp, Zp, c=raw_cleaned.ODO_mg_L)


# # ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
# # cset = ax.contourf(X, Y, Z, zdir='z', offset=-100)

X_d = scipy.signal.resample(raw_cleaned.Distance,len(raw_cleaned.Distance))
Z_d = scipy.signal.resample(raw_cleaned.Z,len(raw_cleaned.Z))
DO_d = scipy.signal.resample(raw_cleaned.ODO_mg_L,len(raw_cleaned.ODO_mg_L))
points_d = np.array(zip(X_d,Z_d))

xi = np.linspace(min(raw_cleaned.Distance), max(raw_cleaned.Distance),num=len(raw_cleaned.Distance))
# yi = np.linspace(min(raw_cleaned.Y_rot), max(raw_cleaned.Y_rot),num=100)
zi = np.linspace(min(raw_cleaned.Z), max(raw_cleaned.Z),num=len(raw_cleaned.Z))

Xg,Zg = np.meshgrid(xi,zi)

DO_2d = scipy.interpolate.griddata(points_d, DO_d, (Xg,Zg), 'linear')

# inter = pd.DataFrame(columns=['X','Z','DO'])
# inter['X']=Xg.flatten()
# inter['Z']=Zg.flatten()
# inter['DO']=DO_3d.flatten()

# inter = pd.DataFrame(columns=['X','Y','Z','DO'])
# inter['X']=inter_rot.X*np.cos(theta)-inter_rot.Y*np.sin(theta)
# inter['Z']=Zg.flatten()
# inter['DO']=DO_2d.flatten()

scatter = pd.DataFrame(columns=['X','Z','DO'])
scatter['X']=scipy.signal.resample(raw_cleaned.Distance,100)
scatter['Z']=scipy.signal.resample(raw_cleaned.Z,100)
scatter['DO']=scipy.signal.resample(raw_cleaned.ODO_mg_L,100)

# scatter_rot = pd.DataFrame(columns=['X','Y','Z','DO'])
# scatter_rot['X']=scipy.signal.resample(raw_cleaned.X_rot,100)
# scatter_rot['Z']=scipy.signal.resample(raw_cleaned.Z,100)
# scatter_rot['DO']=scipy.signal.resample(raw_cleaned.ODO_mg_L,100)


# inter.to_csv('TRL_DO_model_2D.csv', index=False)
# inter_rot.to_csv('TRL_DO_model_rot_2D.csv', index=False)

# scatter.to_csv('TRL_DO_raw_2D.csv', index=False)
# scatter_rot.to_csv('TRL_DO_points_rot_2D.csv', index=False)

# v = np.linspace(0,10,200)

# plt.figure(num=None, figsize=(20, 6), dpi=50, facecolor='w', edgecolor='k')

# do = plt.contourf(Xg,Zg,DO_2d, v, cmap='jet')

# plt.plot(raw_cleaned.Distance, raw_cleaned.Z, '-', linewidth=1.5, color='#A79D95')

# plt.gca().invert_yaxis()
# # plt.axes().set_aspect('equal')
# plt.ylim(np.max(Zg), np.min(Zg))
# plt.xlim(np.min(Xg), np.max(Xg))
# plt.title('Dissolved Oxygen, in mg/L')
# plt.xlabel('Distance, in feet')
# plt.ylabel('Depth, in feet')
# cbar = plt.colorbar(do)

# plt.show()
#%%
sio.savemat('inter_grid.mat',dict(x=X_d,y=Z_d,z=DO_d.T,numx=60, numz=60))

import matlab.engine
eng = matlab.engine.start_matlab()
eng.run_gridfit(nargout=0)



mat = sio.loadmat('gridfit_output.mat')
DO_gf = mat['ZI']
X_gf = mat['XI']
Z_gf = mat['YI']
#%%

v = np.linspace(0,10,200)

plt.figure(num=None, figsize=(21, 6), dpi=60, facecolor='w', edgecolor='k')

do = plt.contourf(X_gf,Z_gf,DO_gf, v, cmap='jet')

plt.plot(raw_cleaned.Distance, raw_cleaned.Z, '-', linewidth=1.5, color='#A79D95')

plt.gca().invert_yaxis()
# plt.axes().set_aspect('equal')
plt.ylim(np.max(Zg), np.min(Zg))
plt.xlim(np.min(Xg), np.max(Xg))
plt.title('Dissolved Oxygen, in mg/L')
plt.xlabel('Distance, in feet')
plt.ylabel('Depth, in feet')
cbar = plt.colorbar(do,ticks=np.arange(0,10))

plt.show()