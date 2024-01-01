#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Jan 16 11:06:04 2019

@author: Yongzhen Fan
"""

import numpy as np
import h5py
from scipy import interpolate


class AUXData(object):

    def __init__(self):
        self.auxpath='./auxdata/common/'
        #read landmask
#        print('Read land/water mask ... \n')
        resol=2 #1X1 grid
        if resol == 1:
            f=h5py.File(self.auxpath+'landmask_GMT15ARC.nc','r')
            data=f.get('lat')
            dim_lat=np.array(data).shape[0]
            self.landmask_lat=np.zeros(dim_lat,dtype='float64')
            self.landmask_lat=np.array(data)
            data=f.get('lon')
            dim_lon=np.array(data).shape[0]
            self.landmask_lon=np.zeros(dim_lon,dtype='float64')
            self.landmask_lon=np.array(data)
            data=f.get('watermask')
            self.landmask=np.zeros((dim_lat,dim_lon),dtype='int8')
            self.landmask=np.array(data)            
        elif resol ==2:
            f=h5py.File(self.auxpath+'landmask_GMT15ARC_upsample_2X2.h5','r')
            data=f.get('lat')
            dim_lat=np.array(data).shape[0]+2
            self.landmask_lat=np.zeros(dim_lat,dtype='float64')
            self.landmask_lat[0]=-90.0
            self.landmask_lat[1:dim_lat-1]=np.array(data)
            self.landmask_lat[dim_lat-1]=90.0
            data=f.get('lon')
            dim_lon=np.array(data).shape[0]
            self.landmask_lon=np.zeros(dim_lon,dtype='float64')
            self.landmask_lon=np.array(data)
            data=f.get('watermask')
            self.landmask=np.zeros((dim_lat,dim_lon),dtype='int8')
            self.landmask[1:dim_lat-1,:]=np.array(data)
            self.landmask[0,:]=self.landmask[1,:]
            self.landmask[dim_lat-1,:]=self.landmask[dim_lat-2,:]
  
        

    def maskland(self, l1b_lat, l1b_lon, l1b_dxdy):
        print('running land/water mask ... ')
        npixl=len(l1b_lat)
        water_frac=np.zeros(npixl)
        landmask_dxdy=self.landmask_lon[1]-self.landmask_lon[0]
        mask_hsize=int(l1b_dxdy/landmask_dxdy/2)+1 
        if mask_hsize>1:
            for i in range(npixl):
                tp=np.where(self.landmask_lat<l1b_lat[i])[-1][-1]-mask_hsize+1
                bm=tp+2*mask_hsize
                lt=np.where(self.landmask_lon<l1b_lon[i])[-1][-1]-mask_hsize+1
                rt=lt+2*mask_hsize
                water_frac[i]=np.mean(self.landmask[tp:bm,lt:rt])
        else:  
            func=interpolate.RegularGridInterpolator((self.landmask_lat,self.landmask_lon),self.landmask)
            water_frac=func(np.array([l1b_lat,l1b_lon]).transpose())
            
#        m = Basemap(projection='cyl', resolution='h')
#        x,y=m(l1b_lon,l1b_lat)
#        water_frac=np.zeros(npixl)
#        for i in range(npixl):
#            water_frac[i]=m.is_land(x[i],y[i])

        return water_frac
    