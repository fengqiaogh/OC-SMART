#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:08:04 2019

@author: Yongzhen Fan
"""
import numpy as np
from pyhdf.SD import SD, SDC
from scipy import interpolate

class Rayleigh(object):
    
    def __init__(self,info=None):
        self.sensor=info.sensor
        self.band=info.band
        nband=len(self.band)
        self.sensor_taur=info.tauray
        ray_path='./auxdata/Rayleigh/' 
        ray_name=ray_path+self.sensor+'/rayleigh_'+self.sensor.lower()+'_'+str(self.band[0])+'_iqu.hdf'
        f=SD(ray_name, SDC.READ)
        self.solztab=f.select('solz')[:]
        self.senztab=f.select('senz')[:]
        self.sigmatab=f.select('sigma')[:] 
        f.end()
        self.raytab=np.zeros((nband,len(self.sigmatab),len(self.solztab),3,len(self.senztab)),dtype='float64')
        self.taur=np.zeros((nband),dtype='float64')
        for i in range(nband):
            ray_name=ray_path+self.sensor+'/rayleigh_'+self.sensor.lower()+'_'+str(self.band[i])+'_iqu.hdf'
            f=SD(ray_name, SDC.READ)
            self.raytab[i,:,:,:,:]=f.select('i_ray')[:,:,:,:]
            self.taur[i]=f.select('taur')[:] 
            f.end()
            
    def corr_ray(self, l1b_solz, l1b_senz, l1b_relaz, l1b_ws):
        print('Compute Rayleigh reflectance ...')
        npix=len(l1b_solz)
        nband=len(self.band)
        nsolz=len(self.solztab)
        nsenz=len(self.senztab)
        nsigma=len(self.sigmatab)
        nmod=3
        sigma=0.0731*np.power(l1b_ws,0.5)
        sigma[sigma>0.4]=0.4
        ray_i=np.zeros(npix)
        l1b_ray=np.zeros([npix,nband])
        tmp=np.zeros([nsigma,nsolz,nsenz])
        for i in range(nband):
            ray_i[:]=0.0
            for j in range(nmod):
                tmp[:,:,:]=self.raytab[i,:,:,j,:]
                func=interpolate.RegularGridInterpolator((self.sigmatab,self.solztab,self.senztab),tmp)
                y=func(np.array([sigma, l1b_solz, l1b_senz]).transpose())
                ray_i[:] += y * np.cos(np.deg2rad(l1b_relaz) * j)
                
            l1b_ray[:,i]=ray_i * np.exp(-self.taur[i])/np.exp(-self.sensor_taur[i])
        
        return l1b_ray
    
    def corr_ray_press(self, l1b_solz, l1b_senz, l1b_pressure):
        p0=1013.25
        npix=len(l1b_solz)
        nband=len(self.band)
        airmass=1/np.cos(np.deg2rad(l1b_solz))+1/np.cos(np.deg2rad(l1b_senz))
        logam=np.log(airmass)
        x1=0.8192-1.2541*self.taur
        x2=0.6542-1.608*self.taur
        x=np.matmul(np.expand_dims(airmass, 1),[self.taur]) * (np.matmul(np.expand_dims(logam,1),[x1])-x2)
        x3=np.zeros([npix,nband])
        for i in range(nband):
            x3[:,i]=l1b_pressure/p0
        ray_fac=(1.0-np.exp(-x * x3))/(1.0-np.exp(-x))
        return ray_fac
            
                