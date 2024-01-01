# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 21:34:02 2019

@author: Yongzhen Fan
"""

import numpy as np
from scipy import interpolate

class CHL(object):
    
    def __init__(self, sensorinfo=None):        
        self.sensor=sensorinfo.sensor
        self.band=sensorinfo.band
        self.ci_bid = self.get_ci_bid()
        self.yoc_band=np.array([412,443,490,555])
        self.yoc_coeff=np.array([-1.012,0.342,-2.511,-0.277])
        if self.sensor == 'EPIC':
#            self.ocx_bid=(1,2,3)
#            self.ocx_coefs=np.array([0.813140341748017,-1.02260768677586,-0.13110712807362,0.101545253943379])
            self.ocx_bid=(1,2)            			
            self.ocx_coefs=np.array([-0.0143791002209103,-1.97274091650416,0.266731184697703,0.977030520940235])  
#            self.ocx_coefs=np.array([0.093063,-1.4574,-0.072112,-0.713075,0.474897])              
#            np.array([0.0930629861031035,-1.45740323248020,-0.0721121679237518,-0.713074986570371,0.474897351170013])
        elif self.sensor == 'SGLI':
            self.ocx_bid=(2,3,5)
            self.ocx_coefs=np.array([0.2399,-2.0825,1.6126,-1.0848,-0.2083])
        elif self.sensor == 'GOCI':
            self.ocx_bid=(1,2,3)
            self.ocx_coefs=np.array([0.2515,-2.3798,1.5823,-0.6372,-0.5692])
        elif self.sensor == 'VIIRS':
            self.ocx_bid=(1,2,3)
            self.ocx_coefs=np.array([0.2228,-2.4683,1.5867,-0.4275,-0.7768])
        elif self.sensor == 'OLCI':
            self.ocx_bid=(2,3,4,5)            
            self.ocx_coefs=np.array([0.3255,-2.7677,2.4409,-1.1288,-0.4990])
        elif self.sensor in ['MODIS-Aqua', 'MODIS-Terra']:
            self.ocx_bid=(1,2,4)
            self.ocx_coefs=np.array([0.2424,-2.7423,1.8017,0.0015,-1.2280])
        elif self.sensor == 'SeaWiFS':
            self.ocx_bid=(1,2,4)
            self.ocx_coefs=np.array([0.2515,-2.3798,1.5823,-0.6372,-0.5692])
#            self.ocx_bid=(1,2,3,4)
#            self.ocx_coefs=np.array([0.3272,-2.9940,2.7218,-1.2259,-0.5683])
        elif self.sensor == 'OLI':
            self.ocx_bid=(0,1,2)
            self.ocx_coefs=np.array([0.2412,-2.0546,1.1776,-0.5538,-0.4570])
        elif self.sensor == 'S2A':
            self.ocx_bid=(0,1,2)
            self.ocx_coefs=np.array([0.2521,-2.2146,1.5193,-0.7702,-0.4291])
        elif self.sensor == 'S2B':
            self.ocx_bid=(0,1,2)
            self.ocx_coefs=np.array([0.2521,-2.2146,1.5193,-0.7702,-0.4291])
        elif self.sensor == 'MERSI2':
            self.ocx_bid=(1,2,3)
            self.ocx_coefs=np.array([0.2515,-2.3798,1.5823,-0.6372,-0.5692])
        elif self.sensor == 'HICO':
#            self.ocx_bid=(7,15,19,27)
#            self.ocx_coefs=np.array([0.3255,-2.7677,2.4409,-1.1288,-0.4990])
            self.ocx_bid=(7,15,27)
            self.ocx_coefs=np.array([0.2521,-2.2146,1.5193,-0.7702,-0.4291])
    
    def get_ci_bid(self):
        nband=len(self.band)
        ci_ref_band = np.array([443,555,670])
        ci_bid=np.zeros(3,dtype='int8')
        diff = np.zeros(nband)
        for i in range(3):
            diff[:] = np.abs(self.band - ci_ref_band[i])
            minval = np.min(diff)
            ci_bid[i] = np.where(diff == minval)[0][0]
        return ci_bid
    
    def get_chl_ocx(self,rrs): 
        print('Retrieving CHLa (OCx) ...')
        nband=len(self.ocx_bid)
        chl=np.zeros(rrs.shape[0])+np.nan
        rrs_ocx=rrs[:,self.ocx_bid]
#        if self.sensor =='EPIC':
#            ratio = rrs_ocx[:,0]/rrs_ocx[:,1]*(rrs_ocx[:,1]/rrs_ocx[:,2])
#        else:
        if nband==2:
            ratio=rrs_ocx[:,0]/rrs_ocx[:,1]
        elif nband==3:
            ratio=np.amax(rrs_ocx[:,0:2],axis=1)/rrs_ocx[:,2]
        elif nband==4:
            ratio=np.amax(rrs_ocx[:,0:3],axis=1)/rrs_ocx[:,3]
        
        npw=len(self.ocx_coefs)
        chl_ocx=np.zeros(rrs.shape[0])
        for i in np.arange(npw):
            chl_ocx = chl_ocx + self.ocx_coefs[i]*np.log10(ratio)**i  
             
        chl_ocx=10**chl_ocx
        chl_ocx[chl_ocx<0.001] = 0.001
        chl_ocx[chl_ocx>1000.] = 1000.        
        chl = chl_ocx  
        return chl
    
    def get_chl_oci(self,rrs): 
        print('Retrieving CHLa (OCi) ...')
        nband=len(self.ocx_bid)
        chl=np.zeros(rrs.shape[0])+np.nan
        rrs_ocx=rrs[:,self.ocx_bid]
        ci_coef = np.array([-0.4909, 191.6590])
        rrs_ci=rrs[:,self.ci_bid]
        rrs_ci[:,1] = self.convt_rrs555(rrs_ci[:,1])
        
        ci = rrs_ci[:,1] - (rrs_ci[:,0] + (555-443)/(670-443)*(rrs_ci[:,2] - rrs_ci[:,0]))
        ci[ci>0.0]=0.0
        chl_ci = 10**(ci_coef[0]+ci*ci_coef[1])
        chl_ci[chl_ci<0.001]=0.001
        chl_ci[chl_ci>1000.]=1000.
        
        idx1 = chl_ci < 0.15
        idx2 = chl_ci >0.2
        idx3 = np.where((chl_ci>=0.15) & (chl_ci<=0.2))[0]
        chl[idx1] = chl_ci[idx1] # use ci algorithm when chl<0.15
        
#        if self.sensor =='EPIC':
#            ratio = rrs_ocx[:,0]/rrs_ocx[:,1]*(rrs_ocx[:,1]/rrs_ocx[:,2])
#        else:
        if nband==2:
            ratio=rrs_ocx[:,0]/rrs_ocx[:,1]
        elif nband==3:
            ratio=np.amax(rrs_ocx[:,0:2],axis=1)/rrs_ocx[:,2]
        elif nband==4:
            ratio=np.amax(rrs_ocx[:,0:3],axis=1)/rrs_ocx[:,3]
            
        npw=len(self.ocx_coefs)
        chl_ocx=np.zeros(rrs.shape[0])
        for i in np.arange(npw):
            chl_ocx = chl_ocx + self.ocx_coefs[i]*np.log10(ratio)**i 
                 
        chl_ocx=10**chl_ocx
        chl_ocx[chl_ocx<0.001] = 0.001
        chl_ocx[chl_ocx>1000.] = 1000.        
        chl[idx2] = chl_ocx[idx2] # use ocx algorithm when chl>0.2
        
        h=(chl_ci[idx3]-0.15)/(0.2-0.15)
        chl[idx3]=chl_ci[idx3]*(1-h)+chl_ocx[idx3]*h        
        return chl
    
    def get_chl_yoc(self,rrs): 
        print('Retrieving CHLa (YOC) ...')
        nrrs=rrs.shape[1]
        f=interpolate.interp1d(self.band[0:nrrs],rrs,kind='linear',fill_value='extrapolate')
        rrs_yoc=f(self.yoc_band)
        rrs_yoc[rrs_yoc[:,0]<0,0]=0.0001        
        ratio=np.log10((rrs_yoc[:,1]/rrs_yoc[:,3])*(rrs_yoc[:,0]/rrs_yoc[:,2])**self.yoc_coeff[0])        
        chl  = self.yoc_coeff[1] \
             + self.yoc_coeff[2] * ratio \
             + self.yoc_coeff[3] * ratio**2             
        chl = 10**chl
        return chl
    
    def convt_rrs555(self, rrs):
        ci_band2 = self.band[self.ci_bid[1]]
        if np.abs(ci_band2-555)>2:
            if np.abs(ci_band2-547)<=2:
                sw = 0.001723
                a1 = 0.986
                b1 = 0.081495
                a2 = 1.031
                b2 = 0.000216
            elif np.abs(ci_band2-550)<=2:
                sw = 0.001597
                a1 = 0.988
                b1 = 0.062195
                a2 = 1.014
                b2 = 0.000128
            elif np.abs(ci_band2-560)<=2:
                sw = 0.001148
                a1 = 1.023
                b1 = -0.103624
                a2 = 0.979
                b2 = -0.000121
            elif np.abs(ci_band2-565)<=2:
                sw = 0.000891
                a1 = 1.039
                b1 = -0.183044
                a2 = 0.971
                b2 = -0.000170
        else:
            sw = 0.000891
            a1 = 1.0
            b1 = 0.0
            a2 = 1.0
            b2 = 0.0
            
        rrs555 = np.zeros(rrs.shape)
        rrs555[rrs<sw] = 10**(a1*np.log10(rrs[rrs<sw])+b1)
        rrs555[rrs>=sw] = a2*rrs[rrs>=sw]+b2
        return rrs555
                    
class TSM(object):
    
    def __init__(self, sensorinfo=None): 
        self.sensor=sensorinfo.sensor
        self.band=sensorinfo.band
        self.nechad_band = 710 # single band algorithm
        self.nechad_coeff = np.array([561.94, 1.23, 0.1892])
        if self.sensor in ['OLI', 'SGLI']:
            self.nechad_band = 665 # single band algorithm
            self.nechad_coeff = np.array([355.85, 1.74, 0.1728])
        self.yoc_band=np.array([490,555,670])
        self.yoc_coeff=np.array([0.649,25.623,-0.646])

    def get_tsm_nechad(self, rrs): # DO NOT USE, algorithm is incorrect.
        print('Retrieving TSM (Rrs710) ...')
        nrrs=rrs.shape[1]
        f=interpolate.interp1d(self.band[0:nrrs],rrs)
        rrs_tsm=f(self.nechad_band)*np.pi
        tsm = self.nechad_coeff[0]*rrs_tsm/(1-rrs_tsm/self.nechad_coeff[2])+self.nechad_coeff[1]             
        return tsm
    
    def get_tsm_yoc(self, rrs):
        print('Retrieving TSM (YOC) ...')
        nrrs=rrs.shape[1]
        f=interpolate.interp1d(self.band[0:nrrs],rrs,kind='linear',fill_value='extrapolate')
        self.rrs_tsm=f(self.yoc_band)
        tsm = self.yoc_coeff[0] \
            + self.yoc_coeff[1] * (self.rrs_tsm[:,1]+self.rrs_tsm[:,2]) \
            + self.yoc_coeff[2] * (self.rrs_tsm[:,0]/self.rrs_tsm[:,1])
        tsm[tsm>4.0]=4.0
        tsm[tsm<-4.0]=-4.0
        tsm = 10**tsm
        return tsm

        
class CDOM(object):
    
    def __init__(self, sensorinfo=None): 
        self.sensor=sensorinfo.sensor
        self.band=sensorinfo.band
        self.tassan_band = np.array([443,490,555]) 
        self.tassan_coeff = np.array([0.059,-0.990,-1.781,-2.180])
        self.carder_band = np.array([412,443,555])
        self.carder_coeff = np.array([-1.138,-0.769,-1.082,-0.368,0.727])
        
    
    def get_cdom_tassan(self, rrs):
        print('Retrieving CDOM (Tassan) ...')
        nrrs=rrs.shape[1]        
        f=interpolate.interp1d(self.band[0:nrrs],rrs)
        rrs_cdom=f(self.tassan_band)
        ratio=np.log10(rrs_cdom[:,1]/rrs_cdom[:,2]*rrs_cdom[:,0]**self.tassan_coeff[0])
        cdom = self.tassan_coeff[1] \
             + self.tassan_coeff[2] * ratio \
             + self.tassan_coeff[3] * ratio**2             
        cdom = 10**cdom
        return cdom 
    def get_cdom_carder(self, rrs):
        print('Retrieving CDOM (Carder) ...')
        nrrs=rrs.shape[1]        
        f=interpolate.interp1d(self.band[0:nrrs],rrs,kind='linear',fill_value='extrapolate')
        rrs_cdom=f(self.carder_band)        
        rrs_cdom[rrs_cdom[:,0]<0,0]=0.0001
        ratio1 = np.log10(rrs_cdom[:,0]/rrs_cdom[:,2])
        ratio2 = np.log10(rrs_cdom[:,1]/rrs_cdom[:,2])
        cdom = self.carder_coeff[0] \
             + self.carder_coeff[1] * ratio1 \
             + self.carder_coeff[2] * ratio1**2 \
             + self.carder_coeff[3] * ratio2 \
             + self.carder_coeff[4] * ratio2**2 
        cdom = 1.5*10**cdom
        return cdom  
        
        