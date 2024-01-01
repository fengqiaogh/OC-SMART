#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Jan 16 11:09:04 2019

@author: Yongzhen Fan
"""

import numpy as np
from os.path import basename
import sys

class sensorinfo(object):
    
    def __init__(self, L1Bname, sensor=None):
        self.sensor=sensor
        self.l1bname=L1Bname
        self.l1bbasename = basename(L1Bname)
        self.sensor_status = 0
        if sensor is None:
            self.autodetect()
        if self.sensor_status == 0:          
            info_fname='./auxdata/sensorinfo/'+self.sensor+'.txt'
            info=np.loadtxt(info_fname,dtype=np.float64)
            self.band=info[:,0].astype(int)
            self.koz=info[:,1]
            self.tauray=info[:,2]
            self.kno2=info[:,3]
            self.vgain=info[:,4]
            self.vgaino=info[:,5] 
            self.vgainc=info[:,6] 

    def autodetect(self):
        
        b = self.l1bbasename
        self.datalevel = None
        self.datasource = None

        if (b.startswith('MER_RR') or b.startswith('MER_FR')) and b.endswith('.N1'):
            self.sensor = 'MERIS'

        elif b.startswith('S3A_OL_1') and b.endswith('.SEN3'):
            self.sensor = 'OLCI'

        elif b.startswith('V') and b.endswith('.nc'):
            self.sensor = 'VIIRS'
            self.datasource = 'OBPG'
            
        elif b.startswith('NPP') and b.endswith('.hdf'):
            self.sensor = 'VIIRS'
            self.datasource = 'NASA DAAC'
            
        elif b.endswith('noaa_ops.h5'):
            self.sensor = 'VIIRS'
            self.datasource = 'NOAA'

        elif b.startswith('A') and b.endswith('.L1B_LAC'):
            self.sensor = 'MODIS-Aqua'
            self.datasource = 'OBPG'
        
        elif b.startswith('MYD021KM') and b.endswith('.hdf'):
            self.sensor = 'MODIS-Aqua'
            self.datasource = 'NASA DAAC'
        
        elif b.startswith('T') and b.endswith('.L1B_LAC'):
            self.sensor = 'MODIS-Terra'
            self.datasource = 'OBPG'
        
        elif b.startswith('MOD021KM') and b.endswith('.hdf'):
            self.sensor = 'MODIS-Terra'
            self.datasource = 'NASA DAAC'

        elif b.startswith('S') and '.L1B' in b:
            self.sensor = 'SeaWiFS'            

        elif b.startswith('COMS_GOCI_L1B') and b.endswith('.he5'):
            self.sensor = 'GOCI'
            
        elif b.startswith('GC1SG1') and b.endswith('.h5'):
            self.sensor = 'SGLI'
            
        elif b.startswith('epic_1b') and b.endswith('.h5'):
            self.sensor = 'EPIC'
            
#        elif b.startswith('LC8') or b.startswith('LC08'):
#            self.sensor = 'OLI'
            
        elif b.startswith('S2A_MSIL1C'):
            self.sensor = 'S2A'
            self.datalevel = 'L1C'
            
        elif b.startswith('S2A_MSIL2A'):
            self.sensor = 'S2A'
            self.datalevel = 'L2A'
            
        elif b.startswith('S2B_MSIL1C'):
            self.sensor = 'S2B'
            self.datalevel = 'L1C'
            
        elif b.startswith('S2B_MSIL2A'):
            self.sensor = 'S2B'
            self.datalevel = 'L2A'
            
        elif b.startswith('FY3D') and b.endswith('1000M_MS.HDF'):
            self.sensor = 'MERSI2'
        
        elif b.startswith('H') and b.endswith('ISS.nc'):
            self.sensor = 'HICO'
            self.datasource = 'OBPG'
            
        else:
            print('\033[1;31;47mWARNING: Unable to detect sensor from file "{}", processing terminated ... \n'.format(b),'\033[m')
            self.sensor_status=1
        