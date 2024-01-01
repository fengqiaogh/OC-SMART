# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 18:21:58 2019

@author: Yongzhen Fan
"""

import numpy as np

class Cloudmask(object):
    
    def __init__(self, sensorinfo=None):
        self.cloud_NIR_threshold=0.027
        self.cloud_ratio=2.5
        self.sensor=sensorinfo.sensor
        if self.sensor == 'EPIC':
            self.cband_id=(1,2,3,4) # 443 551 680 780nm
            self.cloud_NIR_threshold=0.026
            self.cloud_ratio=2.0
        elif self.sensor == 'SGLI':
            self.cband_id=(1,5,6,7) # 412 565 672 865nm
            self.cloud_ratio=5.0
        elif self.sensor == 'VIIRS':
            self.cband_id=(0,3,4,6) # 410 551 671 862nm
            self.cloud_ratio=2.25
        elif self.sensor == 'OLCI':
            self.cband_id=(1,5,8,13) # 412 560 674 865nm
        elif self.sensor == 'MODIS-Aqua':
            self.cband_id=(0,4,5,8) # 412 547 667 869nm
        elif self.sensor == 'MODIS-Terra':
            self.cband_id=(0,4,5,8) # 412 547 667 869nm
        elif self.sensor == 'OLI':
            self.cband_id=(0,2,3,4) # 443 561 665 865nm 
        elif self.sensor == 'S2A':
            self.cband_id=(0,2,3,8) # 443 560 665 865nm  
        elif self.sensor == 'S2B':
            self.cband_id=(0,2,3,8) # 442 559 665 864nm  
        elif self.sensor == 'GOCI':
            self.cband_id=(0,3,4,7) # 412 555 660 865nm 
        elif self.sensor == 'MERSI2':
            self.cband_id=(0,3,4,7) # 412 555 670 865nm 
        elif self.sensor == 'HICO':
            self.cband_id=(1,26,46,71) # 410 553 668 862nm 
        elif self.sensor == 'SeaWiFS':
            self.cband_id=(0,4,5,7) # 412 555 670 865nm 
            
    def run_cloudmask(self, lrc):
        dim=lrc.shape
#        nband=len(self.cband_id)
        cm_lrc=lrc[:,self.cband_id]
        cmask_maxval=np.zeros(dim[0])
        cmask_minval=np.zeros(dim[0])
        cmask_maxval=np.amax(cm_lrc,axis=1)
        cmask_minval=np.amin(cm_lrc,axis=1)
        
        c1=cm_lrc[:,-1]>self.cloud_NIR_threshold
        c2=cmask_maxval/cmask_minval<self.cloud_ratio
        cm = c1 & c2
        
        return cm
    
        
        