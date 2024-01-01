# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 19:40:49 2019

@author: Yongzhen Fan
"""

import numpy as np
import h5py

class MLNN(object):

    def __init__(self, sensorinfo=None):
        print('Loading Multilayer Neural Networks (MLNNs) ... ')
        self.sensor = sensorinfo.sensor
        self.path = './auxdata/MLNN_nets/'
        self.band = sensorinfo.band
        self.vgaino = sensorinfo.vgaino
        self.vgainc = sensorinfo.vgainc
        #read aann network
#        aann_fname=self.path+self.sensor+'/'+self.sensor+'_Lt_aaNN.h5'
        aann_fname=self.path+self.sensor+'/'+self.sensor+'_Lrc_aaNN.h5'
        f = h5py.File(aann_fname, 'r')
        self.aann_layers=np.array(f['Layers'])
        self.aann_nlayers=len(self.aann_layers)
        self.aann_norm_in=np.array(f['Norm_in'])
        self.aann_norm_out=np.array(f['Norm_out'])
        self.aann_weights=[]
        self.aann_bias=[]
        for i in range(self.aann_nlayers-1):
            self.aann_weights.append(np.array(f['Weights/Layer'+str(i+1)]))
            self.aann_bias.append(np.array(f['Bias/Layer'+str(i+1)]))
            
        #read aodnn network
#        aodnn_fname=self.path+self.sensor+'/'+self.sensor+'_Lt_aodNN.h5'
        aodnn_fname=self.path+self.sensor+'/'+self.sensor+'_Lrc_aodNN.h5'
        f = h5py.File(aodnn_fname, 'r')
        self.aodnn_layers=np.array(f['Layers'])
        self.aodnn_nlayers=len(self.aodnn_layers)
        self.aodnn_norm_in=np.array(f['Norm_in'])
        self.aodnn_norm_out=np.array(f['Norm_out'])
        self.aodnn_weights=[]
        self.aodnn_bias=[]
        for i in range(self.aodnn_nlayers-1):
            self.aodnn_weights.append(np.array(f['Weights/Layer'+str(i+1)]))
            self.aodnn_bias.append(np.array(f['Bias/Layer'+str(i+1)]))	
            
        #read rrsnn network
#        rrsnn_fname=self.path+self.sensor+'/'+self.sensor+'_Lt_rrsNN.h5'
        rrsnn_fname=self.path+self.sensor+'/'+self.sensor+'_Lrc_rrsNN.h5'
        f = h5py.File(rrsnn_fname, 'r')
        self.rrsnn_layers=np.array(f['Layers'])
        self.rrsnn_nlayers=len(self.rrsnn_layers)
        self.rrsnn_norm_in=np.array(f['Norm_in'])
        self.rrsnn_norm_out=np.array(f['Norm_out'])
        self.rrsnn_weights=[]
        self.rrsnn_bias=[]
        for i in range(self.rrsnn_nlayers-1):
            self.rrsnn_weights.append(np.array(f['Weights/Layer'+str(i+1)]))
            self.rrsnn_bias.append(np.array(f['Bias/Layer'+str(i+1)]))
            
        #read iopnn network
#        iopnn_fname=self.path+self.sensor+'/'+self.sensor+'_OCIOPNN.h5'
#        f = h5py.File(iopnn_fname, 'r')
#        self.iopnn_layers=np.array(f['Layers'])
#        self.iopnn_nlayers=len(self.iopnn_layers)
#        self.iopnn_norm_in=np.array(f['Norm_in'])
#        self.iopnn_norm_out=np.array(f['Norm_out'])
#        self.iopnn_weights=[]
#        self.iopnn_bias=[]
#        for i in range(self.iopnn_nlayers-1):
#            self.iopnn_weights.append(np.array(f['Weights/Layer'+str(i+1)]))
#            self.iopnn_bias.append(np.array(f['Bias/Layer'+str(i+1)])) 
#        
#        #read ocnn network
#        ocnn_fname=self.path+self.sensor+'/'+self.sensor+'_ocNN.h5'
#        f = h5py.File(ocnn_fname, 'r')
#        self.ocnn_layers=np.array(f['Layers'])
#        self.ocnn_nlayers=len(self.ocnn_layers)
#        self.ocnn_norm_in=np.array(f['Norm_in'])
#        self.ocnn_norm_out=np.array(f['Norm_out'])
#        self.ocnn_weights=[]
#        self.ocnn_bias=[]
#        for i in range(self.ocnn_nlayers-1):
#            self.ocnn_weights.append(np.array(f['Weights/Layer'+str(i+1)]))
#            self.ocnn_bias.append(np.array(f['Bias/Layer'+str(i+1)])) 
#            
        #read aphnn network
        aphnn_fname=self.path+self.sensor+'/'+self.sensor+'_aphNN.h5'
        f = h5py.File(aphnn_fname, 'r')
        self.aphnn_layers=np.array(f['Layers'])
        self.aphnn_nlayers=len(self.aphnn_layers)
        self.aphnn_norm_in=np.array(f['Norm_in'])
        self.aphnn_norm_out=np.array(f['Norm_out'])
        self.aphnn_weights=[]
        self.aphnn_bias=[]
        for i in range(self.aphnn_nlayers-1):
            self.aphnn_weights.append(np.array(f['Weights/Layer'+str(i+1)]))
            self.aphnn_bias.append(np.array(f['Bias/Layer'+str(i+1)])) 
            
        #read adgnn network    
        adgnn_fname=self.path+self.sensor+'/'+self.sensor+'_adgNN.h5'
        f = h5py.File(adgnn_fname, 'r')
        self.adgnn_layers=np.array(f['Layers'])
        self.adgnn_nlayers=len(self.adgnn_layers)
        self.adgnn_norm_in=np.array(f['Norm_in'])
        self.adgnn_norm_out=np.array(f['Norm_out'])
        self.adgnn_weights=[]
        self.adgnn_bias=[]
        for i in range(self.adgnn_nlayers-1):
            self.adgnn_weights.append(np.array(f['Weights/Layer'+str(i+1)]))
            self.adgnn_bias.append(np.array(f['Bias/Layer'+str(i+1)])) 
            
        #read bbpnn network    
        bbpnn_fname=self.path+self.sensor+'/'+self.sensor+'_bbpNN.h5'
        f = h5py.File(bbpnn_fname, 'r')
        self.bbpnn_layers=np.array(f['Layers'])
        self.bbpnn_nlayers=len(self.bbpnn_layers)
        self.bbpnn_norm_in=np.array(f['Norm_in'])
        self.bbpnn_norm_out=np.array(f['Norm_out'])
        self.bbpnn_weights=[]
        self.bbpnn_bias=[]
        for i in range(self.bbpnn_nlayers-1):
            self.bbpnn_weights.append(np.array(f['Weights/Layer'+str(i+1)]))
            self.bbpnn_bias.append(np.array(f['Bias/Layer'+str(i+1)])) 
##            
        #read apnn network
        apnn_fname=self.path+self.sensor+'/'+self.sensor+'_apNN.h5'
        f = h5py.File(apnn_fname, 'r')
        self.apnn_layers=np.array(f['Layers'])
        self.apnn_nlayers=len(self.apnn_layers)
        self.apnn_norm_in=np.array(f['Norm_in'])
        self.apnn_norm_out=np.array(f['Norm_out'])
        self.apnn_weights=[]
        self.apnn_bias=[]
        for i in range(self.apnn_nlayers-1):
            self.apnn_weights.append(np.array(f['Weights/Layer'+str(i+1)]))
            self.apnn_bias.append(np.array(f['Bias/Layer'+str(i+1)])) 
            
        #read bpnn network    
        bpnn_fname=self.path+self.sensor+'/'+self.sensor+'_bpNN.h5'
        f = h5py.File(bpnn_fname, 'r')
        self.bpnn_layers=np.array(f['Layers'])
        self.bpnn_nlayers=len(self.bpnn_layers)
        self.bpnn_norm_in=np.array(f['Norm_in'])
        self.bpnn_norm_out=np.array(f['Norm_out'])
        self.bpnn_weights=[]
        self.bpnn_bias=[]
        for i in range(self.bpnn_nlayers-1):
            self.bpnn_weights.append(np.array(f['Weights/Layer'+str(i+1)]))
            self.bpnn_bias.append(np.array(f['Bias/Layer'+str(i+1)]))  
   
    def compute_aann(self, solz, senz, relaz, lrc, rh):
        print('Checking Rayleigh corrected radiance (Lrc) spectral shape ... ')
        #rebuild aaNN and compute data
        ncase=len(solz) 
        nlayers=len(self.aann_layers)
        aainput=np.zeros((ncase,int(self.aann_layers[0])))
        aainput[:,0]=np.cos(np.deg2rad(solz))
        aainput[:,1]=np.cos(np.deg2rad(senz))
        aainput[:,2]=np.cos(np.deg2rad(relaz))
        aainput[:,range(3,int(self.aann_layers[0])-1)]=np.log10(lrc)
        aainput[:,int(self.aann_layers[0])-1]=np.log10(rh)
        #normalize
        for i in range(int(self.aann_layers[0])):
            aainput[:,i]=2*(aainput[:,i]-self.aann_norm_in[i,0])/(self.aann_norm_in[i,1]-self.aann_norm_in[i,0])-1
        #compute using aaNN
        for i in np.arange(nlayers-1):
            if i == 0:
                lastlayer = np.tanh(np.matmul(self.aann_weights[i],aainput.transpose())+self.aann_bias[i])
            elif i < nlayers-2:
                currentlayer = np.tanh(np.matmul(self.aann_weights[i],lastlayer)+self.aann_bias[i])
                lastlayer = currentlayer
            else:
                currentlayer = np.matmul(self.aann_weights[i],lastlayer)+self.aann_bias[i]
        aann_output = currentlayer.transpose()

        #denormlize
        for i in range(int(self.aann_layers[-1])):
            aann_output[:,i]=(aann_output[:,i]+1)/2*(self.aann_norm_out[i,1]-self.aann_norm_out[i,0])+self.aann_norm_out[i,0]
        aann_output=np.power(10, aann_output)
        #set flags for out of scope 
        #pe=(aann_output-lrc)/lrc*100.0
        ratio=np.mean(np.absolute(aann_output/lrc-1),axis=1)
        oos=np.zeros(ncase,dtype='bool')
        oos = ratio >0.07
        return oos,aann_output
        
    def compute_aodnn(self, solz, senz, relaz, lrc, rh):
        print('Retrieving aerosol optical depths (AODs) ... ')
        #rebuild aodNN and compute data
        ncase=len(solz)
        nlayers=len(self.aodnn_layers)
        aodinput=np.zeros((ncase,int(self.aodnn_layers[0])))
        aodinput[:,0]=np.cos(np.deg2rad(solz))
        aodinput[:,1]=np.cos(np.deg2rad(senz))
        aodinput[:,2]=np.cos(np.deg2rad(relaz))
        aodinput[:,range(3,int(self.aodnn_layers[0])-1)]=np.log10(lrc)
        aodinput[:,int(self.aodnn_layers[0])-1]=np.log10(rh)
        #normalize
        for i in range(int(self.aodnn_layers[0])):
            aodinput[:,i]=2*(aodinput[:,i]-self.aodnn_norm_in[i,0])/(self.aodnn_norm_in[i,1]-self.aodnn_norm_in[i,0])-1
        #compute using aodNN
        for i in np.arange(nlayers-1):
            if i == 0:
                lastlayer = np.tanh(np.matmul(self.aodnn_weights[i],aodinput.transpose())+self.aodnn_bias[i])
            elif i < nlayers-2:
                currentlayer = np.tanh(np.matmul(self.aodnn_weights[i],lastlayer)+self.aodnn_bias[i])
                lastlayer = currentlayer
            else:
                currentlayer = np.matmul(self.aodnn_weights[i],lastlayer)+self.aodnn_bias[i]
        aodnn_output = currentlayer.transpose()
        for i in range(int(self.aodnn_layers[-1])):
            aodnn_output[:,i]=(aodnn_output[:,i]+1)/2*(self.aodnn_norm_out[i,1]-self.aodnn_norm_out[i,0])+self.aodnn_norm_out[i,0]
        aod=np.power(10, aodnn_output)
        return aod
		
    def compute_rrsnn(self, solz, senz, relaz, lrc):
        print('Retrieving remote sensing reflectance (Rrs) ... ')
        #rebuild rrsNN and compute data
        ncase=len(solz)
        nlayers=len(self.rrsnn_layers)
        rrsinput=np.zeros((ncase,int(self.rrsnn_layers[0])))
        rrsinput[:,0]=np.cos(np.deg2rad(solz))
        rrsinput[:,1]=np.cos(np.deg2rad(senz))
        rrsinput[:,2]=np.cos(np.deg2rad(relaz))
        rrsinput[:,range(3,int(self.rrsnn_layers[0]))]=np.log10(lrc)
        #normalize
        for i in range(int(self.rrsnn_layers[0])):
            rrsinput[:,i]=2*(rrsinput[:,i]-self.rrsnn_norm_in[i,0])/(self.rrsnn_norm_in[i,1]-self.rrsnn_norm_in[i,0])-1
        #compute using rrsNN
        for i in np.arange(nlayers-1):
            if i == 0:
                lastlayer = np.tanh(np.matmul(self.rrsnn_weights[i],rrsinput.transpose())+self.rrsnn_bias[i])
            elif i < nlayers-2:
                currentlayer = np.tanh(np.matmul(self.rrsnn_weights[i],lastlayer)+self.rrsnn_bias[i])
                lastlayer = currentlayer
            else:
                currentlayer = np.matmul(self.rrsnn_weights[i],lastlayer)+self.rrsnn_bias[i]
        rrsnn_output = currentlayer.transpose()

        #denormlize
        for i in range(int(self.rrsnn_layers[-1])):
            rrsnn_output[:,i]=(rrsnn_output[:,i]+1)/2*(self.rrsnn_norm_out[i,1]-self.rrsnn_norm_out[i,0])+self.rrsnn_norm_out[i,0]
        rrs=np.power(10, rrsnn_output)
        
        nratbands=np.sum(self.band<700)
        # calibration for clear water
        if self.sensor == 'EPIC':
            self.oorat=30
        else:
            self.oorat=70
        ratio = np.amax(rrs[:,0:nratbands],axis=1) / np.amin(rrs[:,0:nratbands],axis=1)
        maxidx = np.argmax(rrs[:,0:nratbands],axis=1)
        if self.sensor == 'EPIC':
            idxo = np.where((ratio>=self.oorat) & (maxidx<=1))[0]
            idxc = np.where((ratio<self.oorat) | (maxidx>1))[0] 
        else:
            idxo = np.where((ratio>=self.oorat) & (maxidx==0))[0] 
            idxc = np.where((ratio<self.oorat) | (maxidx>0))[0] 
        
        rrs[idxo,:] = rrs[idxo,:]*self.vgaino[0:int(self.rrsnn_layers[-1])]       
        rrs[idxc,:] = rrs[idxc,:]*self.vgainc[0:int(self.rrsnn_layers[-1])]
        return rrs
    
    def compute_iopnn(self, rrs):
        print('Retrieving ocean IOPs (aph443, adg443 and bbp443) ... ')
        #rebuild iopNN and compute data
        ncase=len(rrs)
        nlayers=len(self.iopnn_layers)
        iopinput=np.zeros((ncase,int(self.iopnn_layers[0])))
        iopinput[:,:]=np.log10(rrs)
        #normalize
        for i in range(int(self.iopnn_layers[0])):
            iopinput[:,i]=2*(iopinput[:,i]-self.iopnn_norm_in[i,0])/(self.iopnn_norm_in[i,1]-self.iopnn_norm_in[i,0])-1
        
        for i in np.arange(nlayers-1):
            if i == 0:
                lastlayer = np.tanh(np.matmul(self.iopnn_weights[i],iopinput.transpose())+self.iopnn_bias[i])
            elif i < nlayers-2:
                currentlayer = np.tanh(np.matmul(self.iopnn_weights[i],lastlayer)+self.iopnn_bias[i])
                lastlayer = currentlayer
            else:
                currentlayer = np.matmul(self.iopnn_weights[i],lastlayer)+self.iopnn_bias[i]
        iopnn_output = currentlayer.transpose()
        #denormlize
        for i in range(int(self.iopnn_layers[-1])):
            iopnn_output[:,i]=(iopnn_output[:,i]+1)/2*(self.iopnn_norm_out[i,1]-self.iopnn_norm_out[i,0])+self.iopnn_norm_out[i,0]
        iop=np.power(10, iopnn_output)
        return iop
    
    def compute_ocnn(self, rrs):
        print('Retrieving CHLa, CDOM and TSM (NN) ... ')
        #rebuild ocNN and compute data
        ncase=len(rrs)
        nlayers=len(self.ocnn_layers)
        ocinput=np.zeros((ncase,int(self.ocnn_layers[0])))
        ocinput[:,:]=np.log10(rrs)
        #normalize
        for i in range(int(self.ocnn_layers[0])):
            ocinput[:,i]=2*(ocinput[:,i]-self.ocnn_norm_in[i,0])/(self.ocnn_norm_in[i,1]-self.ocnn_norm_in[i,0])-1
        
        for i in np.arange(nlayers-1):
            if i == 0:
                lastlayer = np.tanh(np.matmul(self.ocnn_weights[i],ocinput.transpose())+self.ocnn_bias[i])
            elif i < nlayers-2:
                currentlayer = np.tanh(np.matmul(self.ocnn_weights[i],lastlayer)+self.ocnn_bias[i])
                lastlayer = currentlayer
            else:
                currentlayer = np.matmul(self.ocnn_weights[i],lastlayer)+self.ocnn_bias[i]
        ocnn_output = currentlayer.transpose()
        #denormlize
        for i in range(int(self.ocnn_layers[-1])):
            ocnn_output[:,i]=(ocnn_output[:,i]+1)/2*(self.ocnn_norm_out[i,1]-self.ocnn_norm_out[i,0])+self.ocnn_norm_out[i,0]
        oc=np.power(10, ocnn_output)
        return oc
    
    def compute_aphnn(self, rrs):
        print('Retrieving aph (NN) ... ')
        #rebuild aphNN and compute data
        ncase=len(rrs)
        nlayers=len(self.aphnn_layers)
        aphinput=np.zeros((ncase,int(self.aphnn_layers[0])))
        aphinput[:,:]=np.log10(rrs)
        #normalize
        for i in range(int(self.aphnn_layers[0])):
            aphinput[:,i]=2*(aphinput[:,i]-self.aphnn_norm_in[i,0])/(self.aphnn_norm_in[i,1]-self.aphnn_norm_in[i,0])-1
        
        for i in np.arange(nlayers-1):
            if i == 0:
                lastlayer = np.tanh(np.matmul(self.aphnn_weights[i],aphinput.transpose())+self.aphnn_bias[i])
            elif i < nlayers-2:
                currentlayer = np.tanh(np.matmul(self.aphnn_weights[i],lastlayer)+self.aphnn_bias[i])
                lastlayer = currentlayer
            else:
                currentlayer = np.matmul(self.aphnn_weights[i],lastlayer)+self.aphnn_bias[i]
        aphnn_output = currentlayer.transpose()
        #denormlize
        for i in range(int(self.aphnn_layers[-1])):
            aphnn_output[:,i]=(aphnn_output[:,i]+1)/2*(self.aphnn_norm_out[i,1]-self.aphnn_norm_out[i,0])+self.aphnn_norm_out[i,0]
        aph=np.power(10, aphnn_output)
        return aph
    
    def compute_adgnn(self, rrs):
        print('Retrieving adg (NN) ... ')
        #rebuild adgNN and compute data
        ncase=len(rrs)
        nlayers=len(self.adgnn_layers)
        adginput=np.zeros((ncase,int(self.adgnn_layers[0])))
        adginput[:,:]=np.log10(rrs)
        #normalize
        for i in range(int(self.adgnn_layers[0])):
            adginput[:,i]=2*(adginput[:,i]-self.adgnn_norm_in[i,0])/(self.adgnn_norm_in[i,1]-self.adgnn_norm_in[i,0])-1
        
        for i in np.arange(nlayers-1):
            if i == 0:
                lastlayer = np.tanh(np.matmul(self.adgnn_weights[i],adginput.transpose())+self.adgnn_bias[i])
            elif i < nlayers-2:
                currentlayer = np.tanh(np.matmul(self.adgnn_weights[i],lastlayer)+self.adgnn_bias[i])
                lastlayer = currentlayer
            else:
                currentlayer = np.matmul(self.adgnn_weights[i],lastlayer)+self.adgnn_bias[i]
        adgnn_output = currentlayer.transpose()
        #denormlize
        for i in range(int(self.adgnn_layers[-1])):
            adgnn_output[:,i]=(adgnn_output[:,i]+1)/2*(self.adgnn_norm_out[i,1]-self.adgnn_norm_out[i,0])+self.adgnn_norm_out[i,0]
        adg=np.power(10, adgnn_output)
        return adg
    
    def compute_bbpnn(self, rrs):
        print('Retrieving bbp (NN) ... ')
        #rebuild bbpNN and compute data
        ncase=len(rrs)
        nlayers=len(self.bbpnn_layers)
        bbpinput=np.zeros((ncase,int(self.bbpnn_layers[0])))
        bbpinput[:,:]=np.log10(rrs)
        #normalize
        for i in range(int(self.bbpnn_layers[0])):
            bbpinput[:,i]=2*(bbpinput[:,i]-self.bbpnn_norm_in[i,0])/(self.bbpnn_norm_in[i,1]-self.bbpnn_norm_in[i,0])-1
        
        for i in np.arange(nlayers-1):
            if i == 0:
                lastlayer = np.tanh(np.matmul(self.bbpnn_weights[i],bbpinput.transpose())+self.bbpnn_bias[i])
            elif i < nlayers-2:
                currentlayer = np.tanh(np.matmul(self.bbpnn_weights[i],lastlayer)+self.bbpnn_bias[i])
                lastlayer = currentlayer
            else:
                currentlayer = np.matmul(self.bbpnn_weights[i],lastlayer)+self.bbpnn_bias[i]
        bbpnn_output = currentlayer.transpose()
        #denormlize
        for i in range(int(self.bbpnn_layers[-1])):
            bbpnn_output[:,i]=(bbpnn_output[:,i]+1)/2*(self.bbpnn_norm_out[i,1]-self.bbpnn_norm_out[i,0])+self.bbpnn_norm_out[i,0]
        bbp=np.power(10, bbpnn_output)
        return bbp
    
    def compute_apnn(self, rrs):
        print('Retrieving ap (NN) ... ')
        #rebuild rrsNN and compute data
        ncase=len(rrs)
        nlayers=len(self.apnn_layers)
        apinput=np.zeros((ncase,int(self.apnn_layers[0])))
        apinput[:,:]=np.log10(rrs)
        #normalize
        for i in range(int(self.apnn_layers[0])):
            apinput[:,i]=2*(apinput[:,i]-self.apnn_norm_in[i,0])/(self.apnn_norm_in[i,1]-self.apnn_norm_in[i,0])-1
        #compute using rrsNN
        for i in np.arange(nlayers-1):
            if i == 0:
                lastlayer = np.tanh(np.matmul(self.apnn_weights[i],apinput.transpose())+self.apnn_bias[i])
            elif i < nlayers-2:
                currentlayer = np.tanh(np.matmul(self.apnn_weights[i],lastlayer)+self.apnn_bias[i])
                lastlayer = currentlayer
            else:
                currentlayer = np.matmul(self.apnn_weights[i],lastlayer)+self.apnn_bias[i]
        apnn_output = currentlayer.transpose()
        #denormlize
        for i in range(int(self.apnn_layers[-1])):
            apnn_output[:,i]=(apnn_output[:,i]+1)/2*(self.apnn_norm_out[i,1]-self.apnn_norm_out[i,0])+self.apnn_norm_out[i,0]
        ap=np.power(10, apnn_output)
        return ap
    
    def compute_bpnn(self, rrs):
        print('Retrieving bp (NN) ... ')
        #rebuild rrsNN and compute data
        ncase=len(rrs)
        nlayers=len(self.bpnn_layers)
        bpinput=np.zeros((ncase,int(self.bpnn_layers[0])))
        bpinput[:,:]=np.log10(rrs)
        #normalize
        for i in range(int(self.bpnn_layers[0])):
            bpinput[:,i]=2*(bpinput[:,i]-self.bpnn_norm_in[i,0])/(self.bpnn_norm_in[i,1]-self.bpnn_norm_in[i,0])-1
        #compute using rrsNN
        for i in np.arange(nlayers-1):
            if i == 0:
                lastlayer = np.tanh(np.matmul(self.bpnn_weights[i],bpinput.transpose())+self.bpnn_bias[i])
            elif i < nlayers-2:
                currentlayer = np.tanh(np.matmul(self.bpnn_weights[i],lastlayer)+self.bpnn_bias[i])
                lastlayer = currentlayer
            else:
                currentlayer = np.matmul(self.bpnn_weights[i],lastlayer)+self.bpnn_bias[i]
        bpnn_output = currentlayer.transpose()
        #denormlize
        for i in range(int(self.bpnn_layers[-1])):
            bpnn_output[:,i]=(bpnn_output[:,i]+1)/2*(self.bpnn_norm_out[i,1]-self.bpnn_norm_out[i,0])+self.bpnn_norm_out[i,0]
        bp=np.power(10, bpnn_output)
        return bp
