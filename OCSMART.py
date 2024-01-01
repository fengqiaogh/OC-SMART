#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:10:04 2019

@author: Yongzhen Fan
"""

import os
from os.path import isdir
from os import makedirs
import sys
import numpy as np
import h5py
from netCDF4 import Dataset
import time

from src.L1B import L1B
from src.auxUtils import AUXData
from src.sensorinfo import sensorinfo
from src.ancillary import ANCILLARY
from src.rayleigh import Rayleigh
from src.cloudmask import Cloudmask
#from glint import get_glint_coeff
from src.mlnn import MLNN
from src.ocparam import CHL, TSM, CDOM

#######  Read Input parameters  #####################
input_param = {}
with open("OCSMART_Input.txt") as inputfile:
    for line in inputfile:
        if "=" in line:
            name, var = line.split("=")
            input_param[name.strip()] = var.strip()
####################################################
        
#######  directory setup  ##########################
if 'l1b_path' in input_param.keys():
    L1_path = input_param['l1b_path']
else:
    print('\033[1;31;47mError: no L1B data directory, please set L1B_path in OCSMART_input.txt','\033[m')
    print('Quit OC-SMART ...')
    time.sleep(5)
    sys.exit()
if 'geo_path' in input_param.keys():    
    GEO_path = input_param['geo_path']
else:
    GEO_path = ''
if 'l2_path' in input_param.keys():
    L2_path = input_param['l2_path']
else:
    print('Warning: L2 data directory not specified, using default L2 data directory: ./L2/')
    L2_path = './L2/'
####################################################
    
#######  parameter setup  ##########################
if 'solz_limit' in input_param.keys():
    solz_limit = float(input_param['solz_limit'])
else:    
    solz_limit = 70.0
if 'senz_limit' in input_param.keys():
    senz_limit = float(input_param['senz_limit'])
else:    
    senz_limit = 70.0

water_subpixl_limit = 0.95 # for land/water mask
#glint_max = 0.05 (not needed)
####################################################

#######  subimage setup  ##############################################
# 3 options to define the subimage in the input file
# Option A. (north, south, east, west)
# Option B. (latitude_center, longitude_center, box_width, box_height)
# Option C. (start_line, end_line, start_pixel, end_pixel)
#######################################################################

#######  block processing setup  ###################################################
# block processing can be used to save memory, but it takes longer processing time.
# if you have enough memory, set block_size = -1 to turn off the block processing
block_size=-1
####################################################################################

mission = { 'MODIS-Aqua'  : 'Aqua MODIS',
            'MODIS-Terra' : 'Terra MODIS',
            'SeaWiFS'     : 'SeaWiFS',
            'VIIRS'       : 'Suomi-NPP VIIRS',
            'GOCI'        : 'COMS GOCI',
            'SGLI'        : 'GCOM-C SGLI',
            'OLI'         : 'Landsat-8 OLI',
            'OLCI'        : 'Sentinel-3 OLCI',
            'S2A'         : 'Sentinel-2A MSI',
            'S2B'         : 'Sentinel-2B MSI',
            'EPIC'        : 'DSCOVR EPIC',
            'MERSI2'      : 'FengYun-3D MERSI-II',
            'HICO'        : 'ISS HICO'}
print('Start OC-SMART ... \n')
if isdir(L1_path):
    print('Input level 1 data directory : {}'.format(L1_path))
else:
    print('Input level 1 data directory : {} does not exist, please check input file ...'.format(L1_path)) 
    print('Quit OC-SMART ...')
    sys.exit()
if isdir(L2_path):
    print('Output level 2 data directory : {}'.format(L2_path))
else:
    print('Level 2 data directory: {} does not exist, creating level 2 data directory ... '.format(L2_path))
    makedirs(L2_path)
#print('Output level 2 data format : {} \n'.format(L2_format))

# list all files in the L1 directory
#L1files = [f for f in os.listdir(L1_path) if isfile(join(L1_path, f))]
L1files = [f for f in os.listdir(L1_path)]
nfiles=len(L1files)
print('{} files found in the level 1 directory. \n'.format(nfiles))

# read auxilary data (land/water mask)
aux=AUXData()

# start processing all files
for ifile in np.arange(nfiles):
    fname=L1files[ifile]
    t_start=time.time()    
    print('Processing file {}  {}'.format(ifile+1, fname))    
    
    # get sensor information
    sinfo=sensorinfo(L1_path+fname) 
    if sinfo.sensor_status == 0:
        print('Sensor :',mission[sinfo.sensor])       

        # read level1B data
        l1b=L1B(sensorinfo=sinfo,L1Bname=L1_path+fname,GEOpath=GEO_path)
        l1b.readgeo()
        if l1b.geoloc_status == 0:
            if 'north' in input_param.keys() and 'south' in input_param.keys() and 'east' in input_param.keys() and 'west' in input_param.keys():
                north = float(input_param['north'])
                south = float(input_param['south'])
                east = float(input_param['east'])
                west = float(input_param['west'])                
                l1b.latlon2linepixl(north=north, south=south, east=east, west=west)
            elif 'latitude_center' in input_param.keys() and 'longitude_center' in input_param.keys() and 'box_width' in input_param.keys() and 'box_height' in input_param.keys():
                lat_center = float(input_param['latitude_center'])
                lon_center = float(input_param['longitude_center'])
                box_width = float(input_param['box_width'])
                box_height = float(input_param['box_height'])
                l1b.latlon2linepixl(lat_center=lat_center, lon_center=lon_center, box_width=box_width, box_height=box_height)
            elif 'start_line' in input_param.keys() and 'end_line' in input_param.keys() and 'start_pixel' in input_param.keys() and 'end_pixel' in input_param.keys():
                sline = int(input_param['start_line'])
                eline = int(input_param['end_line'])
                spixl = int(input_param['start_pixel'])
                epixl = int(input_param['end_pixel'])    
                l1b.latlon2linepixl(start_line=sline, end_line=eline, start_pixel=spixl, end_pixel=epixl) 
            else:
                l1b.sline = 0
                l1b.spixl = 0
                l1b.eline = l1b.imagedim[0]
                l1b.epixl = l1b.imagedim[1]

            if l1b.lp_status ==0:
                l1b.readl1b()
                
                # check ancillary files and download from NASA if needed
                anc=ANCILLARY(sensorinfo=sinfo,L1Bname=L1_path+fname)
                anc.download()
                anc.read_no2()
                anc.read_ozone()
                anc.read_met()
                
                #read Rayleigh table
                ray=Rayleigh(info=sinfo)
                
                # get dimension
                l1b_dim=l1b.reflectance.shape
                
                #initialize cloud mask
                cm=Cloudmask(sensorinfo=sinfo)
                
                # initialize the Multilayer Neural Networks
                mlnn=MLNN(sensorinfo=sinfo)
                
                # initialize CHLa, CDOM and TSM algorithm
                chl=CHL(sensorinfo=sinfo)
                tsm=TSM(sensorinfo=sinfo)
                cdom=CDOM(sensorinfo=sinfo)
                
                # initialize all needed matrices 
                niopbands=np.sum(sinfo.band<700)
                l2_mask=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='int16')
                water_portion=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='float64')
                l1b_pressure=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='float64')
                l1b_rh=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='float64')
                l1b_ws=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='float64')
                l1b_oz=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='float64')
                l1b_no2_frac=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='float64')
                l1b_no2_strat=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='float64')
                l1b_no2_tropo=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='float64')
                tg_sol_oz=np.zeros(l1b_dim,dtype='float64')+np.nan
                tg_sen_oz=np.zeros(l1b_dim,dtype='float64')+np.nan
                tg_sol_no2=np.zeros(l1b_dim,dtype='float64')+np.nan
                tg_sen_no2=np.zeros(l1b_dim,dtype='float64')+np.nan
                if sinfo.sensor == 'SeaWiFS':
                    tg_o2 = np.zeros([l1b_dim[0],l1b_dim[1]],dtype='float64')+1.0
                l1b_ray=np.zeros(l1b_dim,dtype='float64')+np.nan
                press_fac=np.zeros(l1b_dim,dtype='float64')+np.nan
                l1b_wcaps=np.zeros(l1b_dim,dtype='float64')+np.nan
                lrc=np.zeros(l1b_dim,dtype='float64')-999.   
                cmask=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='bool')
                glint_coeff=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='float64')
                blockmask=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='bool')
                oos_flag=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='bool')
                aods=np.zeros([l1b_dim[0],l1b_dim[1], int(mlnn.aodnn_layers[-1])],dtype='float64')+np.nan
                rrs=np.zeros([l1b_dim[0],l1b_dim[1], int(mlnn.rrsnn_layers[-1])],dtype='float64')+np.nan    
                lrc_aann=np.zeros([l1b_dim[0],l1b_dim[1], int(mlnn.aann_layers[-1])],dtype='float64')+np.nan 
                chl_oci=np.zeros([l1b_dim[0],l1b_dim[1]])+np.nan
                chl_ocx=np.zeros([l1b_dim[0],l1b_dim[1]])+np.nan
                chl_yoc=np.zeros([l1b_dim[0],l1b_dim[1]])+np.nan
                tsm_yoc=np.zeros([l1b_dim[0],l1b_dim[1]])+np.nan
                aph=np.zeros([l1b_dim[0],l1b_dim[1],int(mlnn.aphnn_layers[-1])],dtype='float64')+np.nan
                adg=np.zeros([l1b_dim[0],l1b_dim[1],int(mlnn.adgnn_layers[-1])],dtype='float64')+np.nan
                bbp=np.zeros([l1b_dim[0],l1b_dim[1],int(mlnn.bbpnn_layers[-1])],dtype='float64')+np.nan
                ap=np.zeros([l1b_dim[0],l1b_dim[1],int(mlnn.apnn_layers[-1])],dtype='float64')+np.nan
                bp=np.zeros([l1b_dim[0],l1b_dim[1],int(mlnn.bpnn_layers[-1])],dtype='float64')+np.nan
                                
                
                # mask invalid radiances data
                mask_valid = np.sum(l1b.reflectance <= 0.0, 2) == 0
                l2_mask[~mask_valid] = 1
                
                # mask high solar and sensor zenith angels
                if sinfo.sensor == 'EPIC':
                    senz_limit = 65.0
                mask_solz=l1b.solz < solz_limit
#                mask_nsolz=l1b.solz >= solz_limit
                mask_senz=l1b.senz < senz_limit
#                mask_nsenz=l1b.senz >= senz_limit
                mask_valid_geo = (mask_valid & (mask_solz & mask_senz)) 
#                l2_mask[mask_valid & (mask_nsolz | mask_nsenz)] = 4
                l2_mask[mask_valid & ~(mask_solz & mask_senz)] = 4
                
                #compute finest resolution for land/water mask
                midimg_x=int(l1b.latitude.shape[0]/2)
                midimg_y=int(l1b.latitude.shape[1]/2)
                lat_delta=np.array([np.abs(l1b.latitude[midimg_x,midimg_y]-l1b.latitude[midimg_x+1,midimg_y]),\
                                           np.abs(l1b.latitude[midimg_x,midimg_y]-l1b.latitude[midimg_x-1,midimg_y]),\
                                           np.abs(l1b.latitude[midimg_x,midimg_y]-l1b.latitude[midimg_x,midimg_y+1]),\
                                           np.abs(l1b.latitude[midimg_x,midimg_y]-l1b.latitude[midimg_x,midimg_y-1])])
                lon_delta=np.array([np.abs(l1b.longitude[midimg_x,midimg_y]-l1b.longitude[midimg_x+1,midimg_y]),\
                                           np.abs(l1b.longitude[midimg_x,midimg_y]-l1b.longitude[midimg_x-1,midimg_y]),\
                                           np.abs(l1b.longitude[midimg_x,midimg_y]-l1b.longitude[midimg_x,midimg_y+1]),\
                                           np.abs(l1b.longitude[midimg_x,midimg_y]-l1b.longitude[midimg_x,midimg_y-1])])
                
                l1b_dxdy=np.amax([np.mean(lat_delta[np.where(lat_delta<1)[0]]),\
                                  np.mean(lon_delta[np.where(lon_delta<1)[0]])])
                
                # compute land/water mask
                if sinfo.sensor in ['EPIC', 'VIIRS', 'MODIS-Aqua', 'MODIS-Terra', 'SeaWiFS','MERSI2']:
                    water_portion[mask_valid_geo]=aux.maskland(l1b.latitude[mask_valid_geo], l1b.longitude[mask_valid_geo], l1b_dxdy)
                    mask_water = water_portion > water_subpixl_limit
                    mask_nwater = water_portion <= water_subpixl_limit
                elif sinfo.sensor in ['SGLI', 'OLCI', 'OLI', 'S2A', 'S2B', 'GOCI','HICO']:
                    mask_water = l1b.landmask == 0
                    mask_nwater = ~mask_water    
                
                # Sentinel-2, Landsat-8 and GOCI images are too large, set block processing anyways                
                if sinfo.sensor in ['OLI', 'S2A', 'S2B', 'GOCI']:
                    block_size=2500
                else:
                    block_size=-1   
                    
                #mask all land pixels
                mask_valid_geo_water = mask_valid_geo & mask_water
                l2_mask[mask_valid_geo & mask_nwater] = 16
                
                #compute transmittance of gases
                tg_sol_oz[mask_valid_geo_water,:], tg_sen_oz[mask_valid_geo_water,:] = anc.trans_ozone(sinfo.koz,\
                                                                                   l1b.latitude[mask_valid_geo_water],\
                                                                                   l1b.longitude[mask_valid_geo_water], \
                                                                                   l1b.solz[mask_valid_geo_water], \
                                                                                   l1b.senz[mask_valid_geo_water])
                
                tg_sol_no2[mask_valid_geo_water,:], tg_sen_no2[mask_valid_geo_water,:] = anc.trans_no2(sinfo.kno2, \
                                                                                     anc.month, \
                                                                                     l1b.latitude[mask_valid_geo_water],\
                                                                                     l1b.longitude[mask_valid_geo_water], \
                                                                                     l1b.solz[mask_valid_geo_water], \
                                                                                     l1b.senz[mask_valid_geo_water]) 
                
                
                # get met data: pressure, RH, windspeed
                l1b_pressure[mask_valid_geo_water], l1b_rh[mask_valid_geo_water], l1b_ws[mask_valid_geo_water] = anc.get_metdata(l1b.latitude[mask_valid_geo_water],l1b.longitude[mask_valid_geo_water])
                l1b_oz[mask_valid_geo_water] = anc.l1b_oz
                l1b_no2_frac[mask_valid_geo_water] = anc.l1b_no2_frac
                l1b_no2_strat[mask_valid_geo_water] = anc.l1b_no2_strat
                l1b_no2_tropo[mask_valid_geo_water] = anc.l1b_no2_tropo
                # get Rayleigh reflectance and correct for real time pressue
                l1b_ray[mask_valid_geo_water,:]=ray.corr_ray(l1b.solz[mask_valid_geo_water],\
                                                           l1b.senz[mask_valid_geo_water],\
                                                           l1b.relaz[mask_valid_geo_water],\
                                                           l1b_ws[mask_valid_geo_water])
                
                press_fac[mask_valid_geo_water,:]=ray.corr_ray_press(l1b.solz[mask_valid_geo_water],\
                                                                   l1b.senz[mask_valid_geo_water],\
                                                                   l1b_pressure[mask_valid_geo_water])
                
                l1b_ray[mask_valid_geo_water,:]=l1b_ray[mask_valid_geo_water,:]*press_fac[mask_valid_geo_water,:]
                
                # Oxygen absorption correction for SeaWiFS band 7
                if sinfo.sensor == 'SeaWiFS':
                    tg_o2[mask_valid_geo_water] = anc.trans_o2_ray(l1b.solz[mask_valid_geo_water],l1b.senz[mask_valid_geo_water])
                    l1b_ray[mask_valid_geo_water,6]=l1b_ray[mask_valid_geo_water,6]*tg_o2[mask_valid_geo_water]
                # get whitecaps reflectance
                l1b_wcaps[mask_valid_geo_water,:]=anc.whitecaps(sinfo.band,l1b.solz[mask_valid_geo_water],\
                                                              l1b.senz[mask_valid_geo_water], \
                                                              l1b_ws[mask_valid_geo_water], \
                                                              l1b_pressure[mask_valid_geo_water], ray.taur)
                 
                # compute Rayleigh corrected reflectance and mask negative value, if any
                lrc[mask_valid_geo_water,:]=l1b.reflectance[mask_valid_geo_water,:]/tg_sol_oz[mask_valid_geo_water,:]/tg_sen_oz[mask_valid_geo_water,:]\
                                           /tg_sol_no2[mask_valid_geo_water,:]/tg_sen_no2[mask_valid_geo_water,:]-l1b_wcaps[mask_valid_geo_water,:]-l1b_ray[mask_valid_geo_water,:]
#                lrc[mask_valid_geo_water,:]=l1b.reflectance[mask_valid_geo_water,:]/tg_sol_oz[mask_valid_geo_water,:]/tg_sen_oz[mask_valid_geo_water,:]\
#                                           /tg_sol_no2[mask_valid_geo_water,:]/tg_sen_no2[mask_valid_geo_water,:]-l1b_wcaps[mask_valid_geo_water,:]
                
                neg=np.sum(lrc<0.0,axis=2)
                mask_valid_geo_water_lrcposi = neg == 0
                lrc[lrc==-999]=999
                neg1=np.sum(lrc<0.0,axis=2)
                mask_valid_geo_water_lrcneg = neg1 > 0
                l2_mask[mask_valid_geo_water_lrcneg] = 1024
                
                # compute sunglint risk (not needed)
            #    glint_coeff[mask_valid_geo_water] = get_glint_coeff(l1b.solz[mask_valid_geo_water], \
            #                                                        l1b.senz[mask_valid_geo_water], \
            #                                                        l1b.relaz[mask_valid_geo_water], \
            #                                                        l1b_ws[mask_valid_geo_water])
            #    mask_glint = glint_coeff > glint_max
            #    mask_nglint = glint_coeff <= glint_max   
            #    mask_valid_geo_water_lrcposi_nglint = mask_valid_geo_water_lrcposi & mask_nglint
            #    l2_mask[mask_valid_geo_water_lrcposi & mask_glint] = 128
                
                # cloud mask, use bands near 412,555,670,865 
                if sinfo.sensor == 'OLI':
                    cmask = l1b.cloud
                else:
                    cmask[mask_valid_geo_water_lrcposi]=cm.run_cloudmask(lrc[mask_valid_geo_water_lrcposi,:])
                #expand the cloud mask by 1 pixel
#                if sinfo.sensor in ['EPIC']:
#                    cmask_idx = np.concatenate((np.where(cmask==1)[0]+1,np.where(cmask==1)[0]-1))
#                    cmask_idy = np.concatenate((np.where(cmask==1)[1],np.where(cmask==1)[1]))
#                    cmask[cmask_idx,cmask_idy] = True
#                    cmask_idx = np.concatenate((np.where(cmask==1)[0],np.where(cmask==1)[0]))
#                    cmask_idy = np.concatenate((np.where(cmask==1)[1]+1,np.where(cmask==1)[1]-1))
#                    cmask[cmask_idx,cmask_idy] = True
                mask_valid_geo_water_lrcposi_cloud= mask_valid_geo_water_lrcposi & cmask
                mask_valid_geo_water_lrcposi_nocloud= mask_valid_geo_water_lrcposi & ~cmask
                l2_mask[mask_valid_geo_water_lrcposi_cloud] = 64
                
                # run Multilayer Neural Network (MLNN) retrieval on Lrc data
                #if image is too large, separate into blocks to process
                if block_size < 0:
                    
                    oos_flag[mask_valid_geo_water_lrcposi_nocloud],lrc_aann[mask_valid_geo_water_lrcposi_nocloud,:]=\
                                                                                     mlnn.compute_aann(l1b.solz[mask_valid_geo_water_lrcposi_nocloud],\
                                                                                     l1b.senz[mask_valid_geo_water_lrcposi_nocloud],\
                                                                                     l1b.relaz[mask_valid_geo_water_lrcposi_nocloud],\
                                                                                     lrc[mask_valid_geo_water_lrcposi_nocloud,:],\
                                                                                     l1b_rh[mask_valid_geo_water_lrcposi_nocloud])    
                    aods[mask_valid_geo_water_lrcposi_nocloud,:]=mlnn.compute_aodnn(l1b.solz[mask_valid_geo_water_lrcposi_nocloud],\
                                                                                     l1b.senz[mask_valid_geo_water_lrcposi_nocloud],\
                                                                                     l1b.relaz[mask_valid_geo_water_lrcposi_nocloud],\
                                                                                     lrc[mask_valid_geo_water_lrcposi_nocloud,:],\
                                                                                     l1b_rh[mask_valid_geo_water_lrcposi_nocloud])
                    rrs[mask_valid_geo_water_lrcposi_nocloud,:]=mlnn.compute_rrsnn(l1b.solz[mask_valid_geo_water_lrcposi_nocloud],\
                                                                                     l1b.senz[mask_valid_geo_water_lrcposi_nocloud],\
                                                                                     l1b.relaz[mask_valid_geo_water_lrcposi_nocloud],\
                                                                                     lrc[mask_valid_geo_water_lrcposi_nocloud,:])
                    aph[mask_valid_geo_water_lrcposi_nocloud,:]=mlnn.compute_aphnn(rrs[mask_valid_geo_water_lrcposi_nocloud,:])
                    adg[mask_valid_geo_water_lrcposi_nocloud,:]=mlnn.compute_adgnn(rrs[mask_valid_geo_water_lrcposi_nocloud,:])
                    bbp[mask_valid_geo_water_lrcposi_nocloud,:]=mlnn.compute_bbpnn(rrs[mask_valid_geo_water_lrcposi_nocloud,:])
                    bp[mask_valid_geo_water_lrcposi_nocloud,:]=mlnn.compute_bpnn(rrs[mask_valid_geo_water_lrcposi_nocloud,:])
                    ap[mask_valid_geo_water_lrcposi_nocloud,:]=mlnn.compute_apnn(rrs[mask_valid_geo_water_lrcposi_nocloud,:])
                else:                    
                    blockmask=np.zeros([l1b_dim[0],l1b_dim[1]],dtype='bool')
                    nblocks_x=int(np.ceil(l1b_dim[1]/block_size))
                    nblocks_y=int(np.ceil(l1b_dim[0]/block_size))
                    print('Processing image in {} blocks ... '.format(nblocks_x*nblocks_y))
                    block_boundary_x=np.zeros(nblocks_x+1,dtype=int)
                    block_boundary_y=np.zeros(nblocks_y+1,dtype=int)
                    if nblocks_x == 1:
                        block_boundary_x[0] = 0
                        block_boundary_x[1] = l1b_dim[1]
                    else:
                        block_boundary_x[0:nblocks_x]=np.arange(0,l1b_dim[1],block_size)
                        block_boundary_x[nblocks_x] = l1b_dim[1]
                    if nblocks_y == 1:
                        block_boundary_y[0] = 0
                        block_boundary_y[1] = l1b_dim[0]
                    else:
                        block_boundary_y[0:nblocks_y]=np.arange(0,l1b_dim[0],block_size)
                        block_boundary_y[nblocks_y] = l1b_dim[0]
                    for iby in np.arange(nblocks_y):
                        for ibx in np.arange(nblocks_x):                            
                            if ibx == 1:
                                print('Porcessing block ',iby)
                            else:
                                print('Porcessing block ',iby*nblocks_y+ibx+1)
                            blockmask[:,:] = False
                            blockmask[block_boundary_y[iby]:block_boundary_y[iby+1],block_boundary_x[ibx]:block_boundary_x[ibx+1]] = True                        
                            process_mask = mask_valid_geo_water_lrcposi_nocloud & blockmask
                            if np.sum(process_mask)>0:
                                oos_flag[process_mask],lrc_aann[process_mask,:]=mlnn.compute_aann(l1b.solz[process_mask],\
                                                                         l1b.senz[process_mask],\
                                                                         l1b.relaz[process_mask],\
                                                                         lrc[process_mask,:],\
                                                                         l1b_rh[process_mask])
                                aods[process_mask,:]=mlnn.compute_aodnn(l1b.solz[process_mask],\
                                                                         l1b.senz[process_mask],\
                                                                         l1b.relaz[process_mask],\
                                                                         lrc[process_mask,:],\
                                                                         l1b_rh[process_mask])
                                rrs[process_mask,:]=mlnn.compute_rrsnn(l1b.solz[process_mask],\
                                                                         l1b.senz[process_mask],\
                                                                         l1b.relaz[process_mask],\
                                                                         lrc[process_mask,:])
                                aph[process_mask,:]=mlnn.compute_aphnn(rrs[process_mask,:])
                                adg[process_mask,:]=mlnn.compute_adgnn(rrs[process_mask,:])
                                bbp[process_mask,:]=mlnn.compute_bbpnn(rrs[process_mask,:])
                                ap[process_mask,:]=mlnn.compute_apnn(rrs[process_mask,:])
                                bp[process_mask,:]=mlnn.compute_bpnn(rrs[process_mask,:])
            
                mask_valid_geo_water_lrcposi_nocloud_oos = mask_valid_geo_water_lrcposi_nocloud & oos_flag
                l2_mask[mask_valid_geo_water_lrcposi_nocloud_oos] = 256
                
                # retrieve CHL, CDOM and TSM
#                chl_ocx[mask_valid_geo_water_lrcposi_nocloud]=chl.get_chl_ocx(rrs[mask_valid_geo_water_lrcposi_nocloud,:])
                chl_oci[mask_valid_geo_water_lrcposi_nocloud]=chl.get_chl_oci(rrs[mask_valid_geo_water_lrcposi_nocloud,:])
                chl_yoc[mask_valid_geo_water_lrcposi_nocloud]=chl.get_chl_yoc(rrs[mask_valid_geo_water_lrcposi_nocloud,:])
                tsm_yoc[mask_valid_geo_water_lrcposi_nocloud]=tsm.get_tsm_yoc(rrs[mask_valid_geo_water_lrcposi_nocloud,:])

                # write L2 file in H5 format
                print('Writing level 2 file {} ... '.format(os.path.splitext(fname)[0]+'_L2_OCSMART.h5'))
                hf = h5py.File(L2_path+os.path.splitext(fname)[0]+'_L2_OCSMART.h5','w')    
                hf.create_dataset('Latitude',dtype='float32',data = l1b.latitude,compression="gzip", compression_opts=9)
                hf.create_dataset('Longitude',dtype='float32',data = l1b.longitude,compression="gzip", compression_opts=9)
                hf.create_dataset('Solar_zenith',dtype='float32',data = l1b.solz,compression="gzip", compression_opts=9)
                hf.create_dataset('Sensor_zenith',dtype='float32',data = l1b.senz,compression="gzip", compression_opts=9)
                hf.create_dataset('Relative_azimuth',dtype='float32',data = l1b.relaz,compression="gzip", compression_opts=9) 
                hf.create_dataset('L2_flags',dtype='int16',data = l2_mask,compression="gzip", compression_opts=9)
                hf.create_dataset('chlor_a(oci)',dtype='float64',data = chl_oci,compression="gzip", compression_opts=9)
#                hf.create_dataset('chlor_a(ocx)',dtype='float64',data = chl_ocx,compression="gzip", compression_opts=9)
                hf.create_dataset('chlor_a(yoc)',dtype='float64',data = chl_yoc,compression="gzip", compression_opts=9)
                hf.create_dataset('tsm(yoc)',dtype='float64',data = tsm_yoc,compression="gzip", compression_opts=9)
                                   
                gw = hf.create_group('AOD')
                for i in np.arange(mlnn.aodnn_layers[-1]):
                    gw.create_dataset('AOD_'+str(sinfo.band[i])+'nm',dtype='float64',data = aods[:,:,i],compression="gzip", compression_opts=9)
                gw = hf.create_group('Rrs')    
                for i in np.arange(mlnn.rrsnn_layers[-1]):
                    gw.create_dataset('Rrs_'+str(sinfo.band[i])+'nm',dtype='float64',data = rrs[:,:,i],compression="gzip", compression_opts=9)
                gw = hf.create_group('aph')    
                for i in np.arange(mlnn.aphnn_layers[-1]):
                    gw.create_dataset('aph_'+str(sinfo.band[i])+'nm',dtype='float64',data = aph[:,:,i],compression="gzip", compression_opts=9)
                gw = hf.create_group('adg')    
                for i in np.arange(mlnn.adgnn_layers[-1]):
                    gw.create_dataset('adg_'+str(sinfo.band[i])+'nm',dtype='float64',data = adg[:,:,i],compression="gzip", compression_opts=9)
                gw = hf.create_group('bbp')    
                for i in np.arange(mlnn.bbpnn_layers[-1]):
                    gw.create_dataset('bbp_'+str(sinfo.band[i])+'nm',dtype='float64',data = bbp[:,:,i],compression="gzip", compression_opts=9) 
                gw = hf.create_group('ap')    
                for i in np.arange(mlnn.apnn_layers[-1]):
                    gw.create_dataset('ap_'+str(sinfo.band[i])+'nm',dtype='float64',data = ap[:,:,i],compression="gzip", compression_opts=9) 
                gw = hf.create_group('bp')    
                for i in np.arange(mlnn.bpnn_layers[-1]):
                    gw.create_dataset('bp_'+str(sinfo.band[i])+'nm',dtype='float64',data = bp[:,:,i],compression="gzip", compression_opts=9)     
                hf.close()
                print('Processing finished in %.2f second.\n'%(time.time()-t_start))
            else:
                print('\033[1;31;47mWARNING:Unable to extract subimage, processing terminated... ', '\033[m')
                continue
        else:
            continue
    else:
        continue
