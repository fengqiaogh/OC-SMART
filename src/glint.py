#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 21:27:17 2019

@author: cyberbass
"""

import numpy as np


def reflec(theta):
    ref = 4./3.
    phi = np.zeros(len(theta))
    phi[theta < 0.00001] = 0.0204078
    thetap = np.arcsin(np.sin(theta[theta >= 0.00001])/ref)
    phi[theta >= 0.00001] = ((np.sin(theta[theta >= 0.00001]-thetap)/np.sin(theta[theta >= 0.00001]+thetap))**2\
                           +(np.tan(theta[theta >= 0.00001]-thetap)/np.tan(theta[theta >= 0.00001]+thetap))**2)/2
    
    return phi    

def get_glint_coeff(sza,vza,raz,ws):
    print('Compute glint coefficient ... ')
    solz  = np.radians(sza)
    senz  = np.radians(vza)
    relaz = np.radians(raz)
    
    omega = np.arccos(np.cos(senz)*np.cos(solz)-np.sin(senz)*np.sin(solz)*np.cos(relaz))/2.0;
    
    omega[omega <= 0.] <= 1.0e-7
    
    beta = np.arccos((np.cos(senz)+np.cos(solz))/(2.0*np.cos(omega)));
    
    beta[beta <= 0.] <= 1.0e-7    
    
    alpha = np.arccos((np.cos(beta)*np.cos(solz)-np.cos(omega))/(np.sin(beta)*np.sin(solz)));
    
    alpha[np.sin(relaz) < 0.] = -1.0 * alpha[np.sin(relaz) < 0.]     
    
    sigc = 0.04964*np.sqrt(ws);
    sigu = 0.04964*np.sqrt(ws);
    chi = 0.;
    alphap = alpha+chi;
    swig = np.sin(alphap)*np.tan(beta)/sigc;
    eta = np.cos(alphap)*np.tan(beta)/sigu;
    expon = -1.0*(swig**2+eta**2)/2.;    
    
    expon[np.isnan(expon)] = -30.
    expon[expon < -30.] = -30.0
    expon[expon > 30.] = 30.0

    prob = np.exp(expon)/(2.0*np.pi*sigu*sigc);
#    print(prob)
    rho = reflec(omega);
#    print(rho)
#    print(senz)
#    print(beta)
    glint = rho*prob/(4.*np.cos(senz)*np.cos(beta)**4);
    
    return glint



if __name__ == '__main__':
    
    sza = np.array([33.4653,55.5])
    vza = np.array([10.733335,25.68])
    raz = np.array([56.53,128.658])
    ws  = np.array([5.0,2.5])
    
    glint_coeff = get_glint_coeff(sza,vza,raz,ws)
    print(glint_coeff)    
