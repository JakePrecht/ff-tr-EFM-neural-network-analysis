# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:53:47 2019

@author: Jake
"""
import numpy as np
from igor.binarywave import load as loadibw
from matplotlib import pyplot as plt

disp_path_1 = "C:/Users/Jake/Desktop/Run4/4/wave_phase_16.ibw"
disp_1 = loadibw(disp_path_1)['wave']['wData']
disp_1 = disp_1[:,0]

disp_path_2 = "C:/Users/Jake/Desktop/Run4/10/wave_phase_16.ibw"
disp_2 = loadibw(disp_path_2)['wave']['wData']
disp_2 = disp_2[:,0]

t_array = np.arange(len(disp_1))/10 #units of microseconds
t_array = t_array - t_array.mean() #center it such that the trigger is at t=0

trigger_idx = int(len(disp_1) / 2)


fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (12,6))

ax1.plot(t_array[trigger_idx-250:trigger_idx+250],
         disp_1[trigger_idx-250:trigger_idx+250], c='b', lw=4)
ax1.set_ylabel('Signal', fontsize = 16, weight = 'bold')
ax1.set_xlabel('Time', fontsize = 16, weight = 'bold')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.axvline(x=0, c='k', lw = 3) #this is the trigger
ax1.set_title('Tau = x', fontsize = 20)

ax2.plot(t_array[trigger_idx-250:trigger_idx+250],
         disp_2[trigger_idx-250:trigger_idx+250], c ='r', lw=4)
#ax2.set_ylabel('Signal', fontsize = 16, weight = 'bold')
ax2.set_xlabel('Time', fontsize = 16, weight = 'bold')
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.axvline(x=0, c='k', lw = 3) #this is the trigger
ax2.set_title('Tau = 1000*x', fontsize = 20)