# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:41:10 2018

@author: jakeprecht
"""
from ffta import pixel, utils
from ffta.utils import load
from Biexp import Biexp
import numpy as np
from matplotlib import pyplot as plt
#import pandas as pd

def grab_tfp(tau_e, tau_i = 500, beta_e = 1, beta_i = 1, electronic_fraction = 1, params_path = r'C:\Users\jakeprecht\DDHO\data\tip8.cfg'):
    """tau should be in us
    beta is stretched exponential value"""
    
    
    #Generate cantilever data
    Canti = Biexp("rise", tau_e * 1e-6, beta_e, tau_i * 1e-6, beta_i, electronic_fraction)

    data_bi = Canti.run_stretched_simulation()
    
    #Apply noise
    #noise_magnitude = 0.3e-9
    #noise = abs(np.random.normal(0, noise_magnitude)) #apply random amount of noise.  noise has a standard deviation of "noise_magnitude"
    #noise=0.1e-9
    #noise=0
    
    #data_bi_noisy = Canti.noise_on_z(data_bi,noise)

    #Reshape data for proper resolution
    data_bi_reshape, t_bi_reshape = Canti.Z_reshape(data_bi)
    #data_bi_noisy_reshape, t_bi_reshape = Canti.Z_reshape(data_bi_noisy)
    
    #Save data
    #zdatastring = 'C:/Users/Jake/DDHO/z_data_noisy_' + str(tau_e) + '_' + str(tau_i) + '_us.txt'
    #zdatastring_no_noise = 'C:/Users/Jake/DDHO/z_data_no_noise_' + str(tau_e) + '_' + str(tau_i) + '_us.txt'
    zdatastring_no_noise = 'C:/Users/jakeprecht/DDHO/z_data_no_noise.txt'
    #np.savetxt(zdatastring,data_bi_noisy_reshape,delimiter=',')
    np.savetxt(zdatastring_no_noise,data_bi_reshape,delimiter=',')
    
    signal_file = zdatastring_no_noise
    
    #Hilbert transform / process data using pixel.py
    signal_array = utils.load.signal(signal_file)
    n_pixels, params = utils.load.configuration(params_path)
    p = pixel.Pixel(signal_array, params)
    tfp, shift, inst_freq = p.analyze()
    tfp = tfp * 1e6
    print('tau is ' + str(tau_e) + ' and tfp is ' + str(tfp))
    
    return tfp

def generate_curve(*list_of_taus):
    #taus is list of taus (in us)
    #init collection lists for tau and tfp plotting
    tau_list = []
    tfp_list = []

    for taus in list_of_taus:
        for tau in taus:
            tfp = grab_tfp(tau)
        
            tau_list.append(tau)
            tfp_list.append(tfp)
        
    fig2 = plt.figure(2)
    plt.scatter(tau_list, tfp_list,c='b', label='Calibration Curve')
    plt.xlabel('Tau (us)', fontsize=12, weight = 'bold')
    plt.ylabel('Tfp (us)', fontsize=12, x=-1.0, weight = 'bold')
    plt.xscale('log')
    plt.xlim(0.001,1000)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)#, '725', '750', '775', '800'), fontsize =12)
    #p.plot()

if __name__ == "__main__":
    #taus = [.001,.01,.1,1,10,100,1000]
    #taus = np.linspace(.001,1000,num=51)
    taus = np.logspace(-3,3,num=51)
    generate_curve(taus)
    #grab_tfp(100)