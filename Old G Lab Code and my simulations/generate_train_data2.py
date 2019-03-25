"""Generates MANY displacement curves or instantaneous frequency curves.
Outputs .npy files with inst. freq. or displacement.  last data point in output is the tau"""


#Maybe first manually import everything cuz import is being weird
from ffta import pixel, utils
from ffta.utils import load
from Biexp import Biexp
import numpy as np
#import pandas as pd
import time
import datetime


now = datetime.datetime.now().time()
print('Generation started at ', now)

t0 = time.time()




"""
#change path here to change where you save it (make sure path exists too!!)
path = 'D:/jake/DDHO Data/displacement/10us/'

#GENERATE DISPLACEMENT CURVES
for i in range(10,310,10): #set bucketing in this line
    tau = i
    Canti = Biexp("rise", tau * 1e-6)
    disp_mono, data_bi = Canti.run_simulation()
    disp_mono_reshape = Canti.Z_reshape_ROI_only(disp_mono)
    
    data_no_noise = np.append(disp_mono_reshape, tau)
    
    for j in range(10000):
        data_mono_noisy = Canti.noise_on_z(disp_mono,0.03e-9)
        data_mono_noisier = Canti.noise_on_z(disp_mono,0.3e-9)
        
        data_mono_noisy_reshape = Canti.Z_reshape_ROI_only(data_mono_noisy)
        data_mono_noisier_reshape = Canti.Z_reshape_ROI_only(data_mono_noisier)
        
        
        data_noisy = np.append(data_mono_noisy_reshape, tau)
        data_noisier = np.append(data_mono_noisier_reshape, tau)
        
        inst_freq_string_no_noise = path + '/no_noise/disp_' + str(i) + '_'+ str(j) + '.npy'
        inst_freq_string = path + '/0pt1noise/disp_' + str(i) + '_'+ str(j) + '.npy'
        inst_freq_string_noisier = path + '/1noise/disp_' + str(i) + '_'+ str(j) + '.npy'
        
        np.save(inst_freq_string_no_noise, data_no_noise)
        np.save(inst_freq_string, data_noisy)
        np.save(inst_freq_string_noisier, data_noisier)
        
    print('Finished generating', i, 'us series at ', datetime.datetime.now().time())
"""





"""
#GENERATE INSTANTANEOUS FREQUENCY CURVES

#change path here to change where you save it (make sure path exists too!!)
path = 'D:/jake/DDHO Data/inst_freq/25us/'

for i in range(25,325,25): #set bucketing in this line
    Canti = Biexp("rise", i * 1e-6)
    data_mono, data_bi = Canti.run_simulation()
    data_mono_reshape, t_mono_reshape = Canti.Z_reshape(data_mono)
    signal_array_no_noise = data_mono_reshape
    
    params_file = 'C:/Users/jakeprecht/DDHO/data/parameters.cfg'
    n_pixels, params = utils.load.configuration(params_file)
    
    p_no_noise = pixel.Pixel(signal_array_no_noise, params)
    tfp_no_noise, shift, inst_freq_no_noise = p_no_noise.analyze()
    tfp_no_noise = tfp_no_noise * 1e6
    
    p_no_noise.cut = np.append(p_no_noise.cut, tfp_no_noise)
    p_no_noise.cut = np.append(p_no_noise.cut, i)
    
    for j in range(10000):
        tau = i
        
        data_mono_noisy = Canti.noise_on_z(data_mono,0.03e-9)
        data_mono_noisier = Canti.noise_on_z(data_mono,0.3e-9)
        
        data_mono_noisy_reshape, t_mono_reshape = Canti.Z_reshape(data_mono_noisy)
        data_mono_noisier_reshape, t_mono_reshape = Canti.Z_reshape(data_mono_noisier)
        
        signal_array = data_mono_noisy_reshape
        signal_array_noisier = data_mono_noisier_reshape
        
        params_file = 'C:/Users/jakeprecht/DDHO/data/parameters.cfg'
        n_pixels, params = utils.load.configuration(params_file)
        
        p = pixel.Pixel(signal_array, params)
        p_noisier = pixel.Pixel(signal_array_noisier, params)
        
        tfp, shift, inst_freq = p.analyze()
        tfp_noisier, shift, inst_freq = p_noisier.analyze()
        
        tfp = tfp * 1e6
        tfp_noisier = tfp_noisier * 1e6
        
        p.cut = np.append(p.cut, tfp)
        p.cut = np.append(p.cut, tau)
        p_noisier.cut = np.append(p_noisier.cut, tfp_noisier)
        p_noisier.cut = np.append(p_noisier.cut, tau)
        
        #list_of_taus.append(tau)
        #list_of_tfps.append(tfp)
        
        #list_of_taus_noisier.append(tau)
        #list_of_tfps_noisier.append(tfp_noisier)
        
        inst_freq_string_no_noise = path + '/no_noise/inst_freq_' + str(i) + '_'+ str(j) + '.npy'
        inst_freq_string = path + '/0pt1noise/inst_freq_' + str(i) + '_'+ str(j) + '.npy'
        inst_freq_string_noisier = path + '/1noise/inst_freq_' + str(i) + '_'+ str(j) + '.npy'
        
        np.save(inst_freq_string_no_noise, p_no_noise.cut)
        np.save(inst_freq_string, p.cut)
        np.save(inst_freq_string_noisier,p_noisier.cut)
        
    print('Finished generating', i, 'us series at ', datetime.datetime.now().time())

"""

"""
#Generate random amount of noise on each sample INST_FREQ version:
#change path here to change where you save it (make sure path exists too!!)
path = 'D:/jake/DDHO Data/inst_freq/25us/'

for i in range(25,325,25): #set bucketing in this line
    Canti = Biexp("rise", i * 1e-6)
    data_mono, data_bi = Canti.run_simulation()
    data_mono_reshape, t_mono_reshape = Canti.Z_reshape(data_mono)
    
    tau = i
    noise_magnitude = 0.3e-8
    
    #for j in range(10000):
    for j in range(10000):

        noise = abs(np.random.normal(0, noise_magnitude)) #apply random amount of noise.  noise has a standard deviation of "noise_magnitude"
        #noise = noise_magnitude
        
        data_mono_noisy = Canti.noise_on_z(data_mono,noise)
        
        data_mono_noisy_reshape, t_mono_reshape = Canti.Z_reshape(data_mono_noisy)
        
        signal_array = data_mono_noisy_reshape
        
        params_file = 'C:/Users/jakeprecht/DDHO/data/parameters.cfg'
        n_pixels, params = utils.load.configuration(params_file)
        
        p = pixel.Pixel(signal_array, params)
        
        tfp, shift, inst_freq = p.analyze()
        
        tfp = tfp * 1e6
        
        p.cut = np.append(p.cut, tfp)
        p.cut = np.append(p.cut, tau)
        
        #inst_freq_string = path + '/random_noise/inst_freq_' + str(i) + '_'+ str(j) + '.npy'
        inst_freq_string = path + '/random_noise_10/inst_freq_' + str(i) + '_'+ str(j) + '.npy'


        np.save(inst_freq_string, p.cut)
        
    print('Finished generating', i, 'us series at ', datetime.datetime.now().time())
"""



#Generate random amount of noise on each sample DISPLACEMENT version:
#change path here to change where you save it (make sure path exists too!!)

"""
#GENERATE INST_FREQ CURVES V2
path = 'D:/jake/DDHO Data/inst_freq/2 per decade 10 ns to 1 ms 300 kHz/'
taus = np.array([10e-9, 31.62e-9, 100e-9, 316.2e-9, 1e-6, 3.162e-6, 10e-6, 31.62e-6, 100e-6, 316.2e-6, 1e-3])
#for i in range(10,310,10): #set bucketing in this line
k = 0
for i in taus:
    tau = i
    #Canti = Biexp("rise", tau * 1e-6)
    Canti = Biexp("rise", tau_e = tau, beta_e = 1, tau_i=100e-6, beta_i=1, electronic_fraction=1, path = 'C:/Users/jakeprecht/DDHO/data/parameters_2018_08_29.cfg')
    disp_bi = Canti.run_stretched_simulation()
    #disp_bi_reshape = Canti.Z_reshape_ROI_only(disp_bi) #useful for sim displacement
    disp_bi_reshape, t = Canti.Z_reshape(disp_bi) #useful for sim inst_freq
    
    noise_magnitude = 0.3e-9
    #noise = 0.03e-9
    #noise = 0
    for j in range(2000):
        noise = abs(np.random.normal(0, noise_magnitude)) #apply random amount of noise.  noise has a standard deviation of "noise_magnitude"
        data_mono_noisy = Canti.noise_on_z(disp_bi_reshape,noise)
        
        signal_array = data_mono_noisy
        
        params_file = 'C:/Users/jakeprecht/DDHO/data/parameters_2018_08_29.cfg'
        n_pixels, params = utils.load.configuration(params_file)
        
        p = pixel.Pixel(signal_array, params)
        
        tfp, shift, inst_freq = p.analyze()
        
        tfp = tfp * 1e6
        
        #p.cut = np.append(p.cut, tfp)
        p.cut = np.append(p.cut, tau)
        
        inst_freq_string = path + '/random_noise_1/inst_freq_' + str(k) + '_'+ str(j) + '.npy'


        np.save(inst_freq_string, p.cut)
        
        
        
        
        #data_noisy = np.append(data_mono_noisy, tau)
        
        #disp_string = path + '/1noise/disp_' + str(k) + '_'+ str(j) + '.npy'
        
        #np.save(disp_string, data_noisy)
        
    print('Finished generating', i, 'us series at ', datetime.datetime.now().time())
    
    k+=1


"""


#GENERATE DISPLACEMENT CURVES for series of different tip parameters
save_path = 'D:/jake/DDHO Data/displacement/2018_09_17/'
taus = np.array([10e-9, 31.62e-9, 100e-9, 316.2e-9, 1e-6, 3.162e-6, 10e-6, 31.62e-6, 100e-6, 316.2e-6, 1e-3])
#params_paths = list of paths to different .cfg files with cantilever parameters
params_paths = ['C:/Users/jakeprecht/DDHO/data/2018_09_12 tip1.cfg', 
                'C:/Users/jakeprecht/DDHO/data/2018_09_12 tip2.cfg', 
                'C:/Users/jakeprecht/DDHO/data/2018_09_12 tip3.cfg',
                'C:/Users/jakeprecht/DDHO/data/2018_09_17 tip1.cfg', 
                'C:/Users/jakeprecht/DDHO/data/2018_09_17 tip2.cfg', 
                'C:/Users/jakeprecht/DDHO/data/2018_09_17 tip3.cfg',
                'C:/Users/jakeprecht/DDHO/data/2018_09_17 tip4.cfg', 
                'C:/Users/jakeprecht/DDHO/data/2018_09_17 tip5.cfg', 
                'C:/Users/jakeprecht/DDHO/data/2018_09_17 tip6.cfg']
h=1#counter/index for tip
for params_path in params_paths:
    k = 0 #counter/index for tau
    print(params_path)
    
    for tau in taus:
        #tau = i#this line could be removed by changing above line to "for tau in taus:" ?
        Canti = Biexp("rise", tau_e = tau, beta_e = 1, tau_i=100e-6, beta_i=1, electronic_fraction=1, path = params_path)
        disp_bi = Canti.run_stretched_simulation()
        disp_bi_reshape = Canti.Z_reshape_ROI_only(disp_bi)
    
        #noise_magnitude = 0.3e-8
        #noise = 0.03e-9
        noise=0
        
        for j in range(500): #j is counter/index for specific curve with noise on it
            #noise = abs(np.random.normal(0, noise_magnitude)) #apply random amount of noise.  noise has a standard deviation of "noise_magnitude"
            data_mono_noisy = Canti.noise_on_z(disp_bi_reshape,noise)
            
            append_values = [Canti.k, Canti.q, Canti.drive_freq/(2*np.pi), tau]
            data_noisy = np.append(data_mono_noisy, append_values)
            
            disp_string = save_path + '/0noise/disp_' + str(h) + '_' + str(k) + '_'+ str(j) + '.npy'
            
            np.save(disp_string, data_noisy)
            
            #print('Finished generating', i, 'us series at ', datetime.datetime.now().time())
            
        k+=1
        print('Finished generating', tau, 'us series at ', datetime.datetime.now().time())
        
    h+=1
    print('Finished generating', params_path, 'series at ', datetime.datetime.now().time())





#outputdata = pd.DataFrame({'Tau':list_of_taus,
#                           'Tfp':list_of_tfps})
#outputdata.to_csv('C:/Users/jakeprecht/DDHO/csv_0pt1_percent_noisy_10us.csv', index=False)       

#outputdata_noisier = pd.DataFrame({'Tau':list_of_taus,
#                           'Tfp':list_of_tfps_noisier})
#outputdata_noisier.to_csv('C:/Users/jakeprecht/DDHO/csv_1_percent_noisy_10us.csv', index=False)

t1 = time.time()
part_1_time = int(t1-t0)
print('Generation took ', part_1_time, 's to run')