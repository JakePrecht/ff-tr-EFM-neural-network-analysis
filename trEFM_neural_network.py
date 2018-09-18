
# coding: utf-8

# In[1]:


#Import necessary libraries

import pandas as pd
import numpy as np
import glob
import os
from igor.binarywave import load as loadibw
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LeakyReLU, Conv1D, MaxPooling1D, Flatten, Merge
from keras.utils import np_utils
from keras import callbacks
from keras import metrics
from keras import backend as K
from sklearn.utils import shuffle
from matplotlib import pyplot as plt


class CNN_train():
    def load_simulated_train_data(self, *paths):
        #Example train_path: "D:/jake/DDHO Data/displacement/10us/random_noise_10/*.npy"
        
        #init df for collecting each run
        df_main = pd.DataFrame()
        
        for i in paths:
            
            #grab a list of strings to every file path from the train_path folder
            file_path = glob.glob(i)
        
            test_pixel = np.load(file_path[0])
            #this will be used in for loop below
            #it is just used to grab the length of any one displacement curve
        
            #initialize DataFrame column names list (name for each feature point of NN)
            columns=[]
            #for i in range(len(test_pixel)-2):#-2 because tfp and tau are at end of pixel data
            #for j in range(len(test_pixel)-1):#-1 because tau is at end of pixel data
            for j in range(len(test_pixel)-4):#-4 because k,Q,omega,tau is at end of pixel data
                columns.append('t='+str(j))   #most columns are just time points and I'm making them here
    
            #columns.append('Tfp') #because tfps are appended onto the 2nd to last column of my inst_freq data.  Uncomment for Inst_freq data!
            columns.append('k')
            columns.append('Q')
            columns.append('omega')
            columns.append('Tau') #because taus are appended onto the end of my data
    
    
            #load all of the data into an array for input into DataFrame
            #each entry in the "data1" array is a numpy array of a displacement curve
            data1 = [np.load(file_path[i]) for i in range(0,(len(file_path)))]
    
            #make df for output
            train_data = pd.DataFrame(data=data1,columns=columns,dtype=np.float64)
            train_data = train_data.drop('t=0',axis=1)
            #these t=0 points end up as just NaNs after preprocessing and are useless anyways because by definition freq shift is 0 at trigger for all points
            #dropping t=0 maybe unnecessary with displacement data?
            
            df_main = pd.concat([df_main,train_data], ignore_index=True) #append each run to the final collection DataFrame
        
        df_main = shuffle(df_main) #shuffle data because it is currently ordered and this could impact NN learning
        
        return df_main
    
    def load_experimental_train_data(self, *paths):
        """Path should be to Runx folder:
        E.g.
        path = "C:/Users/Jake/Desktop/75khz RSI data/Run1" for my pc or
        path = "D:/Jake/DDHO data/durmus_data/75khz RSI data/Run1" for simulation computer"""
        #taus = np.array([10e-9, 25e-9, 50e-9, 100e-9, 250e-9, 500e-9,
        #                        1e-6, 5e-6, 10e-6, 100e-6, 1e-3])
        taus= np.array([10e-9, 31.62e-9, 100e-9, 316.2e-9, 1e-6, 3.162e-6, 10e-6, 31.62e-6, 100e-6, 316.2e-6, 1e-3])
        #below code assumes data array has trigger at 16384 and total points 16384*2 (this is what all my and Durmus' data is)
        
        #init df for collecting each run
        df_main = pd.DataFrame()
        
        for i in paths:
            
            tau_paths = glob.glob(i + "/*")
            sorted_tau_paths = sorted(tau_paths, key = lambda x: int(os.path.basename(os.path.normpath(x))))
            #Above line properly sorts so that they are ordered 0,1,2,3,4,5,6,7,8,9,10 rather than 0,1,10,2,3,4...
    
            #Init dataframe for this run
            df_run = pd.DataFrame()
            
            #Loop through every tau in that run
            for j in range(4,len(tau_paths)):
                tau = taus[j]
                displacement_path = glob.glob(sorted_tau_paths[j]+"/*")
            
                #collect relevent parts of the real data into df for that run
                for k in range(len(displacement_path)):
                    disp_array = loadibw(displacement_path[k])['wave']['wData'] #load displacement from .ibw
                    #throw away all displacement before the trigger (16384 pre-trigger points)
                    disp_array = disp_array[16384:,:]
                    disp_array = np.transpose(disp_array)
        
                    #Put loaded stuff into dataframe and label tau
                    columns=[]
                    for l in range(disp_array.shape[1]):
                        columns.append('t='+str(l))
            
                    df_temp = pd.DataFrame(data=disp_array, columns=columns)
                    df_temp['Tau'] = pd.Series(index=df_temp.index) #create Tau column
                    df_temp['Tau'] = tau #assign tau value to tau column (could probably be done in above step with data=tau?)
                    
                    df_run = df_run.append(df_temp,ignore_index=True) #append each tau value to this run
                    #df_run = pd.concat([df_run,df_temp],ignore_index=True)
            
            
            df_main = pd.concat([df_main,df_run], ignore_index=True) #append each run to the final collection DataFrame
            
        
        df_main = shuffle(df_main) #shuffle data because it is currently ordered and this could impact NN learning
        
        return df_main
    
        
    
    def preprocess_train_data(self,train_data):
        #Prep training data

        num_samples = len(train_data)
        train_x = train_data[0:num_samples+1] #this syntax is an artifact from when I was training with partial data sets but it doesn't really add much timing loss so I'm leaving it in case I need to change it again
        
        train_x1 = train_x.drop(['k','Q','omega','Tau'],axis=1)
        
        train_x2 = train_x[['k','Q','omega']]
        
        #train_x = train_x.drop('Tau',axis=1) #dropping Tau because we do not input Tau to the neural network (that's like giving it the solution and then asking for the solution--it cheats)
        #train_x = train_x.drop('Tfp',axis=1) #for simulations only
        
        self.mean_train_x1 = np.mean(train_x1) #saving the mean_train_x for preprocessing the test data in the same manner as our training dat
        self.mean_train_x2 = np.mean(train_x2)
        
        self.SD_train_x1 = np.std(train_x1) #saving the SD_train_x for preprocessing the test data in the same manner as our training data
        self.SD_train_x2 = np.std(train_x2)
        
        train_x1_norm = (train_x1 - self.mean_train_x1) /  (self.SD_train_x1) #normalize and centralize the training data for best neural network performance
        train_x1_norm_reshaped = np.expand_dims(train_x1_norm,axis=2) #formatting for input into CNN
        
        train_x2_norm = (train_x2 - self.mean_train_x2) /  (self.SD_train_x2) #normalize and centralize the training data for best neural network performance
        train_x2_norm_reshaped = np.expand_dims(train_x2_norm,axis=2) #formatting for input into CNN
        
        train_y = np.array(train_data['Tau']) #labeled, true Tau values for the CNN to learn from
        train_y = train_y[0:num_samples+1] #this syntax is an artifact from when I was training with partial data sets but it doesn't really add much timing loss so I'm leaving it in case I need to change it again
        
        #Label encode the y-data as preprocessing for one hot-encoding for classification NN:

        #tau_index is used to recover the original tau's from a one-hot encoded output.
        #e.g. tau = [10, 100, 1000, 10, 10] then
        #unique_tau = [10, 100, 1000]
        #tau_index = [0,1,2,0,0] is index of tau to corresponding unique_tau so
        #unique_tau[tau_index] == tau 
        unique_tau, tau_index = np.unique(train_y,return_inverse=True)

        #make one-hot encoded tau vector
        one_hot_tau = np_utils.to_categorical(tau_index)

        self.number_of_classes = one_hot_tau.shape[1] #used to match number of output Softmax layers in my NN
        
        return train_x1_norm_reshaped, train_x2_norm_reshaped, one_hot_tau
    
    def train_CNN(self, train_x1, train_x2, train_y, num_epochs = 40, kernel1_size = 400, kernel2_size = 100, num_filter1 = 5, num_filter2 = 3):
        #Build CNN and start training!

        self.filter_number1 = num_filter1
        self.filter_number2 = num_filter2
        self.kernel1_size = kernel1_size
        self.kernel2_size = kernel2_size
        
        
        
        #Initialize CNN branch 1 for main convolutional input data (displacement or instantaneous frequency)
        branch1 = Sequential()

        #Add convolution layers
        branch1.add(Conv1D(filters=num_filter1,kernel_size=kernel1_size,strides=2,padding='same',input_shape=(train_x1.shape[1],1)))
        branch1.add(LeakyReLU(alpha=0.01))
        branch1.add(MaxPooling1D())

        branch1.add(Conv1D(filters=num_filter2,kernel_size=kernel2_size,strides=2,padding='same'))
        branch1.add(LeakyReLU(alpha=0.01))
        branch1.add(MaxPooling1D())

        branch1.add(Flatten())
        #Roughly 500 units length of branch 1 (8000 displacement points / (2**4 because each strides = 2 and each maxpool halves data length))
        
        branch1.add(Dense(units=100, kernel_initializer='he_normal',activation='linear'))
        branch1.add(LeakyReLU(alpha=.01))
        branch1.add(Dropout(0.3))

        branch1.add(Dense(units=100, kernel_initializer='he_normal',activation='linear'))
        branch1.add(LeakyReLU(alpha=.01))
        branch1.add(Dropout(0.4))
        
        
        
        #Initialize CNN branch 2 for supplementary data (Q, k, and omega)
        branch2 = Sequential()

        #Add supplementary data inputs
        branch2.add(Dense(units=100, kernel_initializer='he_normal', activation='linear', input_shape=(train_x2.shape[1],1)))
        branch2.add(LeakyReLU(alpha=.01))
        branch2.add(Dropout(0.3))
        
        branch2.add(Dense(units=100, kernel_initializer='he_normal', activation='linear'))
        branch2.add(LeakyReLU(alpha=.01))
        branch2.add(Dropout(0.4))
        
        branch2.add(Flatten())
        
        
        
        #Merge branches 1 and 2
        model = Sequential()
        model.add(Merge([branch1,branch2], mode='concat'))

        
        #Add final fully connected layers
        model.add(Dense(units=100, kernel_initializer='he_normal', activation='linear'))
        model.add(LeakyReLU(alpha=.01))
        model.add(Dropout(0.3))

        model.add(Dense(units=100, kernel_initializer='he_normal', activation='linear'))
        model.add(LeakyReLU(alpha=.01))
        model.add(Dropout(0.4))

        #Add classification layer
        model.add(Dense(units=self.number_of_classes, activation='softmax'))


        #Compile CNN and configure metrics/learning process
        
        
        """below functions are failure metrics that tell me if the true tau was in the top 2, top 3, or top 5 guesses made by the neural network"""
        def inTop2(k=2):
            def top2metric(y_true,y_pred):
                return metrics.top_k_categorical_accuracy(y_true,y_pred,k=2)
            return top2metric
        
        def inTop3(k=3):
            def top3metric(y_true,y_pred):
                return metrics.top_k_categorical_accuracy(y_true,y_pred,k=3)
            return top3metric
        
        def inTop5(k=5):
            def top5metric(y_true,y_pred):
                return metrics.top_k_categorical_accuracy(y_true,y_pred,k=5)
            return top5metric
        
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy', inTop2(), inTop3()])

        #Prepare for visualization
        #tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        #tbCallBack = callbacks.TensorBoard(log_dir="logs/{}".format(time.time()))

        #Train model
        model.fit([train_x1, train_x2], train_y, batch_size=32, epochs=num_epochs,verbose=2, validation_split=0.05)#, callbacks=[tbCallBack])
        self.model = model #save model to self for calling from other functions later
        return
    
    def load_simulated_test_data(self, file_path):#, test_fraction):
        """input string with file path (e.g. "D:/jake/DDHO Data/inst_freq/25us/0pt1noise/*.npy" )
        and also input the fraction of the data you wish to test (e.g. testing 10% of data would be 0.1)
    
        outputs the requested percentage of the data (test_data_norm_reshaped) and their corresponding labels for evaluation (one_hot_tau)
        these outputs are basically test_x and test_y that are formatted to be fed into model.evaluate()
    
        NOTE this function requires previous cells to have been run (train_x must exist!!!)"""
        file_path1 = glob.glob(file_path)
        test_pixel = np.load(file_path1[0])
        #this will be used in for loop below
        #it is just used to grab the length of any one inst. freq. curve
    
        columns2=[]
        for i in range(len(test_pixel)-4):#-4 because k,Q,omega, tau is at end of pixel data
        #for i in range(len(test_pixel)-2):#-2 because tfp and tau are at end of pixel data
            columns2.append('t='+str(i))#most columns are just time points and I'm making them here
    
        #columns2.append('Tfp') #because tfps are appended onto the end of my inst_freq data
        columns2.append('k')
        columns2.append('Q')
        columns2.append('omega')
        columns2.append('Tau') #because taus are appended onto the end of my inst_freq data
    
        #alternate way to load only a fraction of the data to save memory and time
        #num_samples = len(file_path1)
        #num_buckets = self.number_of_classes
        #bucket_range = int(num_samples/num_buckets)
        #test_fraction_range = int(test_fraction * bucket_range)
        #load_list = []

        #for i in range(num_buckets):
        #    for j in range(test_fraction_range):
        #        load_list.append(int((i * bucket_range) + (j)))
    
        #data1 = [np.load(file_path1[i]) for i in load_list]
        data1 = [np.load(file_path1[i]) for i in range(len(file_path1))]

        #make df for output
        test_data = pd.DataFrame(data=data1,columns=columns2,dtype=np.float64)
        test_data = test_data.drop('t=0',axis=1)
        #these t=0 points end up as just NaNs and are useless anyways because by definition freq shift is 0 at trigger
        test_data = shuffle(test_data)
        #shuffle data because it is currently ordered and this could impact NN learning

        test_y = np.array(test_data['Tau'])
        #Label encode the y-data as preprocessing for one hot-encoding for classification NN:
        unique_tau, tau_index = np.unique(test_y,return_inverse=True)
        #make one-hot encoded tau vector
        one_hot_tau = np_utils.to_categorical(tau_index)
    
        #preprocess test_x
        test_x1 = test_data.drop(['k','Q','omega','Tau'],axis=1)
        test_x2 = test_data[['k','Q','omega']]
        
        #test_data_norm = (test_data - self.mean_train_x ) /  (self.SD_train_x) #important to preprocess my test data same as my train data!!!!
        #test_data_norm_reshaped = np.expand_dims(test_data_norm,axis=2)
        
        test_x1_norm = (test_x1 - self.mean_train_x1 ) /  (self.SD_train_x1) #important to preprocess my test data same as my train data!!!!
        test_x2_norm = (test_x2 - self.mean_train_x2 ) /  (self.SD_train_x2) #important to preprocess my test data same as my train data!!!!
        
        test_x1_norm_reshaped = np.expand_dims(test_x1_norm,axis=2)
        test_x2_norm_reshaped = np.expand_dims(test_x2_norm,axis=2)
        
        
        return test_x1_norm_reshaped, test_x2_norm_reshaped, one_hot_tau
    
    
    def load_experimental_test_data(self, path):
        """input string with file path
    
        outputs the requested percentage of the data (test_data_norm_reshaped) and their corresponding labels for evaluation (one_hot_tau)
        these outputs are basically test_x and test_y that are formatted to be fed into model.evaluate()
    
        NOTE this function requires previous cells to have been run (train_x must exist!!!)"""
        #taus = np.array([10e-9, 25e-9, 50e-9, 100e-9, 250e-9, 500e-9,
        #                        1e-6, 5e-6, 10e-6, 100e-6, 1e-3])
        taus= np.array([10e-9, 31.62e-9, 100e-9, 316.2e-9, 1e-6, 3.162e-6, 10e-6, 31.62e-6, 100e-6, 316.2e-6, 1e-3])
        
        
        tau_paths = glob.glob(path + "/*")
        sorted_tau_paths = sorted(tau_paths, key = lambda x: int(os.path.basename(os.path.normpath(x))))
        #Above line properly sorts so that they are ordered 0,1,2,3,4,5,6,7,8,9,10 rather than 0,1,10,2,3,4...
    
        #Init dataframe for this run
        df = pd.DataFrame()
        
        for j in range(4,len(tau_paths)):
            tau = taus[j]
            displacement_path = glob.glob(sorted_tau_paths[j]+"/*")
            
            #print(displacement_path)
            #print(j)
            #print('the above two should be matching indices')
            
            disp_array = loadibw(displacement_path[0])['wave']['wData']
            #throw away all displacement before the trigger
            disp_array = disp_array[16384:,:]
            disp_array = np.transpose(disp_array)
        
            #Put loaded stuff into dataframe and label tau
            columns=[]
            for k in range(disp_array.shape[1]):
                columns.append('t='+str(k))
        
            df_temp = pd.DataFrame(data=disp_array, columns=columns)
            df_temp['Tau'] = pd.Series(index=df_temp.index)
            df_temp['Tau'] = tau
            df = df.append(df_temp,ignore_index=True)
       
        """for j in range(len(tau_paths)):
            tau = taus[j]
            displacement_path = glob.glob(sorted_tau_paths[j]+"/*")
            
            print(displacement_path)
            #print(j)
            #print('the above two should be matching indices')
            
            for i in range(len(displacement_path)):
                disp_array = loadibw(displacement_path[i])['wave']['wData']
                #throw away all displacement before the trigger
                disp_array = disp_array[16384:,:]
                disp_array = np.transpose(disp_array)
        
                #Put loaded stuff into dataframe and label tau
                columns=[]
                for k in range(disp_array.shape[1]):
                    columns.append('t='+str(k))
        
                df_temp = pd.DataFrame(data=disp_array, columns=columns)
                df_temp['Tau'] = pd.Series(index=df_temp.index)
                df_temp['Tau'] = tau
                df = df.append(df_temp,ignore_index=True)
            
        df = shuffle(df) #shuffle data because it is currently ordered and this could impact NN learning
        """
        
        df = shuffle(df) #shuffle data because it is currently ordered and this could impact NN learning
        
        test_y = np.array(df['Tau'])
        #Label encode the y-data as preprocessing for one hot-encoding for classification NN:
        unique_tau, tau_index = np.unique(test_y,return_inverse=True)
        #make one-hot encoded tau vector
        one_hot_tau = np_utils.to_categorical(tau_index)
    
        #preprocess test_x
        df = df.drop('Tau',axis=1)
        #test_data = test_data.drop('Tfp',axis=1)
        df_norm = (df - self.mean_train_x ) /  (self.SD_train_x) #important to preprocess my test data same as my train data!!!!
        df_norm_reshaped = np.expand_dims(df_norm,axis=2)
    
        return df_norm_reshaped, one_hot_tau
    
    
    def load_experimental_test_data_averaged(self, path):
        """input string with file path
    
        outputs the requested percentage of the data (test_data_norm_reshaped) and their corresponding labels for evaluation (one_hot_tau)
        these outputs are basically test_x and test_y that are formatted to be fed into model.evaluate()
    
        NOTE this function requires previous cells to have been run (train_x must exist!!!)"""
        #This function is unused and not useful
        #taus = np.array([10e-9, 25e-9, 50e-9, 100e-9, 250e-9, 500e-9,
        #                        1e-6, 5e-6, 10e-6, 100e-6, 1e-3])
        taus= np.array([10e-9, 31.62e-9, 100e-9, 316.2e-9, 1e-6, 3.162e-6, 10e-6, 31.62e-6, 100e-6, 316.2e-6, 1e-3])
        
        
        tau_paths = glob.glob(path + "/*")
        sorted_tau_paths = sorted(tau_paths, key = lambda x: int(os.path.basename(os.path.normpath(x))))
        #Above line properly sorts so that they are ordered 0,1,2,3,4,5,6,7,8,9,10 rather than 0,1,10,2,3,4...
    
        #Init dataframe for this run
        df = pd.DataFrame()
        
        for j in range(len(tau_paths)):
            tau = taus[j]
            displacement_path = glob.glob(sorted_tau_paths[j]+"/*")
            
            #print(displacement_path)
            #print(j)
            #print('the above two should be matching indices')
            
            disp_array = loadibw(displacement_path[0])['wave']['wData']
            #throw away all displacement before the trigger
            disp_array = disp_array[16384:,:]
            disp_array = np.transpose(disp_array)
        
            #Put loaded stuff into dataframe and label tau
            columns=[]
            for k in range(disp_array.shape[1]):
                columns.append('t='+str(k))
        
            df_temp = pd.DataFrame(data=disp_array, columns=columns)
            df_temp['Tau'] = pd.Series(index=df_temp.index)
            df_temp['Tau'] = tau
            
            df_temp = df_temp.mean(axis=0)
            
            df = df.append(df_temp,ignore_index=True)
        
        #df = shuffle(df) #shuffle data because it is currently ordered and this could impact NN learning
        
        test_y = np.array(df['Tau'])
        #Label encode the y-data as preprocessing for one hot-encoding for classification NN:
        unique_tau, tau_index = np.unique(test_y,return_inverse=True)
        #make one-hot encoded tau vector
        one_hot_tau = np_utils.to_categorical(tau_index)
    
        #preprocess test_x
        df = df.drop('Tau',axis=1)
        #test_data = test_data.drop('Tfp',axis=1)
        df_norm = (df - self.mean_train_x ) /  (self.SD_train_x) #important to preprocess my test data same as my train data!!!!
        df_norm_reshaped = np.expand_dims(df_norm,axis=2)
    
        return df_norm_reshaped, one_hot_tau
    
    
    def test_closeness(self, test_x1, test_x2, test_y):
        """This function looks at the predicted tau values from model.predict(test_x) and compares them 
        to the true tau values from test_y.  
        It then returns three values telling you what percentage of the incorrect predictions varied by spacing of
        one tau value, two tau values, or three tau values.
        
        E.G.
        Say possible taus = [1,2,3,4,5,6,7,8,9]
        model.predict(test_x) = [2,2,2,3,3,3,4,4,5,6]
        test_y = [2,2,2,2,2,2,2,2,2,2]
        
        test_closeness returns [0.3,0.2,0.1]
        because 30% of the predictions varied by one tau value (tau = 2 but 3 times it guessed tau = 3)
        because 20% of the predictions varied by two tau values (tau = 2 but 2 times it guessed tau = 4)
        because 10% of the predictions varied by three tau values (tau = 2 but 1 time it guessed tau = 5)
        """
        
        pred_tau = self.model.predict([test_x1,test_x2],verbose=0)
        
        pred_tau_am = pred_tau.argmax(axis=-1) #pluck out actual prediction value!
        test_y_am = test_y.argmax(axis=-1) #does not actually need argmax, but this makes it same format as pred_tau_am
        
        incorrect_indices = np.nonzero(pred_tau_am != test_y_am) #indices of incorrect predictions
        
        total_samples = len(pred_tau)
        total_fails = len(incorrect_indices[0])
        
        #init diff collection variables (how many tau values away the true value was from the predicted value)
        num_diff_1 = 0
        num_diff_2 = 0
        num_diff_3 = 0
        num_greater = 0
        
        #init array for seeing which taus it is bad at predicting
        #which_taus_failed = np.zeros(11) #CHANGE THIS HARD-CODED 11 TO WHATEVER THE NUMBER OF CLASSES IS!!!
        which_taus_failed = np.zeros(self.number_of_classes) #CHANGE THIS HARD-CODED 11 TO WHATEVER THE NUMBER OF CLASSES IS!!!
        
        for element in incorrect_indices[0]:
            
            #collect diff (how many tau values away the true value was from the predicted value)
            diff = abs(pred_tau_am[element] - test_y_am[element])
            if diff == 1:
                num_diff_1 += 1
            elif diff == 2:
                num_diff_2 += 1
            elif diff == 3:
                num_diff_3 += 1
            else:
                num_greater += 1
            
            #collect how many of each tau failed
            i=0
            while True:
                if test_y_am[element] == i:
                    which_taus_failed[i] += 1
                    break
                else:
                    i += 1
                    
            which_taus_failed_percent = np.round((which_taus_failed / total_fails),4) * 100

        
        percent_num_diff_1 = round((num_diff_1 / total_samples), 4) * 100
        percent_num_diff_2 = round((num_diff_2 / total_samples), 4) * 100
        percent_num_diff_3 = round((num_diff_3 / total_samples), 4) * 100
        percent_num_diff_greater = round((num_greater / total_samples), 4) * 100
            
        #Next section is for debugging purposes
        #percent_incorrect = (len(incorrect_indices[0])/total_samples)
        #percent_incorrect_calculated = percent_num_diff_1 + percent_num_diff_2 + percent_num_diff_3 + percent_num_diff_greater
        #print('percent incorrect should be ' + str(percent_incorrect))
        #print('percent incorrect calculated is ' + str(percent_incorrect_calculated))
        
        return percent_num_diff_1, percent_num_diff_2, percent_num_diff_3, which_taus_failed#_percent
        
        
        
    
    def test_simulated_CNN(self, *paths):
        
        score_string = 'data order is testing against '
        for element in paths:
            score_string += (str(element) + " , \n") 

        
        score_collect = [score_string]
        score_collect.append('column order is loss, accuracy, top2metric, top3metric')
        score_collect.append('top2metric = % that the true tau was one of the top 2 predictions')
        score_collect.append(' ')
        
        
        #score_collect = ['data order is no_noise, 0pt1noise, 1noise, random_noise_1, random_noise_10']
        #score_collect.append('first column is loss, second column is accuracy')
        
        for i in paths:
            test_x1, test_x2, test_y = self.load_simulated_test_data(i)#,test_fraction)
            score = self.model.evaluate([test_x1,test_x2],test_y, batch_size=32)
            percentage = str(round(score[1],5) * 100)
            print('model scored ' + percentage + '% on ' + str(i))
            score_collect.append(str(score))
            
            error1, error2, error3, which_taus_failed = self.test_closeness(test_x1,test_x2,test_y)
            score_collect.append('one_diff_error = ' + str(error1))
            score_collect.append('two_diff_error = ' + str(error2))
            score_collect.append('three_diff_error = ' + str(error3))
            score_collect.append('which taus failed were: ' + str(which_taus_failed))
            print('one_diff_error = ' + str(error1))
            print('two_diff_error = ' + str(error2))
            print('three_diff_error = ' + str(error3))
            print('which taus failed were: ' + str(which_taus_failed))
            print(' ')
            
            score_collect.append('above scores were for ' + str(element)) #new code on 7/2/18
            score_collect.append(' ')
        
        self.score_collect = score_collect
        
        return
    
    def test_experimental_CNN(self, *paths):
        
        score_string = 'data order is testing against '
        for element in paths:
            score_string += (str(element) + " , \n") 

        
        score_collect = [score_string]
        score_collect.append('column order is loss, accuracy, top2metric, top3metric')
        score_collect.append('top2metric = % that the true tau was one of the top 2 predictions')
        score_collect.append(' ')
        
        for element in paths:
            test_x1, test_x2, test_y = self.load_experimental_test_data(element)
            score = self.model.evaluate([test_x1,test_x2],test_y,batch_size = 32)
            percentage = str(round(score[1],5) * 100)
            print('model scored ' + percentage + '% on ' + str(element))
            score_collect.append(str(score))
            #score_collect.append('above score was for ' + str(element)) #new code on 7/2/18
            
            error1, error2, error3, which_taus_failed = self.test_closeness(test_x,test_y)
            score_collect.append('one_diff_error = ' + str(error1))
            score_collect.append('two_diff_error = ' + str(error2))
            score_collect.append('three_diff_error = ' + str(error3))
            score_collect.append('which taus failed were: ' + str(which_taus_failed))
            print('one_diff_error = ' + str(error1))
            print('two_diff_error = ' + str(error2))
            print('three_diff_error = ' + str(error3))
            print('which taus failed were: ' + str(which_taus_failed))
            print(' ')
            
            score_collect.append('above scores were for ' + str(element)) #new code on 7/2/18
            score_collect.append(' ')
            
        
        self.score_collect = score_collect
        
        return
    
    def test_experimental_CNN_averaged(self, *paths):
        
        score_string = 'data order is testing against '
        for element in paths:
            score_string += (str(element) + " , \n") 

        
        score_collect = [score_string]
        score_collect.append('column order is loss, accuracy, top2metric, top3metric')
        score_collect.append('top2metric = % that the true tau was one of the top 2 predictions')
        score_collect.append(' ')
        
        for element in paths:
            test_x, test_y = self.load_experimental_test_data_averaged(element)
            score = self.model.evaluate(test_x,test_y,batch_size = 32)
            percentage = str(round(score[1],5) * 100)
            print('model scored ' + percentage + '% on ' + str(element))
            score_collect.append(str(score))
            #score_collect.append('above score was for ' + str(element)) #new code on 7/2/18
            
            #more new code currently testing
            error1, error2, error3, which_taus_failed = self.test_closeness(test_x,test_y)
            score_collect.append('one_diff_error = ' + str(error1))
            score_collect.append('two_diff_error = ' + str(error2))
            score_collect.append('three_diff_error = ' + str(error3))
            score_collect.append('which taus failed were: ' + str(which_taus_failed))
            print('one_diff_error = ' + str(error1))
            print('two_diff_error = ' + str(error2))
            print('three_diff_error = ' + str(error3))
            print('which taus failed were: ' + str(which_taus_failed))
            print(' ')
            
            score_collect.append('above scores were for ' + str(element)) #new code on 7/2/18
            score_collect.append(' ')
            
        
        self.score_collect = score_collect
        
        return
    
    def save_CNN(self, save_str):
        #save model and test evaluation outputs
        #example save_str: save_str = 'displacement_10us_random_noise_10_2018_06_13_80epoch'
        #requires test_CNN to have been run already
        path = 'C:/Users/jakeprecht/DDHO/saved CNN models/'
        save_str_h5 = path + save_str + '.h5'
        save_str_txt = path + save_str + '_results.txt'
        save_str_weights = path + save_str + '_weights.h5'
        
        self.model.save(save_str_h5)  # creates a HDF5 file 'my_model.h5'
        self.model.save_weights(save_str_weights)
        
        output_scores = open(save_str_txt, 'w')
        for item in self.score_collect:
            output_scores.write("%s\n" % item)
        
        return
    
    def visualize_weights(self, layer_number):
        #layer number 0 = conv layer 1
        #layer number 1 = ReLU 1"""
        weights, biases = self.model.layers[layer_number].get_weights()
        
        if layer_number == 0 or 1:
            number_filters = self.filter_number1
            kernel_length = self.kernel1_size         

        #elif layer_number == 3:
        #    number_filters = self.filter_number2
        #    kernel_length = self.kernel2_size
        
        else:
            raise ValueError("Input for layer_number must be 0 or 3 in current implementation (2018_08_08)")
            
        fig = plt.figure()
        for i in range(number_filters):
            weight_plt = weights[:,:,i]
            weight_plt2 = weight_plt.reshape((kernel_length,))
            #ax = fig.add_subplot(number_filters,1,i+1)
            plt.figure()
            plt.plot(weight_plt2)
            #ax.imshow(weight_plt2,cmap='gray')
            
        return
        
        
    def layer_to_visualize(self, layer, img_to_visualize):
        """img_to_visualize = train_x[image_number]
        this code does not work yet
        """
        layer = self.model.layers[layer]
        img_to_visualize = np.expand_dims(img_to_visualize, axis=0)
        
        inputs = [K.learning_phase()] + self.model.inputs

        _convout1_f = K.function(inputs, [layer.output])
        def convout1_f(X):
            # The [0] is to disable the training phase flag
            return _convout1_f([0] + [X])

        convolutions = convout1_f(img_to_visualize)
        convolutions = np.squeeze(convolutions)

        print ('Shape of conv:', convolutions.shape)
    
        n = convolutions.shape[0]
        n = int(np.ceil(np.sqrt(n)))
    
        # Visualization of each filter of the layer
        fig = plt.figure(figsize=(12,8))
        for i in range(len(convolutions)):
            ax = fig.add_subplot(n,n,i+1)
            ax.imshow(convolutions[i], cmap='gray')
            
        return
    
    
#number_of_classes = one_hot_tau.shape[1] #used to match number of output Softmax layers in my NN
#train_x_norm_reshaped = np.expand_dims(train_x_norm,axis=2) #formatting for input into CNN


# In[ ]:


#below three cells are for quick testing of code (load one run, train one epoch, test two runs)


# In[5]:


#load and prep training data
test = CNN_train()
train_data = test.load_simulated_train_data("D:/jake/DDHO Data/displacement/2018_09_17/0noise/*.npy")
#train_data = test.load_experimental_train_data("D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run5",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run10",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run15",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run20",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run30")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V5/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25")
train_x1, train_x2, train_y = test.preprocess_train_data(train_data)


# In[6]:


model = test.train_CNN(train_x1,train_x2,train_y, num_epochs = 20, kernel1_size = 400, kernel2_size = 100, num_filter1 = 5, num_filter2 = 3)


# #load and test synthetic test data.  also save model and results
# test.test_CNN(test_fraction=0.1)
# test.save_CNN('modular_CNN_Durmus_data_2018_06_21')

# In[10]:


#load and test test data.  also save model and results

test.test_simulated_CNN("D:/jake/DDHO Data/displacement/2018_09_17/test2/0noise/*.npy")
#test.test_experimental_CNN("D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run5")
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run10",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run15",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run20",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run3",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run8",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run22",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run28",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run30")
test.save_CNN('2018_09_17 simulated displacement train 0noise more complex architecture test 2018_09_12 tip1')


# In[11]:


#load and test test data.  also save model and results

test.test_simulated_CNN("D:/jake/DDHO Data/displacement/2018_09_17/test/0noise/*.npy")
#test.test_experimental_CNN("D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run5")
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run10",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run15",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run20",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run3",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run8",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run22",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run28",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run30")
test.save_CNN('2018_09_17 simulated displacement train 0noise more complex architecture test 2018_09_17 tip6')


# In[9]:


test.visualize_weights(0)


# In[ ]:


#load and prep training data
test = CNN_train()
train_data = test.load_simulated_train_data("D:/jake/DDHO Data/displacement/2 per decade 10 ns to 1 ms slow only/random_noise_1/*.npy")
#train_data = test.load_experimental_train_data("D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run5",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run10",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run15",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run20",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run30")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V5/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25")
train_x, train_y = test.preprocess_train_data(train_data)


# In[ ]:


model = test.train_CNN(train_x,train_y, num_epochs = 20, kernel1_size = 400, kernel2_size = 100, num_filter1 = 5, num_filter2 = 3)


# In[ ]:


#load and test test data.  also save model and results

test.test_simulated_CNN("D:/jake/DDHO Data/displacement/2 per decade 10 ns to 1 ms slow only/no_noise/*.npy",
                       "D:/jake/DDHO Data/displacement/2 per decade 10 ns to 1 ms slow only/0pt1noise/*.npy",
                       "D:/jake/DDHO Data/displacement/2 per decade 10 ns to 1 ms slow only/1noise/*.npy",
                       "D:/jake/DDHO Data/displacement/2 per decade 10 ns to 1 ms slow only/random_noise_1/*.npy",
                       "D:/jake/DDHO Data/displacement/2 per decade 10 ns to 1 ms slow only/random_noise_10/*.npy")
#test.test_experimental_CNN("D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run5")
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run10",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run15",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run20",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run3",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run8",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run22",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run28",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run30")
test.save_CNN('2018_08_23 simulated displacement train rand noise 1 slow time only')


# In[ ]:


test.visualize_weights(0)


# In[ ]:


#load and prep training data
test = CNN_train()
train_data = test.load_simulated_train_data("D:/jake/DDHO Data/inst_freq/2 per decade 10 ns to 1 ms 300 kHz/no_noise/*.npy")
#train_data = test.load_experimental_train_data("D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run5",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run10",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run15",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run20",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run30")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V5/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25")
train_x, train_y = test.preprocess_train_data(train_data)


# In[ ]:


model = test.train_CNN(train_x,train_y, num_epochs = 20, kernel1_size = 400, kernel2_size = 100, num_filter1 = 5, num_filter2 = 3)


# In[ ]:


#load and test test data.  also save model and results

test.test_simulated_CNN("D:/jake/DDHO Data/inst_freq/2 per decade 10 ns to 1 ms 300 kHz/0pt1noise/*.npy",
                       "D:/jake/DDHO Data/inst_freq/2 per decade 10 ns to 1 ms 300 kHz/1noise/*.npy",
                       "D:/jake/DDHO Data/inst_freq/2 per decade 10 ns to 1 ms 300 kHz/random_noise_1/*.npy",
                       "D:/jake/DDHO Data/inst_freq/2 per decade 10 ns to 1 ms 300 kHz/random_noise_10/*.npy",
                       "D:/jake/DDHO Data/inst_freq/2 per decade 10 ns to 1 ms 300 kHz/no_noise/*.npy")
#test.test_experimental_CNN("D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run5")
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run10",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run15",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run20",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run3",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run8",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run22",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run28",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run30")
test.save_CNN('2018_08_29 simulated inst_freq train no_noise 300 kHz 400 100 KL')


# In[ ]:


test.visualize_weights(0)


# In[ ]:


img_to_vis = train_x[0]


# In[ ]:


print(img_to_vis.shape)


# In[ ]:


test.layer_to_visualize(1, img_to_vis)


# In[ ]:


test.visualize_weights(3)


# In[ ]:


#load and prep training data
test = CNN_train()

train_data = test.load_experimental_train_data("D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1",
                                         "D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run5")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run10",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run15",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run20",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run30")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V5/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25")
train_x, train_y = test.preprocess_train_data(train_data)


# In[ ]:


model = test.train_CNN(train_x,train_y, num_epochs = 20, kernel1_size = 400, kernel2_size = 100, num_filter1 = 5, num_filter2 = 3)


# In[ ]:


#load and test experimental test data.  also save model and results
test.test_experimental_CNN("D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1",
                          "D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run5",
                          "D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run10",
                          "D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run15")
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run20",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run3",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run8",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run22",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run28",
                          #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run30")
test.save_CNN('test_save_with_which_taus_failed_metric_2_20_epochs')


# In[ ]:


#load and prep training data
test = CNN_train()
#train_data = test.load_train_data()
train_data = test.load_experimental_train_data("D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run30")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run5",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run10",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run15",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run20",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run30")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V5/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25")
train_x, train_y = test.preprocess_train_data(train_data)


# In[ ]:


model = test.train_CNN(train_x,train_y, num_epochs = 20, kernel1_size = 400, kernel2_size = 100, num_filter1 = 5, num_filter2 = 3)


# #load and test synthetic test data.  also save model and results
# test.test_CNN(test_fraction=0.1)
# test.save_CNN('modular_CNN_Durmus_data_2018_06_21')

# In[ ]:


#load and test experimental test data.  also save model and results
test.test_experimental_CNN("D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run29",
                          "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run30",
                          "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run31")
test.save_CNN('2018_08_14 train 30 test 29 30 31 slow timescale only')


# In[ ]:


#load and prep training data
test = CNN_train()
#train_data = test.load_train_data()
train_data = test.load_experimental_train_data("D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run1",
                                              "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run2",
                                              "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run29",
                                              "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run30")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run5",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run10",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run15",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run20",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run30")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V5/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25")
train_x, train_y = test.preprocess_train_data(train_data)


# In[ ]:


model = test.train_CNN(train_x,train_y, num_epochs = 20, kernel1_size = 400, kernel2_size = 100, num_filter1 = 5, num_filter2 = 3)


# #load and test synthetic test data.  also save model and results
# test.test_CNN(test_fraction=0.1)
# test.save_CNN('modular_CNN_Durmus_data_2018_06_21')

# In[ ]:


#load and test experimental test data.  also save model and results
test.test_experimental_CNN("D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run1",
                           "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run2",
                           "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run3",
                           "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run5",
                           "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run10",
                           "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run15",
                           "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run20",
                           "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run25",
                           "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run28",
                           "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run29",
                           "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run30",
                           "D:/Jake/DDHO data/jake_data/2018_08_14 2 samples per decade BD/Run31")
test.save_CNN('2018_08_15 train 1 2 29 30 test many slow timescales only')


# In[ ]:


#load and test experimental test data.  also save model and results
test.test_experimental_CNN_averaged("D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run5",
                          "D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run6",
                          "D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run7",
                          "D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run8",
                          "D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run9",
                          "D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run10",
                          "D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run20",
                          "D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run30",
                          "D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run40",
                          "D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run43",
                          "D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run44",
                          "D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run45")
test.save_CNN('2018_07_19 train 5_6_7_43_44_45 test averaged')


# In[ ]:


#load and prep training data
test = CNN_train()
#train_data = test.load_train_data()
train_data = test.load_experimental_train_data("D:/Jake/DDHO data/jake_data/2018_08_01 2 samples per decade/Run1",
                                              "D:/Jake/DDHO data/jake_data/2018_08_01 2 samples per decade/Run2",
                                              "D:/Jake/DDHO data/jake_data/2018_08_01 2 samples per decade/Run3")
                                         #"D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run6",
                                         #"D:/Jake/DDHO data/jake_data/2018_07_19 first attempt at voltage pulse/Run7")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run5",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run10",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run15",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run20",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25",
                                        #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run30")
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run1",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V5/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V7/Run25",
                                         #"D:/Jake/DDHO data/durmus_data/voltage scan/BD_V10/Run25")
train_x, train_y = test.preprocess_train_data(train_data)


# In[ ]:


model = test.train_CNN(train_x,train_y, num_epochs = 20, kernel1_size = 400, kernel2_size = 100, num_filter1 = 5, num_filter2 = 3)


# #load and test synthetic test data.  also save model and results
# test.test_CNN(test_fraction=0.1)
# test.save_CNN('modular_CNN_Durmus_data_2018_06_21')

# In[ ]:


#load and test experimental test data.  also save model and results
test.test_experimental_CNN("D:/Jake/DDHO data/jake_data/2018_08_01 2 samples per decade/Run1",
                          "D:/Jake/DDHO data/jake_data/2018_08_01 2 samples per decade/Run2",
                          "D:/Jake/DDHO data/jake_data/2018_08_01 2 samples per decade/Run3",
                          "D:/Jake/DDHO data/jake_data/2018_08_01 2 samples per decade/Run4",
                          "D:/Jake/DDHO data/jake_data/2018_08_01 2 samples per decade/Run5",
                          "D:/Jake/DDHO data/jake_data/2018_08_01 2 samples per decade/Run6")
test.save_CNN('2018_08_01 first test')


# In[ ]:


weights, biases = test.model.layers[3].get_weights()


# In[ ]:


print(weights.shape)


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


weight_plt = weights[:,:,0]
weight_plt.shape
#weight_plt2 = weight_plt.reshape((400,))
#plt.plot(weight_plt2)


# In[ ]:


weights, biases = test.model.layers[1].get_weights()

