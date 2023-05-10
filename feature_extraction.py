from google.colab import drive
drive.mount('/content/drive')
import os

os.chdir('/content/drive/My Drive/laba/all_data/')

from numpy.ma.extras import median
import numpy as np
from matplotlib.patches import Ellipse
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob

def plot_poincare(rr):
   rr_n = rr[:-1]
   rr_n1 = rr[1:]

   sd1 = np.sqrt(0.5) * np.std(rr_n1 - rr_n)
   sd2 = np.sqrt(0.5) * np.std(rr_n1 + rr_n)
   mean = np.mean(rr)
   sd = np.std(rr)
   min_rr = np.min(rr)
   max_rr = np.max(rr)
   max_min = max_rr - min_rr
   q25 = np.quantile(rr, .25)
   q50 = np.quantile(rr, .50)
   q75 = np.quantile(rr, .75)
   return sd1, sd2,sd, mean, min_rr, max_rr, q25, q50, q75, max_min

def feature_extraction(rr):
   mean = np.mean(rr)
   sd = np.std(rr)
   min_rr = np.min(rr)
   max_rr = np.max(rr)
   max_min = max_rr - min_rr
   q25 = np.quantile(rr, .25)
   q50 = np.quantile(rr, .50)
   q75 = np.quantile(rr, .75)
   return sd, mean, min_rr, max_rr, q25, q50, q75, max_min

path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))
#ind = 1 
  
# loop over the list of csv files
for f in csv_files:
      
       # read the csv file
       data = pd.read_csv(f)
       
       #print('File Name:', f.split("\\")[-1])
       data.drop(data[(data['TLX_mean'] == -1) & (data.level.str.contains('rest|post')==False)].index, inplace=True)
       data.dropna(inplace=True)
       
       #-------------------------------------------------------
       data['TLX_mf'] = data['TLX_mental_demand']+data['TLX_frustration']
       conditions1 = [
           (data['TLX_mf'] == -2),
           ((data['TLX_mf'] == 2) | (data['TLX_mf'] == 3) | (data['TLX_mf'] == 4)),
           ((data['TLX_mf'] == 5) | (data['TLX_mf'] == 6) | (data['TLX_mf'] == 7)),
           ((data['TLX_mf'] == 8) | (data['TLX_mf'] == 9) | (data['TLX_mf'] == 10))]
       values1 = [0,1,2,3]
       data['multy'] = np.select(conditions1, values1)

      #-------------- 
       conditions1 = [
           (data['TLX_mean'] == -1),
           (data['TLX_mean'] != -1)]
       values1 = [0, 1]
       data['binar'] = np.select(conditions1, values1)
        
       data_fix = data
       data_fix = data_fix.reset_index(drop=True)
       
       #-------------------------------------------------------
       #RR
       rr_filter1 = signal.savgol_filter(data_fix['rr'], 25, polyorder=3, mode="nearest")
       #HEART RATE
       #hr_filter1 = signal.savgol_filter(data_fix['hr_calculate'], 25, polyorder=3, mode="nearest")
       data_fix['hr_filter_rr'] = 60/rr_filter1
       #GSR
       gsr_filter1 = signal.savgol_filter(data_fix['gsr'], 25, polyorder=3, mode="nearest")
       #temperature
       temp_filter1 = signal.savgol_filter(data_fix['temperature'], 25, polyorder=3, mode="nearest")
       
       #-------------------------------------------------------
       data_fix['rr_filter'] = rr_filter1
       data_fix['gsr_filter'] = gsr_filter1
       data_fix['temperature_filter'] = temp_filter1
       data_fix['hr_filter'] = data_fix['hr_filter_rr']
       print(data_fix.columns.tolist())
       
      
       data_bio = columns = pd.DataFrame( columns = ['rr_filter','gsr_filter','temperature_filter','hr_filter','lable','user'])
       data_bio['rr_filter'] = data_fix['rr_filter']
       data_bio['gsr_filter'] = data_fix['gsr_filter']
       data_bio['temperature_filter'] = data_fix['temperature_filter']
       data_bio['hr_filter'] =  data_fix['hr_filter']
       data_bio['lable'] =  data_fix['binar']
       data_bio['user'] =  data_fix['user_id']
       data_bio['multy_lable'] =  data_fix['multy']
       #-------------------------------------------------------
       
       #normalization
       from sklearn import preprocessing
       import pandas as pd
       data_bio = data_bio.drop('user', axis=1)
       data_bio = data_bio.drop('lable', axis=1)
       data_bio = data_bio.drop('multy_lable', axis=1)
       #data_bio = data_bio.drop('Unnamed: 0', axis=1)
       
       d = preprocessing.normalize(data_bio, norm = 'max', axis=0)
       names = data_bio.columns
       data_bio = pd.DataFrame(d, columns=names)
       print(data_bio.head())
       #end normalization
       #standartization
       from sklearn.preprocessing import StandardScaler
       scale= StandardScaler()
        
       # separate the independent and dependent variables
        
       # standardization of dependent variables
       scaled_data = scale.fit_transform(data_bio) 
       data_bio = pd.DataFrame(scaled_data, columns=names)

       #end standartization
       
       
       rr_feature = np.zeros((len(data_bio['rr_filter']),6))
       i=5
       j=0
       for j in range (len(data_bio['rr_filter'])-11):
          results = plot_poincare(data_bio.loc[j:j+10,'rr_filter'])
          rr_feature[i] = results
          i+=1
       
       hr_feature = np.zeros((len(data_bio['gsr_filter']),4))
       gsr_feature = np.zeros((len(data_bio['temperature_filter']),4))
       temp_feature = np.zeros((len(data_bio['hr_filter']),4))
       i=5
       j=0
       for j in range (len(data_bio['hr_filter'])-11):
          results1 = feature_extraction(data_bio.loc[j:j+10,'gsr_filter'])
          gsr_feature[i] = results1
          
          results2 = feature_extraction(data_bio.loc[j:j+10,'temperature_filter'])
          temp_feature[i] = results2
          
          results3 = feature_extraction(data_bio.loc[j:j+10,'hr_filter'])
          hr_feature [i] = results3
       
          i+=1
  
       #-------------------------------------------------------

       df_rr = pd.DataFrame(rr_feature, columns = ['rr_sd1','rr_sd2','rr_sd','rr_mean','rr_min','rr_max'])
       df_hr = pd.DataFrame(hr_feature, columns = ['hr_sd','hr_mean','hr_min','hr_max'])
       df_gsr = pd.DataFrame(gsr_feature, columns = ['gsr_sd','gsr_mean','gsr_min','gsr_max'])
       df_temp = pd.DataFrame(temp_feature, columns = ['temp_sd','temp_mean','temp_min','temp_max'])
       
       
       result_data = pd.concat([df_rr, df_hr, df_gsr, df_temp], axis=1)
       result_data['lable'] = data_fix['binar']
       result_data['lable_multy'] = data_fix['multy']
       result_data['user'] = data_fix['user_id']
       result_data = result_data[result_data['temp_sd'] != 0]
          

       index = result_data.user[10]
       name = '/content/drive/My Drive/laba/multy_lable_norm/stress_' + str(index) + '.csv'
       #data_bio.to_csv(name)
       result_data.to_csv(name)