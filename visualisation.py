from google.colab import drive
drive.mount('/content/drive')
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
import os
import glob

import os

os.chdir('/content/drive/My Drive/laba/cleaned_data/filter/')
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))
idd=0
data_viz = pd.DataFrame()
for f in csv_files:
    data_viz = pd.read_csv(f)
    time = np.array(data_viz['Unnamed: 0'])
    conditions1 = [
    (data_viz['TLX_mean'] == -1),
    (data_viz['TLX_mean'] != -1)]
    values1 = [0, 1]
    data_viz['binar'] = np.select(conditions1, values1)

    conditions2 = [
    (data_viz['task'] == 'quest'),
    (data_viz['task'] != 'quest')]
    values2 = [1, 0]
    data_viz['type'] = np.select(conditions2, values2)
    

    ig, ax1 = plt.subplots(1, figsize=(20, 7))
    ax1.plot(time, data_viz['hr_filter'], color="green", alpha=0.6)
    
    ax1.fill_between(time, (data_viz['binar']*60+60), 60, label="stress", color="cyan", alpha=0.2)
    ax1.fill_between(time, (data_viz['type']*60+30), 60, label="quest", color="red", alpha=0.2)
    
    title = 'hr_filter' ,data_viz.user_id[1] 
    
    plt.title(title )
    plt.legend()
    plt.show()
    #print(data_viz.lable.value_counts())

os.chdir('/content/drive/My Drive/laba/cleaned_data/filter/')
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))
idd=0
data_viz = pd.DataFrame()
for f in csv_files:
    data_viz = pd.read_csv(f)
    time = np.array(data_viz['Unnamed: 0'])
    conditions1 = [
    (data_viz['TLX_mean'] == -1),
    (data_viz['TLX_mean'] != -1)]
    values1 = [0, 1]
    data_viz['binar'] = np.select(conditions1, values1)

    conditions2 = [
    (data_viz['task'] == 'quest'),
    (data_viz['task'] != 'quest')]
    values2 = [1, 0]
    data_viz['type'] = np.select(conditions2, values2)
    

    ig, ax1 = plt.subplots(1, figsize=(20, 7))
    ax1.plot(time, data_viz['rr_filter'], color="green", alpha=0.6)
    
    ax1.fill_between(time, (data_viz['binar']*0.5+0.5), 0.5, label="stress", color="cyan", alpha=0.2)
    ax1.fill_between(time, (data_viz['type']*0.5+0.2), 0.5, label="quest", color="red", alpha=0.2)
    
    title = 'rr_filter' ,data_viz.user_id[1] 
    
    plt.title(title )
    plt.legend()
    plt.show()
    #print(data_viz.lable.value_counts())

os.chdir('/content/drive/My Drive/laba/cleaned_data/filter/')
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))
idd=0
data_viz = pd.DataFrame()
for f in csv_files:
    data_viz = pd.read_csv(f)
    time = np.array(data_viz['Unnamed: 0'])

    conditions1 = [
    (data_viz['TLX_mean'] == -1),
    (data_viz['TLX_mean'] != -1)]
    values1 = [0, 1]
    data_viz['binar'] = np.select(conditions1, values1)

    conditions2 = [
    (data_viz['task'] == 'quest'),
    (data_viz['task'] != 'quest')]
    values2 = [1, 0]
    data_viz['type'] = np.select(conditions2, values2)
    

    ig, ax1 = plt.subplots(1, figsize=(20, 7))
    ax1.plot(time, data_viz['gsr_filter'], color="green", alpha=0.6)
    
    ax1.fill_between(time, (data_viz['binar']*5), 0, label="stress", color="cyan", alpha=0.2)
    ax1.fill_between(time, (data_viz['type']*3), 0, label="quest", color="red", alpha=0.2)
    
    
    title = 'gsr_filter' ,data_viz.user_id[1] 
    
    plt.title(title )
    plt.legend()
    plt.show()

os.chdir('/content/drive/My Drive/laba/cleaned_data/filter/')
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))
idd=0
data_viz = pd.DataFrame()
for f in csv_files:
    data_viz = pd.read_csv(f)
    time = np.array(data_viz['Unnamed: 0'])
    conditions1 = [
    (data_viz['TLX_mean'] == -1),
    (data_viz['TLX_mean'] != -1)]
    values1 = [0, 1]
    data_viz['binar'] = np.select(conditions1, values1)

    conditions2 = [
    (data_viz['task'] == 'quest'),
    (data_viz['task'] != 'quest')]
    values2 = [1, 0]
    data_viz['type'] = np.select(conditions2, values2)

    ig, ax1 = plt.subplots(1, figsize=(20, 7))
    ax1.plot(time, data_viz['temperature_filter'], color="green", alpha=0.6)
    
    ax1.fill_between(time, (data_viz['binar']*5+30), 30, label="stress", color="cyan", alpha=0.2)
    ax1.fill_between(time, (data_viz['type']*3+30), 30, label="quest", color="red", alpha=0.2)
    
    title = 'temperature_filter' ,data_viz.user_id[1] 
    
    plt.title(title )
    plt.legend()
    plt.show()
    #print(data_viz.lable.value_counts())