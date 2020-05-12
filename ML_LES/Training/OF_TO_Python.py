import numpy as np
import matplotlib.pyplot as plt
import string

# Reproducibility
np.random.seed(10)

# Paths
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(os.getcwd())
BASE_DIR = os.path.dirname(PARENT)
DATA_DIR = PARENT+r'/LES_Data'
sys.path.insert(0,PARENT)

# Figure out sampling
num_samples = 1
t_start = 40
t_end = 520
sampling_interval = 40
num_total_samples = 50000

# Import functions for reading OF output
from OF_Readers.read_files import read_coeff_filw_y, read_strain_rate, read_velocity


def concat_sampling(DIR):

    strain_rate = read_strain_rate(DIR+'/S_ij')
    velocity = read_velocity(DIR+'/U')
    # delta = read_coeff_filw_y(DIR+'/del')
    # yw = read_coeff_filw_y(DIR+'/yw')
    Cs = read_coeff_filw_y(DIR+'/Cs')
    # nut = read_coeff_filw_y(DIR+'/nut')

    dataset = np.concatenate((strain_rate,velocity,Cs),axis=1)

    return dataset       

if __name__ == '__main__':

    mode = 'assess'

    if mode == 'generate':
        print('Generating new data:')  
        # Dataset
        t = t_start
        sample = 0
        while t <= t_end:
            if sample == 0:
                dataset = concat_sampling(DATA_DIR+r'/'+str(t))
            else:
                temp = concat_sampling(DATA_DIR+r'/'+str(t))
                dataset = np.concatenate((dataset,temp),axis=0)
            sample = sample + 1

            print('Time:',t)
            t = t + sampling_interval

        np.random.shuffle(dataset)
        dataset = dataset[:num_total_samples,:]
        print('Final dataset shape',np.shape(dataset))
        np.save('Total_Data.npy',dataset)

    else:
        dataset = np.load('Total_Data.npy')
        print('Loading pre-generated data')
        print('Final dataset shape',np.shape(dataset))


    means = np.mean(dataset,axis=0).reshape(1,np.shape(dataset)[1])
    stds = np.std(dataset,axis=0).reshape(1,np.shape(dataset)[1])

    np.savetxt('Means.txt',means,delimiter=',')
    np.savetxt('Stds.txt',stds,delimiter=',')
    
    print('Means:')
    print(means)
    print('Stds:')
    print(stds)

    target = dataset[:,-1].flatten()
    plt.figure()
    plt.hist(target,bins=100,label='Cs')
    plt.show()
