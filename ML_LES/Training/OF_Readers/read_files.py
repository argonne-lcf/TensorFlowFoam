import numpy as np

def read_coeff_filw_y(filename):
    with open(filename) as f:
        lines = f.readlines()
    f.close()

    num_cells = int(lines[20])
    data_array = np.zeros(shape=(num_cells,1))

    sample = 0 
    for line in range(22,22+num_cells):
        data_array[sample,0] = float(lines[line])
        sample = sample + 1

    return data_array

def read_strain_rate(filename):
    with open(filename) as f:
        lines = f.readlines()
    f.close()

    num_cells = int(lines[20])
    data_array = np.zeros(shape=(num_cells,6))

    sample = 0   
    for line in range(22,22+num_cells):
        linestr = np.float_(lines[line][1:-2].split())
        data_array[sample,:] = linestr[:]
        sample = sample + 1

    return data_array

def read_velocity(filename):
    with open(filename) as f:
        lines = f.readlines()
    f.close()

    num_cells = int(lines[20])
    data_array = np.zeros(shape=(num_cells,3))

    sample = 0   
    for line in range(22,22+num_cells):
        linestr = np.float_(lines[line][1:-2].split())
        data_array[sample,:] = linestr[:]
        sample = sample + 1

    return data_array

if __name__ == '__main__':
    print('File reader for OF')