import numpy as np


def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def read_scalar(filename):
    # Read file
    file = open(filename,'r')
    lines_1 = file.readlines()
    file.close()

    num_cells_internal = int(lines_1[20].strip('\n'))
    lines_1 = lines_1[22:22+num_cells_internal]

    for i in range(len(lines_1)):
        lines_1[i] = lines_1[i].strip('\n')

    field = np.asarray(lines_1).astype('double').reshape(num_cells_internal,1)
    field = replaceZeroes(field)

    return field


def read_vector(filename): # Only x,y components
    file = open(filename,'r')
    lines_1 = file.readlines()
    file.close()

    num_cells_internal = int(lines_1[20].strip('\n'))
    lines_1 = lines_1[22:22+num_cells_internal]

    for i in range(len(lines_1)):
        lines_1[i] = lines_1[i].strip('\n')
        lines_1[i] = lines_1[i].strip('(')
        lines_1[i] = lines_1[i].strip(')')
        lines_1[i] = lines_1[i].split()

    field = np.asarray(lines_1).astype('double')[:,:2]

    return field


if __name__ == '__main__':
    print('Velocity reader file')

    heights = [2.0, 1.5, 0.5, 0.75, 1.75, 1.25]
    total_dataset = []

    # Read Cases
    for i, h in enumerate(heights, start=1):
        U = read_vector(f'U_{i}')
        nut = read_scalar(f'nut_{i}')
        cx = read_scalar(f'cx_{i}')
        cy = read_scalar(f'cy_{i}')
        h = np.ones(shape=(np.shape(U)[0],1),dtype='double') * h
        temp_dataset = np.concatenate((U,cx,cy,h,nut),axis=-1)
        total_dataset.append(temp_dataset)
    total_dataset = np.reshape(total_dataset, (-1,6))

    print(total_dataset.shape)

    # Save data   
    np.save('Total_dataset.npy',total_dataset)

    # Save the statistics of the data
    means = np.mean(total_dataset,axis=0).reshape(1,np.shape(total_dataset)[1])
    stds = np.std(total_dataset,axis=0).reshape(1,np.shape(total_dataset)[1])

    # Concatenate
    op_data = np.concatenate((means,stds),axis=0)
    np.savetxt('means',op_data, delimiter=' ')

    # Need to write out in OpenFOAM rectangular matrix format

    print('Means:')
    print(means)
    print('Stds:')
    print(stds)
