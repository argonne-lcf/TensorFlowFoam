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

    # Read Case 1
    U = read_vector('U_1')
    nut = read_scalar('nut_1')
    y = read_scalar('yWall_1')
    cx = read_scalar('cx_1')
    cy = read_scalar('cy_1')
    # create array of Re information
    h = np.ones(shape=(np.shape(U)[0],1),dtype='double')*2.0
    total_dataset = np.concatenate((U,cx,cy,h,nut),axis=-1)

    # Read Case 2
    U = read_vector('U_2')
    nut = read_scalar('nut_2')
    y = read_scalar('yWall_2')
    cx = read_scalar('cx_2')
    cy = read_scalar('cy_2')
    # create array of Re information
    h = np.ones(shape=(np.shape(U)[0],1),dtype='double')*1.5
    temp_dataset = np.concatenate((U,cx,cy,h,nut),axis=-1)
    total_dataset = np.concatenate((total_dataset,temp_dataset),axis=0)

    # Read Case 3
    U = read_vector('U_3')
    nut = read_scalar('nut_3')
    y = read_scalar('yWall_3')
    cx = read_scalar('cx_3')
    cy = read_scalar('cy_3')
    # create array of Re information
    h = np.ones(shape=(np.shape(U)[0],1),dtype='double')*0.5
    temp_dataset = np.concatenate((U,cx,cy,h,nut),axis=-1)
    total_dataset = np.concatenate((total_dataset,temp_dataset),axis=0)

    # Read Case 4
    U = read_vector('U_4')
    nut = read_scalar('nut_4')
    y = read_scalar('yWall_4')
    cx = read_scalar('cx_4')
    cy = read_scalar('cy_4')
    # create array of Re information
    h = np.ones(shape=(np.shape(U)[0],1),dtype='double')*0.75
    temp_dataset = np.concatenate((U,cx,cy,h,nut),axis=-1)
    total_dataset = np.concatenate((total_dataset,temp_dataset),axis=0)

    # Read Case 5
    U = read_vector('U_5')
    nut = read_scalar('nut_5')
    y = read_scalar('yWall_5')
    cx = read_scalar('cx_5')
    cy = read_scalar('cy_5')
    # create array of Re information
    h = np.ones(shape=(np.shape(U)[0],1),dtype='double')*1.75
    temp_dataset = np.concatenate((U,cx,cy,h,nut),axis=-1)
    total_dataset = np.concatenate((total_dataset,temp_dataset),axis=0)

    # Read Case 6
    U = read_vector('U_6')
    nut = read_scalar('nut_6')
    y = read_scalar('yWall_6')
    cx = read_scalar('cx_6')
    cy = read_scalar('cy_6')
    # create array of Re information
    h = np.ones(shape=(np.shape(U)[0],1),dtype='double')*1.25
    temp_dataset = np.concatenate((U,cx,cy,h,nut),axis=-1)
    total_dataset = np.concatenate((total_dataset,temp_dataset),axis=0)

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
