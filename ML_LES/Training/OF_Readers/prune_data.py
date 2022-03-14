import numpy as np

def prune_data(dataset,npoints):
    '''
    for a given dataset, selects a subset of npoints samples through feature distance mapping
    '''
    l_param = 1.5
    comparison_distance = 10.0

    # Select first point randomly
    sample_point = np.random.randint(low=0,high=np.shape(dataset)[0])
    sample_dataset = dataset[sample_point,:].reshape(1,np.shape(dataset)[1])

    # Do sample proposal till npoints samples are reached
    k = 0
    stride = 0
    while stride < np.shape(dataset)[0]:
        # Choose a random point
        sample_point = np.random.randint(low=0,high=np.shape(dataset)[0])
        sample = dataset[sample_point,:].reshape(1,np.shape(dataset)[1])
        # Find relative distance from each existing member of dataset
        sample_distance = np.divide(np.subtract(sample_dataset[:,:-1],sample[:,:-1]),np.maximum(sample_dataset[:,:-1],sample[:,:-1])) # Distance of input features
        sample_distance = np.sum(np.abs(sample_distance),axis=1)

        if sample_distance[0]>comparison_distance:
            sample_dataset = np.concatenate((sample_dataset,sample),axis=0)
            k = k+1

        if k == npoints-1:
            break

        stride = stride + 1

        if stride == np.shape(dataset)[0] - 1:
        	stride = 0
        	comparison_distance = comparison_distance / l_param
    
    return sample_dataset





if __name__ == '__main__':
    print('This file has different data pruning techniques')