import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.externals import joblib
import matplotlib.cm as cm

# Fixing paths
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == '__main__':
    print('Training assessment file')
    num_inputs = 5

    # Load data
    total_data = np.load(HERE+'/Total_dataset.npy')
    true_op = np.copy(total_data[:,num_inputs:])
    # We use our "load_graph" function
    graph = load_graph('ML_SA_CG.pb')
    # We access the input and output nodes 
    x = graph.get_tensor_by_name('prefix/input_placeholder:0')
    y = graph.get_tensor_by_name('prefix/output_value/BiasAdd:0')

    # Scale features
    scaler_filename = "mv_scaler.save"
    scaler = joblib.load(scaler_filename)
    print(scaler.mean_)
    print(scaler.var_)

    total_data[:,:] = scaler.transform(total_data[:,:])
    true_transformed = np.copy(total_data[:,num_inputs:])

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 
        y_out = sess.run(y, feed_dict={
            x: total_data[:,:num_inputs]
        })

    # # Scatter plot 
    # Rescale y_out to be in physical domain
    total_data[:,num_inputs:] = y_out[:,:]
    total_data[:,:] = scaler.inverse_transform(total_data[:,:])

    # Scatter plot 
    plt.figure()
    # plt.scatter(true_op[:,0].flatten(),total_data[:,-1].flatten(),label='ML',c=wall_distance,cmap=cm.YlOrBr,alpha=0.5)
    plt.plot(true_op[:,0].flatten(),total_data[:,-1].flatten(),'ro',label='ML',markersize=1,alpha=0.5)
    plt.plot(true_op[:,0].flatten(),true_op[:,0].flatten(),'k-',linestyle='--') # identity line
    plt.title('Scatter')
    plt.xlabel('Truth')
    plt.ylabel('Predicted')
    plt.xlim((true_op[:,0].min(),true_op[:,0].max()))
    plt.ylim((true_op[:,0].min(),true_op[:,0].max()))
    plt.show()

    # Plot of the histogram if truth and predictions
    r_min = total_data[:,-1].flatten().min()
    r_max = total_data[:,-1].flatten().max()

    plt.figure()
    plt.hist(total_data[:,-1].flatten(),range=(r_min,r_max),bins=30,label='Predicted',alpha=0.5)
    plt.hist(true_op[:,0].flatten(),range=(r_min,r_max),bins=30,label='True',alpha=0.5)
    plt.legend()
    plt.show()

    # Plot the convergence of learning
    loss_log = np.loadtxt('training.log',skiprows=1,delimiter=',')
    
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(7,6))
    ax[0].plot(loss_log[:,0],loss_log[:,1],label='Training loss')
    ax[0].plot(loss_log[:,0],loss_log[:,3],label='Validation loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Mean-squared error')

    ax[1].plot(loss_log[:,0],loss_log[:,2],label=r'Training $R^2$')
    ax[1].plot(loss_log[:,0],loss_log[:,4],label=r'Validation $R^2$')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel(r'$R^2$')

    plt.tight_layout()
    plt.show()

