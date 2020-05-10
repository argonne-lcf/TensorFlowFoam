import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.externals import joblib
import matplotlib.cm as cm
from sklearn.metrics import r2_score
import pandas as pd

# Fixing paths
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)

from ML_Model import get_model


def permutation_importances(model,X_p,y_p,variable_names):
    baseline = r2_score(model.predict(X_p),y_p)
    imp = []
    temp_df = pd.DataFrame(data=X_p,columns=variable_names)
    for col in temp_df.columns:
        save = temp_df[col].copy()
        temp_df[col] = np.random.permutation(temp_df[col])
        m = r2_score(model.predict(temp_df), y_p)
        temp_df[col] = save
        imp.append(baseline - m)
    return np.array(imp)


if __name__ == '__main__':
    print('Training assessment file')
    num_inputs = 5
    num_outputs = 1
    model = get_model(num_inputs,num_outputs,6,40)
    model.load_weights('model.h5')

    # Load data
    total_data = np.load(HERE+'/Total_dataset.npy')
    true_op = np.copy(total_data[:,num_inputs:])

    # Scale features
    scaler_filename = "mv_scaler.save"
    scaler = joblib.load(scaler_filename)
    # print(scaler.mean_)
    # print(scaler.var_)

    total_data[:,:] = scaler.transform(total_data[:,:])
    true_transformed = np.copy(total_data[:,num_inputs:])

    # Using *.h5 file
    y_out = model.predict(total_data[:,:num_inputs])

    # Plot feature importances
    variable_names = ['ux','uy','cx','cy','h']
    imp_array = permutation_importances(model,total_data[:1000,:num_inputs],total_data[:1000,num_inputs:],variable_names)
    indices = np.argsort(imp_array)[::-1]
    # Plot individual feature importance
    plt.figure(figsize=(12,10))
    x = np.arange(len(variable_names))
    plt.barh(x,width=imp_array[indices])
    plt.yticks(x, [variable_names[indices[f]] for f in range(len(variable_names))])

    plt.ylabel('Feature',fontsize=24)
    plt.xlabel('Relative decisiveness',fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    plt.close()


    # Scatter plot 
    # Rescale y_out to be in physical domain
    total_data[:,num_inputs:] = y_out[:,:]
    total_data[:,:] = scaler.inverse_transform(total_data[:,:])

    # Scatter plot 
    plt.figure()
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


