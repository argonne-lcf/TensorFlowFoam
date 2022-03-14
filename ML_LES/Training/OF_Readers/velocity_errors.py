import numpy as np

from OF_Readers.mesh_params import num_cells_internal, num_cells_outlet, num_cells_inlet, num_cells_upperWall, num_cells_lowerWall
from OF_Readers.mesh_params import internal_cells_start
from OF_Readers.mesh_params import umag_outlet_start

def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def u_read(filename):
    # Read file
    file = open(filename,'r')
    lines_1 = file.readlines()[internal_cells_start:internal_cells_start+num_cells_internal] # internal field
    file.close()

    for i in range(len(lines_1)):
        lines_1[i] = lines_1[i].strip('\n')
        lines_1[i] = lines_1[i].strip('(')
        lines_1[i] = lines_1[i].strip(')')
        lines_1[i] = lines_1[i].split()

    field = np.asarray(lines_1).astype('double')[:,:2]

    return field

def umag_read(filename):
    # Read file
    file = open(filename,'r')
    lines = file.readlines()
    lines_1 = lines[internal_cells_start:internal_cells_start+num_cells_internal] # internal field
    lines_2 = lines[umag_outlet_start:umag_outlet_start+num_cells_outlet] # Outlet
    file.close()

    for i in range(len(lines_1)):
        lines_1[i] = lines_1[i].strip('\n')

    for i in range(len(lines_2)):
        lines_2[i] = lines_2[i].strip('\n')


    field = np.asarray(lines_1).astype('double').reshape(num_cells_internal,1)
    field = replaceZeroes(field)

    outlet = np.asarray(lines_2).astype('double').reshape(num_cells_outlet,1)
    outlet = replaceZeroes(outlet)

    total = np.concatenate((field,outlet),axis=0)

    return total

def u_grad_read(filename):
	# Read file
    file = open(filename,'r')
    lines = file.readlines()
    lines_1 = lines[internal_cells_start:internal_cells_start+num_cells_internal] # internal field
    lines_2 = lines[gradnutmag_inlet_start:gradnutmag_inlet_start+num_cells_inlet] # Inlet
    lines_3 = lines[gradnutmag_outlet_start:gradnutmag_outlet_start+num_cells_outlet] # Outlet
    lines_4 = lines[gradnutmag_upperWall_start:gradnutmag_upperWall_start+num_cells_upperWall] # upperWall
    lines_5 = lines[gradnutmag_lowerWall_start:gradnutmag_lowerWall_start+num_cells_lowerWall] # lowerWall
    file.close()

    for i in range(len(lines_1)):
        lines_1[i] = lines_1[i].strip('\n')
        lines_1[i] = lines_1[i].strip('(')
        lines_1[i] = lines_1[i].strip(')')
        lines_1[i] = lines_1[i].split()

    for i in range(len(lines_2)):
        lines_2[i] = lines_2[i].strip('\n')
        lines_2[i] = lines_2[i].strip('(')
        lines_2[i] = lines_2[i].strip(')')
        lines_2[i] = lines_2[i].split()

    for i in range(len(lines_3)):
        lines_3[i] = lines_3[i].strip('\n')
        lines_3[i] = lines_3[i].strip('(')
        lines_3[i] = lines_3[i].strip(')')
        lines_3[i] = lines_3[i].split()

    for i in range(len(lines_4)):
        lines_4[i] = lines_4[i].strip('\n')
        lines_4[i] = lines_4[i].strip('(')
        lines_4[i] = lines_4[i].strip(')')
        lines_4[i] = lines_4[i].split()

    for i in range(len(lines_5)):
        lines_5[i] = lines_5[i].strip('\n')
        lines_5[i] = lines_5[i].strip('(')
        lines_5[i] = lines_5[i].strip(')')
        lines_5[i] = lines_5[i].split()


    field = np.asarray(lines_1).astype('double')[:,:2]
    inlet = np.asarray(lines_2).astype('double')[:,:2]
    outlet = np.asarray(lines_3).astype('double')[:,:2]
    upperWall = np.asarray(lines_4).astype('double')[:,:2]
    lowerWall = np.asarray(lines_5).astype('double')[:,:2]

    total = np.concatenate((field,inlet,outlet,upperWall,lowerWall),axis=0)

    return total

def umag_error_write(fname_true,fname_pred,fname_w):
    # Calculating errors and writing out
    umag_true = umag_read(fname_true) # nut error
    umag_pred = umag_read(fname_pred)
    umag_err = np.abs(umag_true-umag_pred)
    # Read from file
    file = open(fname_pred,'r')
    lines = file.readlines()
    file.close()

    # Umag lines correction for inlet BC
    lines[20570] = '        value           uniform 0.0;\n'

    start = 0
    for i in range(num_cells_internal):
        lines[internal_cells_start+i] = str(umag_err[start,-1])+'\n'
        start = start + 1

    for i in range(num_cells_outlet):
        lines[umag_outlet_start+i] = str(umag_err[i,-1])+'\n'
        start = start + 1

    # Write to file
    file = open(fname_w,'w')
    file.writelines(lines)
    file.close()

    # Write true array as well
    file = open(fname_true,'r')
    lines = file.readlines()
    file.close()
    lines[13] ='    object      umag_true'+';\n'
    file = open(fname_pred+'_true','w')
    file.writelines(lines)
    file.close()


def u_error_write(fname_true,fname_pred,fname_w):
	# Calculating errors and writing out
    u_true = u_read(fname_true) # nut error
    u_pred = u_read(fname_pred)
    u_err = np.abs(umag_true-umag_pred)
    # Read from file
    file = open(fname_pred,'r')
    lines = file.readlines()
    file.close()

    start = 0
    for i in range(num_cells_internal):
        lines[internal_cells_start+i] = '('+str(u_err[start,0])+' '+str(u_err[start,1])+' 0)'+'\n'
        start = start + 1

    # Write to file
    file = open(fname_w,'w')
    file.writelines(lines)
    file.close()

    # Write true array as well
    file = open(fname_true,'r')
    lines = file.readlines()
    file.close()
    lines[13] ='    object      U_true'+';\n'
    file = open(fname_pred+'_true','w')
    file.writelines(lines)
    file.close()


def gradux_error_write(fname_true,fname_pred,fname_w):
	# Calculating errors and writing out
    u_grad_true = u_grad_read(fname_true) # nut error
    u_grad_pred = u_grad_read(fname_pred)
    u_grad_err = np.abs(umag_true-umag_pred)
    # Read from file
    file = open(fname_pred,'r')
    lines = file.readlines()
    file.close()

    start = 0
    for i in range(num_cells_internal):
        lines[internal_cells_start+i] = '('+str(u_grad_err[start,0])+' '+str(u_grad_err[start,1])+' 0)'+'\n'
        start = start + 1

    for i in range(num_cells_inlet):
        lines[internal_cells_start+i] = '('+str(u_grad_err[start,0])+' '+str(u_grad_err[start,1])+' 0)'+'\n'
        start = start + 1

    for i in range(num_cells_outlet):
        lines[internal_cells_start+i] = '('+str(u_grad_err[start,0])+' '+str(u_grad_err[start,1])+' 0)'+'\n'
        start = start + 1

    for i in range(num_cells_upperWall):
        lines[internal_cells_start+i] = '('+str(u_grad_err[start,0])+' '+str(u_grad_err[start,1])+' 0)'+'\n'
        start = start + 1

    for i in range(num_cells_lowerWall):
        lines[internal_cells_start+i] = '('+str(u_grad_err[start,0])+' '+str(u_grad_err[start,1])+' 0)'+'\n'
        start = start + 1

    # Write to file
    file = open(fname_w,'w')
    file.writelines(lines)
    file.close()

    # Write true array as well
    file = open(fname_true,'r')
    lines = file.readlines()
    file.close()
    lines[13] ='    object      gradux_true'+';\n'
    file = open(fname_pred+'_true','w')
    file.writelines(lines)
    file.close()


def graduy_error_write(fname_true,fname_pred,fname_w):
	# Calculating errors and writing out
    u_grad_true = u_grad_read(fname_true) # nut error
    u_grad_pred = u_grad_read(fname_pred)
    u_grad_err = np.abs(umag_true-umag_pred)
    # Read from file
    file = open(fname_pred,'r')
    lines = file.readlines()
    file.close()

    start = 0
    for i in range(num_cells_internal):
        lines[internal_cells_start+i] = '('+str(u_grad_err[start,0])+' '+str(u_grad_err[start,1])+' 0)'+'\n'
        start = start + 1

    for i in range(num_cells_inlet):
        lines[internal_cells_start+i] = '('+str(u_grad_err[start,0])+' '+str(u_grad_err[start,1])+' 0)'+'\n'
        start = start + 1

    for i in range(num_cells_outlet):
        lines[internal_cells_start+i] = '('+str(u_grad_err[start,0])+' '+str(u_grad_err[start,1])+' 0)'+'\n'
        start = start + 1

    for i in range(num_cells_upperWall):
        lines[internal_cells_start+i] = '('+str(u_grad_err[start,0])+' '+str(u_grad_err[start,1])+' 0)'+'\n'
        start = start + 1

    for i in range(num_cells_lowerWall):
        lines[internal_cells_start+i] = '('+str(u_grad_err[start,0])+' '+str(u_grad_err[start,1])+' 0)'+'\n'
        start = start + 1

    # Write to file
    file = open(fname_w,'w')
    file.writelines(lines)
    file.close()

    # Write true array as well
    file = open(fname_true,'r')
    lines = file.readlines()
    file.close()
    lines[13] ='    object      graduy_true'+';\n'
    file = open(fname_pred+'_true','w')
    file.writelines(lines)
    file.close()


if __name__ == '__main__':
    print('Velocity error calculations')