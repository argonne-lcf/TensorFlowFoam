import numpy as np

from OF_Readers.mesh_params import num_cells_internal, num_cells_outlet, num_cells_inlet, num_cells_upperWall, num_cells_lowerWall
from OF_Readers.mesh_params import internal_cells_start
from OF_Readers.mesh_params import rotation_inlet_start, rotation_outlet_start, rotation_upperWall_start, rotation_lowerWall_start

def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def rotation_read(filename):
    # Read file
    file = open(filename,'r')
    lines = file.readlines()
    lines_1 = lines[internal_cells_start:internal_cells_start+num_cells_internal] # 23 is the size of the OpenFOAM header and other stuff
    lines_2 = lines[rotation_inlet_start:rotation_inlet_start+num_cells_inlet] # Inlet
    lines_3 = lines[rotation_outlet_start:rotation_outlet_start+num_cells_outlet] # Outlet
    lines_4 = lines[rotation_upperWall_start:rotation_upperWall_start+num_cells_upperWall] # upperWall
    lines_5 = lines[rotation_lowerWall_start:rotation_lowerWall_start+num_cells_lowerWall] # lowerWall
    file.close()

    for i in range(len(lines_1)):
        lines_1[i] = lines_1[i].strip('\n')

    for i in range(len(lines_2)):
        lines_2[i] = lines_2[i].strip('\n')

    for i in range(len(lines_3)):
        lines_3[i] = lines_3[i].strip('\n')

    for i in range(len(lines_4)):
        lines_4[i] = lines_4[i].strip('\n')

    for i in range(len(lines_5)):
        lines_5[i] = lines_5[i].strip('\n')


    field = np.asarray(lines_1).astype('double').reshape(num_cells_internal,1)
    field = replaceZeroes(field)

    inlet = np.asarray(lines_2).astype('double').reshape(num_cells_inlet,1)
    inlet = replaceZeroes(inlet)

    outlet = np.asarray(lines_3).astype('double').reshape(num_cells_outlet,1)
    outlet = replaceZeroes(outlet)

    upperWall = np.asarray(lines_4).astype('double').reshape(num_cells_upperWall,1)
    upperWall = replaceZeroes(upperWall)

    lowerWall = np.asarray(lines_5).astype('double').reshape(num_cells_lowerWall,1)
    lowerWall = replaceZeroes(lowerWall)

    total = np.concatenate((field,inlet,outlet,upperWall,lowerWall),axis=0)

    return total

def rotation_error_write(fname_true,fname_pred,fname_w):
    # Calculating errors and writing out
    rotation_true = rotation_read(fname_true) # nut error
    rotation_pred = rotation_read(fname_pred)
    rotation_err = np.abs(rotation_true-rotation_pred)
    # Read from file
    file = open(fname_pred,'r')
    lines = file.readlines()
    file.close()

    start = 0
    for i in range(num_cells_internal):
        lines[internal_cells_start+i] = str(rotation_err[start,-1])+'\n'
        start = start + 1

    for i in range(num_cells_inlet):
        lines[rotation_inlet_start+i] = str(rotation_err[i,-1])+'\n'
        start = start + 1

    for i in range(num_cells_outlet):
        lines[rotation_outlet_start+i] = str(rotation_err[i,-1])+'\n'
        start = start + 1

    for i in range(num_cells_upperWall):
        lines[rotation_upperWall_start+i] = str(rotation_err[i,-1])+'\n'
        start = start + 1

    for i in range(num_cells_lowerWall):
        lines[rotation_lowerWall_start+i] = str(rotation_err[i,-1])+'\n'
        start = start + 1

    # Write to file
    file = open(fname_w,'w')
    file.writelines(lines)
    file.close()

    # Write true array as well
    file = open(fname_true,'r')
    lines = file.readlines()
    file.close()
    lines[13] ='    object      rot_true'+';\n'
    file = open(fname_pred+'_true','w')
    file.writelines(lines)
    file.close()

if __name__ == '__main__':
    print('Rotation error calculation')