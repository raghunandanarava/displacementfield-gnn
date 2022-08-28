import numpy as np
import os

file_directory = "/home/users/arava/NewInterpolation/Thesis/Parameteric_Analysis/AnsysData/"

"""Import the files list"""
def import_files_list():
    files_list = os.listdir(file_directory)
    return files_list

"""Import the file data"""
def import_data(filename):
    data = np.genfromtxt(os.path.join(file_directory, filename), delimiter=",")
    #return data
    initial_pos = data[:, 0:2] - data[:, 2:4]
    return np.column_stack((data[:, [2, 3, 5, 6, 7, 8]], initial_pos)), initial_pos#np.column_stack((initial_pos, data[:, 4]))
