import numpy as np
import os
import pdb

from tensorflow.python.keras.backend import dtype

displacement_dir = '/home/data/litho/Raghu/Interpolation/AssymetricSimulations/NotUsed/Undercut_03/displacement-field/'
strain_dir = '/home/data/litho/Raghu/Interpolation/AssymetricSimulations/NotUsed/Undercut_03/elastic-strain-field/'

def import_files_list():
    """List of files of displacement fields and strain fields"""
    displacements = os.listdir(displacement_dir)
    displacements = [i + j for i, j in zip([displacement_dir] * len(displacements), displacements)]
    displacements = sorted(displacements)
    strains = os.listdir(strain_dir)
    strains = [k + l for k, l in zip([strain_dir] * len(strains), strains)]
    strains = sorted(strains)
    return np.column_stack((displacements, strains))


def import_data(fileNames):
    """Displacement Field and Elastic Strain Field are extracted with X, Y, Ux, Uy """
    displacement_field = np.genfromtxt(fileNames[0], delimiter=',')
    elastic_strain_field = np.genfromtxt(fileNames[1], delimiter=',')
    # displacement_field[:, 1] = displacement_field[:, 1] - displacement_field[:, 4]
    # displacement_field[:, 2] = displacement_field[:, 2] - displacement_field[:, 5]

    return displacement_field[:, [1, 2]] * 1000, np.column_stack((displacement_field[:, 1] - displacement_field[:, 4], displacement_field[:, 2] - displacement_field[:, 5])) * 1000
