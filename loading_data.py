import h5py
from pathlib import Path

path = Path("C:/Users/adifranc/OneDrive - UAB - The University of Alabama at Birmingham/Finger, Adam's files - DBS_ML_FinalProject")
patient = Path("ug3008.mat")

data = h5py.File(path / patient)

with h5py.File(path / patient, 'r') as f:
    f.close()


def file_opener(path, patient):

    path = Path(path) if type(path) == str else path
    patient = Path(patient) if type(patient) == str else patient
    
    patient_data = h5py.File(path / patient)

    return patient_data