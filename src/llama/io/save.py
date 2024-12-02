import h5py
import numpy as np

# Example array
array = np.random.random((100, 100))

# File name
file_name = "example.h5"

# Save the array to an HDF5 file
with h5py.File(file_name, "w") as h5file:
    # Create a dataset named 'dataset_name' in the file
    h5file.create_dataset("dataset_name", data=array)

print(f"Array saved to {file_name}")
