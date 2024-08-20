import nrrd
import numpy as np
import os

from sklearn.decomposition import PCA

def calculate_orientation(array_3d, label_value):
    indices = np.where(array_3d == label_value)
    coordinates = np.array(indices).T
    pca = PCA(n_components=3)
    pca.fit(coordinates)
    
    # The third principal component is perpendicular to the structure
    perpendicular_direction = pca.components_[2]
    
    # Ensure the vector points towards the positive octant
    if np.sum(perpendicular_direction) < 0:
        perpendicular_direction = -perpendicular_direction
    
    return perpendicular_direction

def rotate_to_z_axis(array_3d, direction_vector):
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    axis_vectors = np.eye(3)
    dot_products = np.abs(np.dot(axis_vectors, direction_vector))
    closest_axis = np.argmax(dot_products)
    
    if closest_axis == 2:  # Already closest to z-axis
        return array_3d, (None, 0)
    elif closest_axis == 1:  # y-axis
        plane = (1, 2)  # yz-plane
        k = 1  # 90 degree rotation
    else:  # x-axis
        plane = (0, 2)  # xz-plane
        k = 1  # 90 degree rotation
    
    rotated_array = np.rot90(array_3d, k=k, axes=plane)
    
    return rotated_array, (plane, k)

def unapply_rotation(array_3d, rotation_info):
    plane, k = rotation_info
    if plane is None:
        return array_3d
    
    # Reverse the rotation by rotating in the opposite direction
    return np.rot90(array_3d, k=-k, axes=plane)

# Load the data
current_directory = os.getcwd()
input_nrrd_path = f'{current_directory}/data/label/09936_03280_04560_zyx_256_chunk_s1_vol_label.nrrd'  # Path to your NRRD file
data, header = nrrd.read(input_nrrd_path)
data = np.rot90(data, k=1, axes=(0, 1))
nrrd.write(f'{current_directory}/output/09936_03280_04560_zyx_256_chunk_s1_vol_label_test_rotated.nrrd', data, header)

#rotate the data
primary_direction = calculate_orientation(data, 1)
rot_data, rotation_info = rotate_to_z_axis(data, primary_direction)
nrrd.write(f'{current_directory}/output/09936_03280_04560_zyx_256_chunk_s1_vol_label_rotated.nrrd', rot_data, header)
unrot_data = unapply_rotation(rot_data, rotation_info)
nrrd.write(f'{current_directory}/output/09936_03280_04560_zyx_256_chunk_s1_vol_label_unrotated.nrrd', unrot_data, header)