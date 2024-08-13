import os
import numpy as np
import nrrd
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize
# import skimage
from scipy.spatial import cKDTree
from scipy import ndimage
import time 
import concurrent.futures
from midline_helper import *
import graph_tool.all as gt

def calculate_seam_iter(directed_graph, src, tgt, weights, test_size, x_pos, y_pos, z_pos):
    # Compute the residual capacity of the edges
    res = gt.boykov_kolmogorov_max_flow(directed_graph, src, tgt, weights)
    # Use the residual graph to get the max flow
    flow = sum(weights[e] - res[e] for e in tgt.in_edges())
    # Determine the minimum cut partition
    part = gt.min_st_cut(directed_graph, src, weights, res)
    # Find the boundary vertices
    boundary_vertices = find_boundary_vertices(np.array(directed_graph.get_edges()), part)
    print(f"Number of boundary vertices: {len(boundary_vertices)}, Number of vertices: {len(directed_graph.get_vertices())}")
    shape = (test_size, test_size, test_size)
    # Convert the boundary vertices to a 3D array
    boundary_array = boundary_vertices_to_array_masked(boundary_vertices, shape, 'x', x_pos, y_pos, z_pos)
    return boundary_array, flow

def multi_res_seam_iter(res_index, mask_array_data, b_arr_up):
    masked_array = mask_array_data[res_index].copy().astype(np.int16)
    masked_array[b_arr_up == 0] = -1
    stime = time.time()
    directed_graph, src, tgt, weights, x_pos, y_pos, z_pos = create_masked_directed_energy_graph_from_mask(masked_array)
    boundary_array, flow = calculate_seam_iter(directed_graph, src, tgt, weights, masked_array.shape[0], x_pos, y_pos, z_pos)
    return boundary_array

def multi_res_seam_calculation(mask_array_data, res_index=3, upscale_factor=2, dilation_amount=1):
    directed_graph, src, tgt, weights, x_pos, y_pos, z_pos = create_masked_directed_energy_graph_from_mask(mask_array_data[res_index])
    boundary_array, flow = calculate_seam_iter(directed_graph, src, tgt, weights, mask_array_data[res_index].shape[0], x_pos, y_pos, z_pos)
    b_arr_up = upscale_and_dilate_3d(boundary_array, upscale_factor=2, dilation_amount=dilation_amount)
    for i in range(res_index-1, -1, -1):
        boundary_array = multi_res_seam_iter(i, mask_array_data, b_arr_up)
        if i != 0:
            b_arr_up = upscale_and_dilate_3d(boundary_array, upscale_factor=upscale_factor, dilation_amount=dilation_amount)
    return boundary_array

def process_label(labeled_array, label):
    # Create a binary mask for the current label
    mask = (labeled_array == label)
    
    # Compute the distance transform within the label
    distance_map = ndimage.distance_transform_edt(mask)
    
    # Zero out areas outside the label
    distance_map[~mask] = 0
    
    return distance_map

def create_label_distance_map(labeled_array, max_workers=None):
    # Get unique labels, excluding background (assumed to be 0)
    labels = np.unique(labeled_array)
    labels = labels[labels != 0]
    
    # Create an output array with the same shape as the input
    output = np.zeros_like(labeled_array, dtype=float)
    
    # Use ThreadPoolExecutor for parallelization
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for each label
        future_to_label = {executor.submit(process_label, labeled_array, label): label for label in labels}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_label):
            label = future_to_label[future]
            try:
                distance_map = future.result()
                # Add the distance map for this label to the output
                output += distance_map
            except Exception as exc:
                print(f'Label {label} generated an exception: {exc}')
    
    return output.astype(int)

def filter_and_reassign_labels(label_data, cc_min_size):
    """
    Filter out small disconnected components within each label and reassign
    remaining labels starting from 1 and incrementing by 1.

    Parameters:
    label_data (numpy.ndarray): Input label data
    cc_min_size (int): Minimum size for a connected component to be kept

    Returns:
    numpy.ndarray: Filtered and reassigned label data
    """
    unique_labels = np.unique(label_data)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    new_label_data = np.zeros_like(label_data)
    new_label = 1

    for label in unique_labels:
        label_mask = label_data == label
        labeled_components, _ = ndimage.label(label_mask)
        
        valid_component_mask = np.zeros_like(label_mask, dtype=bool)
        
        for component in range(1, labeled_components.max() + 1):
            component_mask = labeled_components == component
            if np.sum(component_mask) >= cc_min_size:
                valid_component_mask |= component_mask

        if np.any(valid_component_mask):
            new_label_data[valid_component_mask] = new_label
            new_label += 1

    return new_label_data

def apply_pca_thinning(data, label_value):
    # Extract coordinates of the current structure
    coords = np.column_stack(np.where(data == label_value))
    
    # Apply PCA
    pca = PCA(n_components=3)
    pca.fit(coords)
    normal_vector = pca.components_[-1]  # The component with the least variance
    
    # Project points onto the normal vector
    mean_center = pca.mean_
    projections = np.dot((coords - mean_center), normal_vector)
    
    # Sort coordinates based on their projections
    sorted_indices = np.argsort(projections)
    sorted_coords = coords[sorted_indices]
    
    # Calculate the number of points to consider for front and back (e.g., 10% of total points)
    n_points = len(coords)
    n_edge_points = max(int(0.1 * n_points), 1)  # At least 1 point
    
    # Calculate average positions for front and back portions
    front_avg = np.mean(sorted_coords[-n_edge_points:], axis=0)
    back_avg = np.mean(sorted_coords[:n_edge_points], axis=0)
    
    # Calculate the midpoint between front and back averages
    midpoint = (front_avg + back_avg) / 2
    
    # Project all points onto the plane passing through the midpoint
    plane_projections = coords - np.dot((coords - midpoint), normal_vector)[:, np.newaxis] * normal_vector
    
    # Create a KD-tree for efficient nearest neighbor search
    tree = cKDTree(plane_projections)
    
    # For each unique projected point, find the closest front and back points
    unique_projections, unique_indices = np.unique(plane_projections, axis=0, return_index=True)
    
    midline_points = []
    for proj in unique_projections:
        # Find points in the original coords that project to this point (or very close to it)
        _, idx = tree.query(proj, k=10)  # Get 10 nearest neighbors
        nearby_original = coords[idx]
        
        # Calculate the front and back points for this projection
        front_point = nearby_original[np.argmax(np.dot(nearby_original - proj, normal_vector))]
        back_point = nearby_original[np.argmin(np.dot(nearby_original - proj, normal_vector))]
        
        # Calculate the midpoint
        midpoint = (front_point + back_point) / 2
        midline_points.append(midpoint)
    
    # Convert midline points to integer coordinates
    midline_coords = np.round(midline_points).astype(int)
    
    # Ensure coordinates are within the bounds of the original data dimensions
    midline_coords = np.clip(midline_coords, 0, np.array(data.shape) - 1)
    
    # Create a new array to store the thinned structure
    thinned_structure = np.zeros_like(data, dtype=np.uint8)
    thinned_structure[tuple(midline_coords.T)] = label_value
    
    return thinned_structure

def process_single_label(label_data, label_value, output_path):
    # Create a binary mask for the specified label
    mask = (label_data == label_value)

    # Compute the distance map within the label
    distance_map = create_label_distance_map(mask)

    # Mask out areas outside the label
    distance_map[~mask] = -1

    # Create the energy graph
    directed_graph, src, tgt, weights, x_pos, y_pos, z_pos = create_masked_directed_energy_graph_from_mask(distance_map)

    # Calculate the seam
    seam_array, flow = calculate_seam_iter(directed_graph, src, tgt, weights, distance_map.shape[0], x_pos, y_pos, z_pos)

    # Convert the boundary vertices to a 3D array
    # seam_array = boundary_vertices_to_array_masked(boundary_array, distance_map.shape, 'x', x_pos, y_pos, z_pos)

    # Save the seam array as an NRRD file
    nrrd.write(output_path, seam_array.astype(np.uint8))

    return seam_array

def process_structures(nrrd_path, output_path, sk=False):
    # print(skimage.__version__)
    # Load the data
    data, header = nrrd.read(nrrd_path)
    # data = filter_and_reassign_labels(data, 300)  # Filter out small disconnected components

    # Prepare a new array to store all thinned structures
    thinned_data = np.zeros_like(data, dtype=np.uint8)  # Ensure thinned_data is of type uint8

    stime = time.time()
    # data = data == 1
    # thinned_data = create_label_distance_map(data)
    thinned_data = process_single_label(data, 1, output_path)
    print(f"Time taken: {time.time() - stime:.2f} seconds")
    # if sk:
    #     # thinned_data = skeletonize_3d_multi_label_slice(data)
    #     data = data == 1
    #     thinned_data = skeletonize(data, method='lee', surface=True)
    # else:
    
    #     # Process each structure
    #     for i in range(1, data.max() + 1):
    #         print(f"Processing structure {i}")
    #         thinned_data += apply_pca_thinning(data, i).astype(np.uint8)  # Explicit conversion to uint8
    
    # Save the thinned structures as a new NRRD file
    nrrd.write(output_path, thinned_data.astype(np.uint8), header)

# Example usage:
current_directory = os.getcwd()
input_nrrd_path = f'{current_directory}/data/label/09936_03280_04560_zyx_256_chunk_s1_vol_label.nrrd'  # Path to your NRRD file
output_nrrd_path = f'{current_directory}/output/09936_03280_04560_zyx_256_chunk_s1_vol_label_thinned.nrrd'  # Path where the output will be saved
sk = True
process_structures(input_nrrd_path, output_nrrd_path, sk)
