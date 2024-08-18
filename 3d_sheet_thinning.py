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

def process_label_with_roi(labeled_array, label, roi_array):
    """
    Compute the distance map for a specific label within a given ROI.
    
    Args:
    labeled_array (numpy.ndarray): 3D array with labeled regions
    label (int): The label to process
    roi_array (numpy.ndarray): 3D array defining the ROI
    
    Returns:
    numpy.ndarray: Distance map with positive values inside the label and negative values outside (within ROI)
    """
    # Create a binary mask for the current label
    mask = (labeled_array == label)
    
    # Compute the distance transform within the label
    pos_distance_map = ndimage.distance_transform_edt(mask)
    
    # Compute the distance transform outside the label (within ROI)
    neg_distance_map = ndimage.distance_transform_edt(~mask)
    
    # Create the final distance map
    distance_map = pos_distance_map - neg_distance_map
    
    # Apply the ROI mask
    roi_mask = (roi_array != 0)
    distance_map[~roi_mask] = 0
    
    return distance_map

def create_label_distance_map_with_roi(labeled_array, roi_array, max_workers=None):
    """
    Create a distance map for all labels in the labeled_array within the given ROI.
    
    Args:
    labeled_array (numpy.ndarray): 3D array with labeled regions
    roi_array (numpy.ndarray): 3D array defining the ROI
    max_workers (int): Maximum number of worker threads
    
    Returns:
    numpy.ndarray: Combined distance map for all labels
    """
    # Ensure labeled_array and roi_array have the same shape
    assert labeled_array.shape == roi_array.shape, "labeled_array and roi_array must have the same shape"
    
    # Get unique labels, excluding background (assumed to be 0)
    labels = np.unique(labeled_array)
    labels = labels[labels != 0]
    
    # Create an output array with the same shape as the input
    output = np.zeros_like(labeled_array, dtype=float)
    
    # Use ThreadPoolExecutor for parallelization
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for each label
        future_to_label = {executor.submit(process_label_with_roi, labeled_array, label, roi_array): label for label in labels}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_label):
            label = future_to_label[future]
            try:
                distance_map = future.result()
                # Add the distance map for this label to the output
                output += distance_map
            except Exception as exc:
                print(f'Label {label} generated an exception: {exc}')
    
    return output

def prepare_distance_map(distance_map, roi_mask, value_to_add=0):
    #add value delta between the label and out of label distances
    mask = (distance_map > 0)
    distance_map[mask] += value_to_add

    #mask areas outside the roi and normalize the distance map
    distance_map[roi_mask == -1] = -1
    distance_map += abs(distance_map.min())+1
    distance_map[roi_mask == -1] = -1
    return distance_map

def process_single_label(label_data, label_value, output_path, fill_holes=False):
    # Create a binary mask for the specified label
    mask = (label_data == label_value)
    # nrrd.write('mask.nrrd', mask.astype(np.uint8))
    
    
    if fill_holes:
        stime = time.time()
        # distance_map = process_array_with_bounding_box(distance_map)
        roi_mask = generate_volume_roi(mask, erode_dilate_iters=10)
        nrrd.write('output/roi_mask.nrrd', roi_mask.astype(np.uint8))
        print(f"Time taken to process ROI: {time.time() - stime:.2f} seconds")
    else:
        roi_mask = mask

    # Compute the distance map within the label
    stime = time.time()
    distance_map = create_label_distance_map_with_roi(mask, roi_mask)
    print(f"Time taken to calculate distance map: {time.time() - stime:.2f} seconds")

    #Mask out areas outside the ROI, normalize the distance map and remask
    distance_map = prepare_distance_map(distance_map, roi_mask, value_to_add=0)
    print("0's in dist map (should be 0):", np.sum(distance_map == 0))

    # # Mask out areas outside the label
    if not fill_holes:
        distance_map[~mask] = -1
    nrrd.write('output/distance_map.nrrd', distance_map.astype(np.float32))

  
    # Create the energy graph
    stime = time.time()
    weight_array = [-1]
    # directed_graph, src, tgt, weights, x_pos, y_pos, z_pos = create_masked_directed_energy_graph_6_connect(distance_map, weight_array=weight_array)
    if not fill_holes:
        directed_graph, src, tgt, weights, x_pos, y_pos, z_pos = create_masked_directed_energy_graph_from_mask_non_monotonic(distance_map, weight_array=weight_array)
    else:
        directed_graph, src, tgt, weights, x_pos, y_pos, z_pos = create_masked_directed_energy_graph_from_mask(distance_map)
    print(f"Time taken to create energy graph: {time.time() - stime:.2f} seconds")

    # Calculate the seam
    stime = time.time()
    seam_array, flow = calculate_seam_iter(directed_graph, src, tgt, weights, distance_map.shape[0], x_pos, y_pos, z_pos)
    # seam_array = multi_res_seam_calculation(distance_map, res_index=1, upscale_factor=2, dilation_amount=1)
    print(f"Time taken to calculate seam: {time.time() - stime:.2f} seconds")

    # Convert the boundary vertices to a 3D array
    # seam_array = boundary_vertices_to_array_masked(boundary_array, distance_map.shape, 'x', x_pos, y_pos, z_pos)

    # Save the seam array as an NRRD file
    nrrd.write(output_path, seam_array.astype(np.uint8))

    return seam_array

def collapse_to_2d(structure_3d):
    return np.any(structure_3d, axis=2)

def identify_morphological_holes(structure_2d):    
    # Flood fill from the border
    filled = ndimage.binary_fill_holes(structure_2d)
    
    # Identify holes
    holes = filled & ~structure_2d
    holes = ndimage.binary_dilation(holes, iterations=1)
    
    return holes

def get_border_voxels(structure_3d, holes_2d):
    border_voxels = []
    for x in range(structure_3d.shape[0]):
        for y in range(structure_3d.shape[1]):
            if holes_2d[x, y]:
                z_values = np.where(structure_3d[x, y, :])[0]
                if len(z_values) > 0:
                    border_voxels.append((x, y, z_values[0]))
                    if len(z_values) > 1:
                        border_voxels.append((x, y, z_values[-1]))
    return border_voxels

def create_border_structure(structure_3d, border_voxels):
    border_structure = np.zeros((structure_3d.shape[0], structure_3d.shape[1], structure_3d.shape[2]))
    d_borders = np.zeros((structure_3d.shape[0], structure_3d.shape[1]))
    for x, y, z in border_voxels:
        # print(x, y, z)
        border_structure[x, y, z] = 1
        d_borders[x,y] = 1
    return border_structure, d_borders

def fill_holes(structure_3d, border_voxels):
    filled_structure = structure_3d.copy()
    for x, y, _ in border_voxels:
        z_values = np.where(structure_3d[x, y, :])[0]
        if len(z_values) >= 2:
            z_min, z_max = z_values[0], z_values[-1]
            filled_structure[x, y, z_min:z_max+1] = 1
        elif len(z_values) == 1:
            filled_structure[x, y, z_values[0]] = 1
    return filled_structure

def find_segments(column):
    # Convert to integers for diff operation
    column_int = column.astype(int)
    # Find the indices where the value changes
    diff = np.diff(column_int)
    start_indices = np.where(diff == 1)[0] + 1
    stop_indices = np.where(diff == -1)[0]
    
    # Handle edge cases
    if column[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if column[-1]:
        stop_indices = np.append(stop_indices, len(column) - 1)
    
    return list(zip(start_indices, stop_indices))

def bresenham_3d(x1, y1, z1, x2, y2, z2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1
    
    # Driving axis is X-axis
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            yield (x1, y1, z1)
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
    # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            yield (x1, y1, z1)
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
    # Driving axis is Z-axis
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            yield (x1, y1, z1)
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
    yield (x2, y2, z2)


def fill_line(holes_3d, col, start, stop):
    top_z = np.argmax(holes_3d[col, start, :])
    bottom_z = np.argmax(holes_3d[col, stop, :])

    for x, y, z in bresenham_3d(col, start, top_z, col, stop, bottom_z):
        holes_3d[x, y, z] = 2

def create_3d_array(holes_2d, border_voxels):
    # Assuming holes_2d is a 2D NumPy array
    width, height = holes_2d.shape
    
    # Create an empty 3D array with the same x,y,z dimension as holes_2d
    holes_3d = np.zeros((height, width, width))

    # Assign border voxels in the 3D array
    for x, y, z in border_voxels:
        holes_3d[x, y, z] = 1 
    
    # Label connected components
    labeled_array, num_features = ndimage.label(holes_2d)
    
    # Iterate through each connected component
    for component in range(1, num_features + 1):
        component_mask = labeled_array == component
        component_array = holes_2d * component_mask
        
        # Iterate through each column, col is y val, segments start/stop is x val
        for col in range(component_array.shape[0]):
            column = component_array[col]
            segments = find_segments(column)
            
            if segments:
                for start, stop in segments:
                    fill_line(holes_3d, col, start, stop)

    return holes_3d

def process_3d_structure(structure_3d):
    # Collapse to 2D
    print(f'Structure shape: {structure_3d.shape}', np.sum(structure_3d))
    structure_2d = collapse_to_2d(structure_3d)
    
    # Identify holes
    holes_2d = identify_morphological_holes(structure_2d)
    nrrd.write('output/holes_2d.nrrd', holes_2d.astype(np.uint8))
    
    # Get border voxels
    border_voxels = get_border_voxels(structure_3d, holes_2d)
    print(f'Number of border voxels: {len(border_voxels)}') 
    print(f'Border voxels 1st value: {border_voxels[0]}')
    
    # Create border structure
    border_structure, border_2d = create_border_structure(structure_3d, border_voxels)
    print(f'Border structure shape: {border_structure.shape}', np.sum(border_structure))
    nrrd.write('2d_border.nrrd', border_2d.astype(np.uint8))
    
    # Save border structure as NRRD
    nrrd.write('output/hole_borders.nrrd', border_structure.astype(np.uint8))

    # Create 3D array
    holes_3d = create_3d_array(holes_2d, border_voxels)
    nrrd.write('output/filled_holes_3d.nrrd', holes_3d.astype(np.uint8))
    
    # Fill holes
    filled_structure = fill_holes(structure_3d, border_voxels)
    print(f'Filled structure shape: {filled_structure.shape}', np.sum(filled_structure))
    return filled_structure

def process_structures(nrrd_path, output_path, pad_to_remove_edge_effects=True, fill_holes=False):
    # print(skimage.__version__)
    # Load the data
    data, header = nrrd.read(nrrd_path)
    # data = filter_and_reassign_labels(data, 300)  # Filter out small disconnected components
    label_val = 6
    mask = data == label_val
    data[mask != 1] = 0

    if pad_to_remove_edge_effects:
        pad_amount = 10 #similar to erode dilate iterations value
        data = np.pad(data, pad_amount, mode='constant', constant_values=0)
        
        #TODO: use adjacent labels to inform the connection instead of straight projection to edge
        data = connect_to_edge_3d(data, label_val, pad_amount+1, use_z=True, create_outline=False)
        nrrd.write('output/padded_data.nrrd', data.astype(np.uint8))
    thinned_data = np.zeros_like(data, dtype=np.uint8)  # Ensure thinned_data is of type uint8
    
    stime = time.time()
    thinned_data = process_single_label(data, label_val, output_path, fill_holes=fill_holes)
    print(f"Time taken: {time.time() - stime:.2f} seconds")
    nrrd.write('output/thinned_data_padded.nrrd', thinned_data.astype(np.uint8))

    if pad_to_remove_edge_effects:
        pad_amount +=1
        print(f"Thinned data shape: {thinned_data.shape}")
        thinned_data = thinned_data[pad_amount:-pad_amount, pad_amount:-pad_amount, pad_amount:-pad_amount]
        thinned_data = np.pad(thinned_data, 1, mode='constant', constant_values=0)
        print(f"Thinned data shape post padding: {thinned_data.shape}")

    thinned_data = process_3d_structure(thinned_data)
    # Save the thinned structures as a new NRRD file
    # space_origin = header['space origin']
    # space_origin += pad_amount
    # header['space origin'] = space_origin
    nrrd.write(output_path, thinned_data.astype(np.uint8), header)

# Example usage:
current_directory = os.getcwd()
input_nrrd_path = f'{current_directory}/data/label/09936_03280_04560_zyx_256_chunk_s1_vol_label.nrrd'  # Path to your NRRD file
output_nrrd_path = f'{current_directory}/output/09936_03280_04560_zyx_256_chunk_s1_vol_label_thinned.nrrd'  # Path where the output will be saved
process_structures(input_nrrd_path, output_nrrd_path, fill_holes=False)
