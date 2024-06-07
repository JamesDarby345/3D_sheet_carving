import nrrd
import numpy as np
import os
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from skimage.segmentation import mark_boundaries
from scipy import ndimage as ndi
from helper import *
import graph_tool.all as gt
import plotly.graph_objects as go
import time

current_directory = os.getcwd()
filename = "manual_2"
label_path = f"{current_directory}/data/label/{filename}_label.nrrd"
raw_data_path = f"{current_directory}/data/raw/{filename}_raw.nrrd"

mask_data, mask_header = nrrd.read(label_path)
raw_data, raw_header = nrrd.read(raw_data_path)

#rotate the data and calculate reduced data representations
#lists of the 3 rotations of the reduced mask and raw data
reduced_mask_data_list = []
reduced_raw_data_list = []

reduced_mask_data_list.append(coarsen_image(mask_data, 3))
reduced_raw_data_list.append(coarsen_image(raw_data, 3))

k = 1 # number of times to rotate the array
axis = 0 #0 for x, 1 for y and 2 for z
mask_data_r = np.rot90(mask_data, k, axes=(axis, (axis+1)%3))
raw_data_r = np.rot90(raw_data, k, axes=(axis, (axis+1)%3))

reduced_mask_data_list.append(coarsen_image(mask_data_r, 3))
reduced_raw_data_list.append(coarsen_image(raw_data_r, 3))

axis = 1
mask_data_r = np.rot90(mask_data, k, axes=(axis, (axis+1)%3))
raw_data_r = np.rot90(raw_data, k, axes=(axis, (axis+1)%3))

reduced_mask_data_list.append(coarsen_image(mask_data_r, 3))
reduced_raw_data_list.append(coarsen_image(raw_data_r, 3))
num_rotations = 3
#index 0 of a list element is full res, each further index is 2x lower res, 256, 128, 64, 32

# Helper functions that are annoying to move to a seperate file
def calculate_seam_iter(directed_graph, src, tgt, weights, test_size, x_pos, y_pos, z_pos):
    # Compute the residual capactiy of the edges
    res = gt.boykov_kolmogorov_max_flow(directed_graph, src, tgt, weights) #time complexity: edges * vertices^2 * abs(min cut) 
    #use the residual graph to get the max flow
    flow = sum(weights[e] - res[e] for e in tgt.in_edges())
    # Determine the minimum cut partition
    part = gt.min_st_cut(directed_graph, src, weights, res)

    # Find the boundary vertices
    boundary_vertices = find_boundary_vertices(np.array(directed_graph.get_edges()), part)
    shape = (test_size, test_size, test_size)

    # Convert the boundary vertices to a 3D array
    boundary_array = boundary_vertices_to_array_masked(boundary_vertices, shape, 'x', x_pos, y_pos, z_pos)
    
    return boundary_array, flow

def multi_res_seam_iter(res_index, mask_array_data, b_arr_up):
    masked_array = mask_array_data[res_index].copy().astype(np.int16)
    masked_array[b_arr_up == 0] = -1
    stime= time.time()
    directed_graph, src, tgt, weights, x_pos, y_pos, z_pos = create_masked_directed_energy_graph_from_mask(masked_array)
    # print(f"Time taken to create graph {res_index}:", time.time()-stime)
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

# Define the function to be executed in parallel
def process_data(k, reduced_mask_data_list, reduced_raw_data_list, filename, num_seams_to_remove=120, res_index=3, save_interval=20):
    print(f"Processing data for k={k}")
    mask_array_data = reduced_mask_data_list[k]
    raw_array_data = reduced_raw_data_list[k][0]
    boundary_arrays = []

    for i in range(num_seams_to_remove):
        stime = time.time()
        boundary_array = multi_res_seam_calculation(mask_array_data, res_index, dilation_amount=1)
        mask_array_data, raw_array_data = remove_voxels(mask_array_data[0], raw_array_data, boundary_array)
        mask_array_data = coarsen_image(mask_array_data, res_index)
        boundary_arrays.append(boundary_array)
        print(f"Time taken to calculate and remove seam {i} for k={k}:", time.time() - stime)
        if i !=0 and (i % save_interval == 0 or i == 76):
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # filename = f"output_{k}_{i}_{timestamp}.nrrd"
            save_nrrd(mask_array_data, raw_array_data, filename, i, k % num_rotations)

# Parallel execution using ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor
if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_data, k, reduced_mask_data_list, reduced_raw_data_list, filename) for k in range(len(reduced_mask_data_list))]

    # Wait for all futures to complete
    for future in futures:
        future.result()
