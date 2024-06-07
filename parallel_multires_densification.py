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
import argparse
from concurrent.futures import ProcessPoolExecutor

# Helper functions that are annoying to move to a separate file
def calculate_seam_iter(directed_graph, src, tgt, weights, test_size, x_pos, y_pos, z_pos):
    # Compute the residual capacity of the edges
    res = gt.boykov_kolmogorov_max_flow(directed_graph, src, tgt, weights)
    # Use the residual graph to get the max flow
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

def process_data(k, reduced_mask_data_list, reduced_raw_data_list, filename, num_seams_to_remove=120, res_index=3, save_interval=20, num_rotations=3):
    print(f"Processing data for {filename} rot={k%num_rotations}")
    mask_array_data = reduced_mask_data_list[k]
    raw_array_data = reduced_raw_data_list[k][0]
    boundary_arrays = []

    for i in range(num_seams_to_remove):
        stime = time.time()
        boundary_array = multi_res_seam_calculation(mask_array_data, res_index, dilation_amount=1)
        mask_array_data, raw_array_data = remove_voxels(mask_array_data[0], raw_array_data, boundary_array)
        mask_array_data = coarsen_image(mask_array_data, res_index)
        boundary_arrays.append(boundary_array)
        print(f"Time taken to calculate and remove seam {i} for {filename} rot={k%num_rotations}:", time.time() - stime)
        if i != 0 and (i % save_interval == 0 or i == 76):
            save_nrrd(mask_array_data, raw_array_data, filename, i, k % num_rotations)

def main(folder_name):
    current_directory = os.getcwd()
    label_folder = os.path.join(current_directory, folder_name, 'label')
    raw_folder = os.path.join(current_directory, folder_name, 'raw')

    label_files = [f for f in os.listdir(label_folder) if f.endswith('_label.nrrd')]
    raw_files = [f for f in os.listdir(raw_folder) if f.endswith('_raw.nrrd')]

    label_file_bases = {os.path.splitext(f)[0].replace('_label', '') for f in label_files}
    raw_file_bases = {os.path.splitext(f)[0].replace('_raw', '') for f in raw_files}

    common_bases = label_file_bases.intersection(raw_file_bases)

    reduced_mask_data_list = []
    reduced_raw_data_list = []
    filenames = []
    for base in common_bases:
        filenames.append(base)
        label_path = os.path.join(label_folder, f'{base}_label.nrrd')
        raw_path = os.path.join(raw_folder, f'{base}_raw.nrrd')

        mask_data, _ = nrrd.read(label_path)
        raw_data, _ = nrrd.read(raw_path)

        # Rotate the data and calculate reduced data representations
        reduced_mask_data_list.append(coarsen_image(mask_data, 3))
        reduced_raw_data_list.append(coarsen_image(raw_data, 3))

        k = 1 # Number of times to rotate the array
        for axis in [0, 1]:
            mask_data_r = np.rot90(mask_data, k, axes=(axis, (axis+1)%3))
            raw_data_r = np.rot90(raw_data, k, axes=(axis, (axis+1)%3))
            reduced_mask_data_list.append(coarsen_image(mask_data_r, 3))
            reduced_raw_data_list.append(coarsen_image(raw_data_r, 3))

    num_rotations = 3 # Number of rotations that are performed
    num_seams_to_remove = 120
    save_interval = 20
    print(len(reduced_mask_data_list))
    print(filenames)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_data, k, reduced_mask_data_list, reduced_raw_data_list, filenames[k//num_rotations], num_seams_to_remove=num_seams_to_remove, save_interval=save_interval, num_rotations=num_rotations) for k in range(len(reduced_mask_data_list))]

    for future in futures:
        future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a folder of NRRD files.')
    parser.add_argument('folder_name', type=str, help='The folder containing the label and raw subfolders with NRRD files')
    args = parser.parse_args()

    if args is None:
        args.folder_name = 'data'

    main(args.folder_name)