import numpy as np
from skimage.color import gray2rgb, label2rgb
from skimage.segmentation import find_boundaries
from skimage.util import img_as_float
from skimage.morphology import dilation, square
import random
import matplotlib.pyplot as plt

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

def save_nrrd(mask_array_data, raw_array_data, filename, num_seams_removed, rot):
    output_dir = os.path.join(os.getcwd(), 'output/densified_cubes')
    os.makedirs(output_dir, exist_ok=True)
    # Save mask_array_data[0] as NRRD with a timestamp
    mask_nrrd_path = os.path.join(output_dir, f'{filename}_iters_{num_seams_removed}_rot_{rot}_densified_label.nrrd')
    nrrd.write(mask_nrrd_path, mask_array_data[0])
    print(f"Saved mask_array_data[0] to {mask_nrrd_path}")

    # Save raw_array_data as NRRD with a timestamp
    raw_nrrd_path = os.path.join(output_dir, f'{filename}_iters_{num_seams_removed}_rot_{rot}_densified_data.nrrd')
    nrrd.write(raw_nrrd_path, raw_array_data)
    print(f"Saved raw_array_data to {raw_nrrd_path}")

def erode_structures(mask_array, iterations=1):
    unique_values = np.unique(mask_array)
    unique_values = unique_values[unique_values > 0]  # Ignore background (assuming background is 0)
    
    eroded_array = np.zeros_like(mask_array)
    
    for value in unique_values:
        structure_mask = mask_array == value
        
        # Erode the structure
        eroded_structure = ndi.binary_erosion(structure_mask, iterations=iterations)
        
        # Add the eroded structure to the new array
        eroded_array[eroded_structure] = value
    
    return eroded_array

def dilate_structures(mask_array, iterations=1):
    unique_values = np.unique(mask_array)
    unique_values = unique_values[unique_values > 0]  # Ignore background (assuming background is 0)
    
    dilated_array = np.zeros_like(mask_array)
    
    for value in unique_values:
        structure_mask = mask_array == value
        
        # Dilate the structure
        dilated_structure = ndi.binary_dilation(structure_mask, iterations=iterations)
        
        # Add the dilated structure to the new array
        dilated_array[dilated_structure] = value
    
    return dilated_array

def boundary_vertices_to_array_masked(boundary_vertices, shape, face, x_pos, y_pos, z_pos):
    z_dim, y_dim, x_dim = shape
    boundary_array = np.zeros(shape, dtype=np.int8)

    #Compute the 3d coordinates from the x,y,z positions
    for vertex in boundary_vertices:
        # print(vertex)
        x = x_pos[vertex]
        y = y_pos[vertex]
        z = z_pos[vertex]

         # Check if indices are within the valid range
        if 0 <= z < z_dim and 0 <= y < y_dim and 0 <= x < x_dim:
            boundary_array[z, y, x] = 1  # Mark the boundary vertex in the array
        else:
            print(f"Index out of bounds: z={z}, y={y}, x={x}")

    # Keep only the top-most value closest to the face in each column perpendicular to the face
    if face == 'x':
        for y in range(y_dim):
            for z in range(z_dim):
                row = boundary_array[z, y, :]
                if np.any(row == 1):
                    first_one_index = np.argmax(row == 1)
                    row[:first_one_index] = 0  # Set all values below to 0
                    row[first_one_index+1:] = 0  # Set all values above to 0
                else:
                    row[:] = 0  # No values found in this row


    elif face == 'y':
        for x in range(x_dim):
            for z in range(z_dim):
                row = boundary_array[z, :, x]
                if np.any(row == 1):
                    first_one_index = np.argmax(row == 1)
                    row[:first_one_index] = 0  # Set all values below to 0
                    row[first_one_index+1:] = 0  # Set all values above to 0
                else:
                    row[:] = 0  # No values found in this row

    elif face == 'z':
        for x in range(x_dim):
            for y in range(y_dim):
                row = boundary_array[:, y, x]
                if np.any(row == 1):
                    first_one_index = np.argmax(row == 1)
                    row[:first_one_index] = 0  # Set all values below to 0
                    row[first_one_index+1:] = 0  # Set all values above to 0
                else:
                    row[:] = 0  # No values found in this row

    return boundary_array

def create_masked_directed_energy_graph_from_mask(mask_data, direction='left', large_weight=1e8):
    z, y, x = mask_data.shape  # Dimensions of the 3D mask array
    # print(z, y, x)
    g = gt.Graph(directed=True)
    weight_prop = g.new_edge_property("int")  # Edge property for weights

    # Create vertex properties for i, j, k positions
    x_prop = g.new_vertex_property("int")
    y_prop = g.new_vertex_property("int")
    z_prop = g.new_vertex_property("int")

    # Create a mapping from mask coordinates to vertex indices
    coord_to_vertex = {}

    # Add vertices only for the non-zero elements in the mask
    # stime = time.time()
    # Find indices of non -1 elements using numpy vectorization
    non_neg_indices = np.argwhere(mask_data != -1)

    # Add all vertices at once
    g.add_vertex(len(non_neg_indices))
    
    # Assign vertices to coordinates and set properties
    for idx, (i, j, k) in enumerate(non_neg_indices):
        v = g.vertex(idx)
        coord_to_vertex[(i, j, k)] = v
        x_prop[v] = k
        y_prop[v] = j
        z_prop[v] = i

    # Define neighbor offsets based on directionality
    directions = {
        'left': [(0, 0, 1)],  # propagate right
        'right': [(0, 0, -1)],  # propagate left
        'top': [(0, 1, 0)],  # propagate downwards
        'bottom': [(0, -1, 0)],  # propagate upwards
        'front': [(1, 0, 0)],  # propagate back
        'back': [(-1, 0, 0)]  # propagate front
    }

    neighbors = directions[direction]
    
    edges = []
    weights = []

    # stime = time.time()

    for (i, j, k), current_vertex in coord_to_vertex.items():
        # Check each neighbor direction for valid connections
        for di, dj, dk in neighbors:
            ni, nj, nk = i + di, j + dj, k + dk
            if (ni, nj, nk) in coord_to_vertex:
                neighbor_vertex = coord_to_vertex[(ni, nj, nk)]
                # Determine edge weight from distance map, larger mask value means smaller weight
                weight = int(1000/(mask_data[i, j, k]+1+1e-8))
                # Add edge and assign weight
                edges.append((int(current_vertex), int(neighbor_vertex)))  # forward edge with energy value
                weights.append(weight)
                edges.append((int(neighbor_vertex), int(current_vertex)))  # backward edge with large energy value
                weights.append(int(large_weight))

        # Add each diagonal backwards neighbor inf edge, i.e., x-1, y-1 and x-1, y+1 for YX plane
        if (i, j-1, k-1) in coord_to_vertex:
            neighbor_vertex = coord_to_vertex[(i, j-1, k-1)]
            edges.append((int(current_vertex), int(neighbor_vertex)))
            weights.append(int(large_weight) + 1)
        if (i, j+1, k-1) in coord_to_vertex:
            neighbor_vertex = coord_to_vertex[(i, j+1, k-1)]
            edges.append((int(current_vertex), int(neighbor_vertex)))
            weights.append(int(large_weight) + 1)

        # Add each diagonal backwards neighbor inf edge for IK plane
        if (i-1, j, k-1) in coord_to_vertex:
            neighbor_vertex = coord_to_vertex[(i-1, j, k-1)]
            edges.append((int(current_vertex), int(neighbor_vertex)))
            weights.append(int(large_weight) + 1)
        if (i+1, j, k-1) in coord_to_vertex:
            neighbor_vertex = coord_to_vertex[(i+1, j, k-1)]
            edges.append((int(current_vertex), int(neighbor_vertex)))
            weights.append(int(large_weight) + 1)

    # Convert edges and weights to numpy arrays
    edges = np.array(edges, dtype=np.int32)
    weights = np.array(weights, dtype=np.int32)

    # Add edges to the graph using add_edge_list
    g.add_edge_list(edges)
    weight_prop.a = weights

    # print("Time taken to add edges to graph:", time.time()-stime)
    # stime = time.time()
    # Add source and sink nodes
    source = g.add_vertex()
    sink = g.add_vertex()

    # Helper function to get vertex indices for a face
    def get_face_vertices(face, coord_to_vertex, z, y, x):
        indices = []
        if face == 'left':
            for j in range(y):
                for k in range(x):
                    for i in range(z):
                        if (i, j, k) in coord_to_vertex:
                            indices.append(coord_to_vertex[(i, j, k)])
                            break
        elif face == 'right':
            for j in range(y):
                for k in range(x):
                    for i in range (z):
                        if (z-i, j, k) in coord_to_vertex:
                            indices.append(coord_to_vertex[(z-i, j, k)])
                            break
        elif face == 'top':
            for i in range(z):
                for k in range(x):
                    for j in range(y):
                        if (i, j, k) in coord_to_vertex:
                            indices.append(coord_to_vertex[(i, j, k)])
                            break
        elif face == 'bottom':
            for i in range(z):
                for k in range(x):
                    for j in range(y):
                        if (i, y-j, k) in coord_to_vertex:
                            indices.append(coord_to_vertex[(i, y-j, k)])
                            break
        elif face == 'front':
            for i in range(z):
                for j in range(y):
                    for k in range(x):
                        if (i, j, k) in coord_to_vertex:
                            indices.append(coord_to_vertex[(i, j, k)])
                            break
        elif face == 'back':
            for i in range(z):
                for j in range(y):
                    for k in range(x):
                        if (i, j, x-k) in coord_to_vertex:
                            indices.append(coord_to_vertex[(i, j, x-k)])
                            break
        return indices

    # Connect source to 'front' face
    front_vertices = get_face_vertices('front', coord_to_vertex, z, y, x)
    for v in front_vertices:
        e = g.add_edge(source, v)
        weight_prop[e] = large_weight

    # Connect sink to 'back' face
    back_vertices = get_face_vertices('back', coord_to_vertex, z, y, x)
    for v in back_vertices:
        e = g.add_edge(v, sink)
        weight_prop[e] = large_weight

    g.edge_properties["weight"] = weight_prop
    # print("Time taken to add source and sink nodes:", time.time()-stime)
    return g, source, sink, weight_prop, x_prop, y_prop, z_prop


def find_boundary_vertices(edges, part):
    """
    Find vertices that cross the partition array border.
    
    Parameters:
    edges (np.ndarray): An array of edges, where each edge is represented by a tuple (source, target).
    part (np.ndarray): A partition array where part[i] is the partition of vertex i.
    
    Returns:
    set: A set of boundary vertices.
    """

    part = np.array(part.a)
    
    # Get the source and target vertices for each edge
    source_vertices = edges[:, 0]
    target_vertices = edges[:, 1]
    
    # Find edges that cross the partition border
    cross_partition = part[source_vertices] != part[target_vertices]
    # print("edges that cross the partition:", len(cross_partition))
    
    # Get the boundary vertices
    boundary_vertices = np.unique(np.concatenate((source_vertices[cross_partition], target_vertices[cross_partition])))
    boundary_vertices = boundary_vertices[part[boundary_vertices] == 0]
    # print("boundary vertices:", len(boundary_vertices))
    
    return set(boundary_vertices)

def upscale_and_dilate_3d(array, upscale_factor, dilation_amount):
    """
    Upscale a 3D array by a given factor and then dilate the result by a specified number of voxels.
    
    Parameters:
        array (np.ndarray): The input 3D array.
        upscale_factor (int): The factor by which to upscale the array.
        dilation_amount (int): The number of voxels by which to dilate the array.
        
    Returns:
        np.ndarray: The upscaled and dilated 3D array.
    """
    # Upscale the array
    upscaled_array = ndi.zoom(array, upscale_factor, order=1)
    
    # Apply dilation
    if dilation_amount > 0:
        structure = np.ones((dilation_amount*2+1, dilation_amount*2+1, dilation_amount*2+1))
        dilated_array = ndi.binary_dilation(upscaled_array, structure=structure)
    else:
        dilated_array = upscaled_array

    return dilated_array

def coarsen_image(image, levels):
    if image.dtype != np.int16 and image.dtype != np.int32:
        normalized_arr = (image - image.min()) / (image.max() - image.min())
        image = (normalized_arr * np.iinfo(np.int16).max).astype(np.int16)

    images = [image]
    for _ in range(levels):
        image = ndi.zoom(image, 0.5, order=1)
        images.append(image)
    return images