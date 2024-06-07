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
    mask_nrrd_path = os.path.join(output_dir, f'{filename}_{num_seams_removed}_{rot}_densified_label.nrrd')
    nrrd.write(mask_nrrd_path, mask_array_data[0])
    print(f"Saved mask_array_data[0] to {mask_nrrd_path}")

    # Save raw_array_data as NRRD with a timestamp
    raw_nrrd_path = os.path.join(output_dir, f'{filename}_{num_seams_removed}_{rot}_densified_data.nrrd')
    nrrd.write(raw_nrrd_path, raw_array_data)
    print(f"Saved raw_array_data to {raw_nrrd_path}")

def boundary_vertices_to_array_masked(boundary_vertices, shape, face, x_pos, y_pos, z_pos):
    z_dim, y_dim, x_dim = shape
    boundary_array = np.zeros(shape, dtype=np.int8)

    #Compute the 3d coordinates from the x,y,z positions
    for vertex in boundary_vertices:
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

    # Create a mapping from mask coordinates to vertex indices
    coord_to_vertex = {}
    
    # Assign vertices to coordinates and set properties
    for idx, (i, j, k) in enumerate(non_neg_indices):
        v = g.vertex(idx)
        coord_to_vertex[(i, j, k)] = v
        x_prop[v] = k
        y_prop[v] = j
        z_prop[v] = i
    # print("Time taken to add vertices to coord_to_vertex:", time.time()-stime)

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
                # Determine edge weight
                weight = 10 if mask_data[ni, nj, nk] != 0 or mask_data[i, j, k] != 0 else 1
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

def remove_voxels(data_array, mask_array, boundary_array, num_voxels_to_remove=1, direction='x'):
    """
    Removes a specified number of voxels from each row in the given direction
    based on a boundary mask array.
    
    Parameters:
    data_array (np.ndarray): The original 3D array from which voxels are to be removed.
    boundary_array (np.ndarray): The boundary mask 3D array specifying the voxels to be removed.
    num_voxels_to_remove (int): Number of voxels to be removed from each row.
    direction (str): The direction in which voxels are to be removed ('x', 'y', 'z').
    
    Returns:
    np.ndarray: The resulting 3D array with the specified voxels removed.
    """
    assert data_array.shape == boundary_array.shape, f"Data array and boundary array must have the same shape. {data_array.shape} != {boundary_array.shape}"
    assert mask_array.shape == boundary_array.shape, f"Mask array and boundary array must have the same shape. {mask_array.shape} != {boundary_array.shape}"
    assert direction in ['x', 'y', 'z'], "Direction must be 'x', 'y', or 'z'."
    
    if direction == 'x':
        new_shape = (data_array.shape[0], data_array.shape[1], data_array.shape[2] - num_voxels_to_remove)
        result_data_array = np.zeros(new_shape, dtype=data_array.dtype)
        result_mask_array = np.zeros(new_shape, dtype=mask_array.dtype)
        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                data_row = data_array[i, j, :]
                boundary_row = boundary_array[i, j, :]
                keep_indices = np.where(boundary_row == 0)[0]
                keep_indices = keep_indices[:new_shape[2]]  # Only keep up to the new size
                result_data_array[i, j, :] = data_row[keep_indices][:new_shape[2]]
                result_mask_array[i, j, :] = mask_array[i, j, keep_indices][:new_shape[2]]
        
        # Create padded arrays with -1
        padded_data_array = -np.ones_like(data_array, dtype=data_array.dtype)
        padded_mask_array = -np.ones_like(mask_array, dtype=mask_array.dtype)
        
        # Copy the result arrays into the padded arrays
        padded_data_array[:, :, :new_shape[2]] = result_data_array
        padded_mask_array[:, :, :new_shape[2]] = result_mask_array
    
    elif direction == 'y':
        new_shape = (data_array.shape[0], data_array.shape[1] - num_voxels_to_remove, data_array.shape[2])
        result_data_array = np.zeros(new_shape, dtype=data_array.dtype)
        result_mask_array = np.zeros(new_shape, dtype=mask_array.dtype)
        for i in range(data_array.shape[0]):
            for k in range(data_array.shape[2]):
                data_row = data_array[i, :, k]
                boundary_row = boundary_array[i, :, k]
                keep_indices = np.where(boundary_row == 0)[0]
                keep_indices = keep_indices[:new_shape[1]]  # Only keep up to the new size
                result_data_array[i, :, k] = data_row[keep_indices][:new_shape[1]]
                result_mask_array[i, :, k] = mask_array[i, keep_indices, k][:new_shape[1]]
        
        # Create padded arrays with -1
        padded_data_array = -np.ones_like(data_array, dtype=data_array.dtype)
        padded_mask_array = -np.ones_like(mask_array, dtype=mask_array.dtype)
        
        # Copy the result arrays into the padded arrays
        padded_data_array[:, :new_shape[1], :] = result_data_array
        padded_mask_array[:, :new_shape[1], :] = result_mask_array
    
    elif direction == 'z':
        new_shape = (data_array.shape[0] - num_voxels_to_remove, data_array.shape[1], data_array.shape[2])
        result_data_array = np.zeros(new_shape, dtype=data_array.dtype)
        result_mask_array = np.zeros(new_shape, dtype=mask_array.dtype)
        for j in range(data_array.shape[1]):
            for k in range(data_array.shape[2]):
                data_row = data_array[:, j, k]
                boundary_row = boundary_array[:, j, k]
                keep_indices = np.where(boundary_row == 0)[0]
                keep_indices = keep_indices[:new_shape[0]]  # Only keep up to the new size
                result_data_array[:, j, k] = data_row[keep_indices][:new_shape[0]]
                result_mask_array[:, j, k] = mask_array[keep_indices, j, k][:new_shape[0]]
        
        # Create padded arrays with -1
        padded_data_array = -np.ones_like(data_array, dtype=data_array.dtype)
        padded_mask_array = -np.ones_like(mask_array, dtype=mask_array.dtype)
        
        # Copy the result arrays into the padded arrays
        padded_data_array[:new_shape[0], :, :] = result_data_array
        padded_mask_array[:new_shape[0], :, :] = result_mask_array
    
    return padded_data_array, padded_mask_array

def mark_boundaries_color(image, label_img, color=None, outline_color=None, mode='outer', background_label=0, dilation_size=1):
    """Return image with boundaries between labeled regions highlighted with consistent colors derived from labels.

    Parameters:
    - image: Input image.
    - label_img: Image with labeled regions.
    - color: Ignored in this version.
    - outline_color: If specified, use this color for the outline. Otherwise, use the same as boundary.
    - mode: Choose 'inner', 'outer', or 'thick' to define boundary type.
    - background_label: Label to be treated as the background.
    - dilation_size: Size of the dilation square for the boundaries.

    Returns:
    - Image with boundaries highlighted.
    """
    # Ensure input image is in float and has three channels
    float_dtype = np.float32  # Use float32 for efficiency
    marked = img_as_float(image, force_copy=True).astype(float_dtype, copy=False)
    if marked.ndim == 2:
        marked = gray2rgb(marked)

    # Create a color map normalized by the number of unique labels
    unique_labels = np.unique(label_img)
    color_map = plt.get_cmap('nipy_spectral')  # You can change 'nipy_spectral' to any other colormap

    # Find boundaries and apply colors
    boundaries = find_boundaries(label_img, mode=mode, background=background_label)
    for label in unique_labels:
        if label == background_label:
            continue
        # Normalize label value to the range of the colormap
        normalized_color = color_map(label / np.max(unique_labels))[:3]  # Get RGB values only
        label_boundaries = find_boundaries(label_img == label, mode=mode)
        label_boundaries = dilation(label_boundaries, square(dilation_size))
        marked[label_boundaries] = normalized_color
        if outline_color is not None:
            outlines = dilation(label_boundaries, square(dilation_size + 1))
            marked[outlines] = outline_color
        else:
            marked[label_boundaries] = normalized_color

    return marked

def mark_boundaries_multicolor(image, label_img, color=None, outline_color=None, mode='outer', background_label=0, dilation_size=1):
    """Return image with boundaries between labeled regions highlighted with consistent colors.

    Parameters are the same as in the original function but color is ignored if provided.
    """
    # Ensure input image is in float and has three channels
    float_dtype = np.float32  # Use float32 for efficiency
    marked = img_as_float(image, force_copy=True).astype(float_dtype, copy=False)
    if marked.ndim == 2:
        marked = gray2rgb(marked)

    # Generate consistent colors for each unique label in label_img
    unique_labels = np.unique(label_img)
    color_map = {label: consistent_color(label) for label in unique_labels if label != background_label}

    # Find boundaries and apply colors
    boundaries = find_boundaries(label_img, mode=mode, background=background_label)
    for label, color in color_map.items():
        label_boundaries = find_boundaries(label_img == label, mode=mode)
        label_boundaries = dilation(label_boundaries, square(dilation_size))
        if outline_color is not None:
            outlines = dilation(label_boundaries, square(dilation_size))
            marked[outlines] = outline_color
        marked[label_boundaries] = color

    return marked