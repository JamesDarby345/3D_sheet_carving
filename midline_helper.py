import numpy as np
import random
import matplotlib.pyplot as plt

import nrrd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from helper import *
import graph_tool.all as gt
import plotly.graph_objects as go
import time

import os
import numpy as np
import nrrd
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize
# import skimage
from scipy.spatial import cKDTree
from scipy import ndimage

def create_outline_with_ray_projection(array, value, distance):
    """
    Create an outline by projecting rays from each edge and counting intersections with the label.
    
    Args:
    array (numpy.ndarray): 3D input array
    value (int or float): The label value to check for and fill with
    distance (int): Distance from the edge to project rays
    
    Returns:
    numpy.ndarray: 3D array with the filled outline
    """
    
    depth, height, width = array.shape
    temp_count = np.zeros_like(array, dtype=int)
    
    # Helper function to project rays and count intersections
    def project_rays(slice_2d, axis):
        nonlocal temp_count
        for i in range(slice_2d.shape[0]):
            for j in range(slice_2d.shape[1]):
                if slice_2d[i, j] == value:
                    if axis == 0:
                        temp_count[0:distance+1, i, j] += 1
                        temp_count[depth-distance-1:depth, i, j] += 1
                    elif axis == 1:
                        temp_count[i, 0:distance+1, j] += 1
                        temp_count[i, height-distance-1:height, j] += 1
                    else:
                        temp_count[i, j, 0:distance+1] += 1
                        temp_count[i, j, width-distance-1:width] += 1
    
    # Project rays from each face
    project_rays(array[distance], 0)  # Front face
    project_rays(array[-distance], 0)  # Back face
    project_rays(array[:, distance, :], 1)  # Left face
    project_rays(array[:, -distance, :], 1)  # Right face
    project_rays(array[:, :, distance], 2)  # Top face
    project_rays(array[:, :, -distance], 2)  # Bottom face
    
    # Create the final outline
    result = np.copy(array)
    result[temp_count >= 2] = value
    
    return result
    

def connect_to_edge_3d(array, value, distance=1, use_x=True, use_y=True, use_z=True, create_outline=False):
    """
    Connect values in a 3D array to the nearest edge using straight lines,
    if they are within the specified distance from the edge.
    When an axis is disabled, vertices closest to that axis are not connected.
    Optionally creates an outline using ray projection method.
    
    Args:
    array (numpy.ndarray): 3D input array
    value (int or float): The value to connect to the edge
    distance (int): Maximum distance from the edge to connect (default 1)
    use_x (bool): Whether to allow connections along the x-axis (default True)
    use_y (bool): Whether to allow connections along the y-axis (default True)
    use_z (bool): Whether to allow connections along the z-axis (default True)
    create_outline (bool): Whether to create the outline using ray projection (default False)
    
    Returns:
    numpy.ndarray: Modified 3D array with values connected to the edge and optional outline
    """
    # Create a copy of the input array
    result = np.copy(array)
    
    # Get the dimensions of the array
    depth, height, width = array.shape
    
    # Find coordinates of voxels with the specified value
    coords = np.argwhere(array == value)
    
    for z, y, x in coords:
        # Check if the voxel is within the specified distance from any edge
        if (z < distance or z >= depth - distance or
            y < distance or y >= height - distance or
            x < distance or x >= width - distance):
            
            # Determine the nearest edge for each dimension
            nearest_z = min(z, depth - 1 - z) if use_z else float('inf')
            nearest_y = min(y, height - 1 - y) if use_y else float('inf')
            nearest_x = min(x, width - 1 - x) if use_x else float('inf')
            
            # Find the dimension with the minimum distance to edge
            min_dist = min(nearest_z, nearest_y, nearest_x)
            
            # Only connect if the nearest edge is on an enabled axis
            if min_dist != float('inf'):
                if min_dist == nearest_z:
                    # Connect to the nearest z-edge
                    z_edge = 0 if z < depth // 2 else depth - 1
                    result[min(z, z_edge):max(z, z_edge)+1, y, x] = value
                elif min_dist == nearest_y:
                    # Connect to the nearest y-edge
                    y_edge = 0 if y < height // 2 else height - 1
                    result[z, min(y, y_edge):max(y, y_edge)+1, x] = value
                elif min_dist == nearest_x:
                    # Connect to the nearest x-edge
                    x_edge = 0 if x < width // 2 else width - 1
                    result[z, y, min(x, x_edge):max(x, x_edge)+1] = value
    
    # Create outline if requested
    if create_outline:
        result = create_outline_with_ray_projection(result, value, distance)
    
    return result

def generate_volume_roi(input_array, erode_dilate_iters=10):
    """
    Generates a Region of Interest (ROI) for a 3D volume by dilating the non-zero structure,
    filling morphological tunnels, and creating a mask that closely covers the volume.

    Args:
    input_array (numpy.ndarray): 3D input array
    erode_dilate_iters (int): Radius for dilation operation
    hole_size (int): Maximum size of holes/tunnels to fill

    Returns:
    numpy.ndarray: Binary mask representing the ROI
    """
    # Ensure the input is a 3D numpy array
    if input_array.ndim != 3:
        raise ValueError("Input must be a 3D numpy array")

    # Create a binary mask of non-zero elements
    binary_mask = (input_array > 0).astype(np.uint8)
    padded_structure = np.pad(binary_mask, pad_width=erode_dilate_iters, mode='constant', constant_values=0)

    # Dilate the binary mask
    dilated_mask = ndimage.binary_dilation(padded_structure, 
                                           structure=ndimage.generate_binary_structure(3, 3),
                                           iterations=erode_dilate_iters)

    # Fill holes in the dilated mask
    filled_mask = ndimage.binary_fill_holes(dilated_mask)

    result = np.zeros_like(input_array, dtype=np.uint8)
    
    # dilate_iters = erode_dilate_iters -1
    
    eroded_padded_structure = ndimage.binary_erosion(filled_mask, iterations=erode_dilate_iters)

    eroded_structure = eroded_padded_structure[
        erode_dilate_iters:-erode_dilate_iters,
        erode_dilate_iters:-erode_dilate_iters,
        erode_dilate_iters:-erode_dilate_iters
    ]
    if eroded_structure.shape != input_array.shape:
        eroded_structure = np.zeros_like(input_array)
    result[eroded_structure] = 1
    
    # Create the final ROI mask
    roi_mask = result.astype(np.int8)
    roi_mask[roi_mask == 0] = -1

    return roi_mask

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

def add_diagonal_edges(coord_to_vertex, current_vertex, i, j, k, weight_array, large_weight):
    edges = []
    weights = []
    
    for edge_dist, weight in enumerate(weight_array, start=1):
        if weight == -1:
            continue
        # Check diagonal neighbors in JK plane
        if (i, j-edge_dist, k-edge_dist) in coord_to_vertex:
            neighbor_vertex = coord_to_vertex[(i, j-edge_dist, k-edge_dist)]
            edges.append((int(current_vertex), int(neighbor_vertex)))
            weights.append(int(large_weight) + weight)
        
        if (i, j+edge_dist, k-edge_dist) in coord_to_vertex:
            neighbor_vertex = coord_to_vertex[(i, j+edge_dist, k-edge_dist)]
            edges.append((int(current_vertex), int(neighbor_vertex)))
            weights.append(int(large_weight) + weight)
        
        # Check diagonal neighbors in IK plane
        if (i-edge_dist, j, k-edge_dist) in coord_to_vertex:
            neighbor_vertex = coord_to_vertex[(i-edge_dist, j, k-edge_dist)]
            edges.append((int(current_vertex), int(neighbor_vertex)))
            weights.append(int(large_weight) + weight)
        
        if (i+edge_dist, j, k-edge_dist) in coord_to_vertex:
            neighbor_vertex = coord_to_vertex[(i+edge_dist, j, k-edge_dist)]
            edges.append((int(current_vertex), int(neighbor_vertex)))
            weights.append(int(large_weight) + weight)
    
    return edges, weights

def create_masked_directed_energy_graph_6_connect(mask_data, large_weight=1e8, weight_array=[1e8]):
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
    
    edges = []
    weights = []

    # stime = time.time()
    edge_dist = 1 #distance between vertices in the graph to connect edges with, default 1 for direct connections
    for (i, j, k), current_vertex in coord_to_vertex.items():
        # Check the 6 direct neighbors
        for dx, dy, dz in [(0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)]:
            neighbor_coord = (i + dx * edge_dist, j + dy * edge_dist, k + dz * edge_dist)
            if neighbor_coord in coord_to_vertex:
                neighbor_vertex = coord_to_vertex[neighbor_coord]
                weight = int(1000/(mask_data[i, j, k]))
                
                edges.append((int(current_vertex), int(neighbor_vertex)))
                weights.append(weight)

        #backward diagonal edges enforce 45 degree angle maximum from the front face
        #also force connectivity of the sheet
        # # Add each diagonal backwards neighbor inf edge, i.e., x-1, y-1 and x-1, y+1 for YX plane
        #val of -1 skips adding an edge to that backwards diagonal edge value
        diag_edges, diag_weights = add_diagonal_edges(coord_to_vertex, current_vertex, i, j, k, weight_array, 0)
        edges.extend(diag_edges)
        weights.extend(diag_weights)

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

def create_masked_directed_energy_graph_from_mask_non_monotonic(mask_data, direction='left', large_weight=1e8, weight_array=[1e8]):
    z, y, x = mask_data.shape  # Dimensions of the 3D mask array
    # print(z, y, x)
    g = gt.Graph(directed=True)
    weight_prop = g.new_edge_property("int")  # Edge property for weights

    # Create vertex properties for i, j, k positionsß
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
            back_edge_dist = 1
            di = di * back_edge_dist
            dj = dj * back_edge_dist
            dk = dk * back_edge_dist
            ni, nj, nk = i + di, j + dj, k + dk
            if (ni, nj, nk) in coord_to_vertex:
                neighbor_vertex = coord_to_vertex[(ni, nj, nk)]
                # Determine edge weight from distance map, larger mask value means smaller weight
                if mask_data[i, j, k] <= 0:
                    weight = 1e6
                else:
                    weight = int(1000/(mask_data[i, j, k]))
                # Add edge and assign weight
                edges.append((int(current_vertex), int(neighbor_vertex)))  # forward edge with energy value
                weights.append(weight)
                edges.append((int(neighbor_vertex), int(current_vertex)))  # backward edge with large energy value
                # weights.append(int(large_weight)) 
                weights.append(1)

        #backward diagonal edges enforce 45 degree angle maximum from the front face
        #also force connectivity of the sheet
        # # Add each diagonal backwards neighbor inf edge, i.e., x-1, y-1 and x-1, y+1 for YX plane
        #val of -1 skips adding an edge to that backwards diagonal edge value
        diag_edges, diag_weights = add_diagonal_edges(coord_to_vertex, current_vertex, i, j, k, weight_array, 0)
        edges.extend(diag_edges)
        weights.extend(diag_weights)

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
                # weight = int(1000/(mask_data[i, j, k]+1+1e-8))
                if mask_data[i, j, k] <= 0:
                    weight = 1e6
                else:
                    weight = int(1000/(mask_data[i, j, k]))
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

def process_array_with_bounding_box(input_array):
    """
    Takes a 3D numpy array with a non-zero valued structure, calculates the bounding box
    of that structure, sets all voxels outside the bounding box to -1, and returns the result.

    Args:
    input_array (numpy.ndarray): 3D input array

    Returns:
    numpy.ndarray: Processed 3D array with voxels outside the bounding box set to -1
    """
    # Ensure the input is a 3D numpy array
    if input_array.ndim != 3:
        raise ValueError("Input must be a 3D numpy array")

    # Find the indices of non-zero elements
    non_zero_indices = np.nonzero(input_array)

    # If there are no non-zero elements, return an array of all -1s
    if len(non_zero_indices[0]) == 0:
        return np.full_like(input_array, -1)

    # Calculate the bounding box
    min_z, max_z = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    min_y, max_y = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    min_x, max_x = np.min(non_zero_indices[2]), np.max(non_zero_indices[2])

    # Create a copy of the input array
    result_array = input_array.copy()

    # Set all voxels outside the bounding box to -1
    result_array[:min_z, :, :] = -1
    result_array[max_z+1:, :, :] = -1
    result_array[:, :min_y, :] = -1
    result_array[:, max_y+1:, :] = -1
    result_array[:, :, :min_x] = -1
    result_array[:, :, max_x+1:] = -1

    return result_array

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