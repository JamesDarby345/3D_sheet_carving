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
import cv2
from scipy import ndimage as ndi
from helper import *
import graph_tool.all as gt
import plotly.graph_objects as go
import time

import concurrent.futures
import time

def update_edges_for_vertex_subset(graph, vertex_subset_indices, new_weight):
    """
    Updates the weights of all outgoing edges for a subset of vertices.
    
    Parameters:
    graph (Graph): The graph object.
    vertex_subset_indices (list): A subset of vertex indices whose outgoing edges need to be updated.
    new_weight (float): The new weight value to be set for the outgoing edges.
    
    Returns:
    list: A list of (source_index, target_index, new_weight) tuples to update in the main process.
    """
    updates = []
    for v_index in vertex_subset_indices:
        vertex = graph.vertex(v_index)
        for edge in vertex.in_edges():
            source_index = int(edge.source())
            target_index = int(edge.target())
            updates.append((source_index, target_index, new_weight))
    return updates

def update_outgoing_edges_p(graph, vertices, new_weight):
    """
    Updates the weights of all outgoing edges of the given vertices in a directed graph and returns the updated weights.
    
    Parameters:
    graph (Graph): A directed graph from graph_tool.
    vertices (list): A list of vertices whose outgoing edge weights need to be updated.
    new_weight (float): The new weight value to be set for the outgoing edges.
    
    Returns:
    PropertyMap: The updated edge weight property map.
    """

    # Ensure the graph has an edge weight property map
    if 'weight' in graph.edge_properties:
        weights = graph.edge_properties['weight']
    else:
        weights = graph.new_edge_property("int")
        graph.edge_properties['weight'] = weights

    # Determine the number of processes to use
    num_processes = min(len(vertices), concurrent.futures.ProcessPoolExecutor()._max_workers)

    # Split vertex indices into equal-sized chunks for each process
    vertex_chunks = np.array_split(vertices, num_processes)

    # Parallelize the edge update process using ProcessPoolExecutor
    stime = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(update_edges_for_vertex_subset, graph, chunk, new_weight) for chunk in vertex_chunks]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    
    # Update the weights in the main process
    for source_index, target_index, weight in results:
        edge = graph.edge(source_index, target_index)
        weights[edge] = weight

    print("Time taken to update weights: ", time.time() - stime)

    return weights

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

import time
def create_masked_directed_energy_graph_from_mask(mask_data, direction='left', large_weight=1e8):
    z, y, x = mask_data.shape  # Dimensions of the 3D mask array
    print(z, y, x)
    g = gt.Graph(directed=True)
    weight_prop = g.new_edge_property("int")  # Edge property for weights

    # Create vertex properties for i, j, k positions
    x_prop = g.new_vertex_property("int")
    y_prop = g.new_vertex_property("int")
    z_prop = g.new_vertex_property("int")

    # Function to get linear index from 3D coordinates, assuming C-style row-major order
    index = lambda i, j, k: i * y * x + j * x + k

    # Create a mapping from mask coordinates to vertex indices
    coord_to_vertex = {}

    # Add vertices only for the non-zero elements in the mask
    stime = time.time()
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
    print("Time taken to add vertices to coord_to_vertex:", time.time()-stime)

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

    stime = time.time()

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

    print("Time taken to add edges to graph:", time.time()-stime)
    stime = time.time()
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
    print("Time taken to add source and sink nodes:", time.time()-stime)
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
    print("edges that cross the partition:", len(cross_partition))
    
    # Get the boundary vertices
    boundary_vertices = np.unique(np.concatenate((source_vertices[cross_partition], target_vertices[cross_partition])))
    
    # Filter to keep only the vertices that cross from the source set to the sink set
    # boundary_vertices = set(boundary_vertices[part[boundary_vertices]])
    # print(len(boundary_vertices))

    boundary_vertices = boundary_vertices[part[boundary_vertices] == 0]
    print("boundary vertices:", len(boundary_vertices))
    
    return set(boundary_vertices)

def boundary_vertices_to_array(boundary_vertices, shape, face):
    """
    Converts a set of boundary vertices back to a 3D array representation.
    
    Parameters:
        boundary_vertices (set): Set of boundary vertex indices.
        shape (tuple): The shape of the 3D array, e.g., (128, 128, 128).
        face (str): The face of the cube to consider ('x', 'y', or 'z').
    
    Returns:
        numpy.ndarray: A 3D array with marked boundary vertices.
    """
    z_dim, y_dim, x_dim = shape
    boundary_array = np.zeros(shape, dtype=np.int8)
    
    # Compute the 3D coordinates from the linear index
    for vertex in boundary_vertices:
        vertex_index = int(vertex)
        z = vertex_index // (y_dim * x_dim)
        y = (vertex_index % (y_dim * x_dim)) // x_dim
        x = (vertex_index % (y_dim * x_dim)) % x_dim
        # print(f"Computed indices - z: {z}, y: {y}, x: {x}")

        # Check if indices are within the valid range
        if 0 <= z < z_dim and 0 <= y < y_dim and 0 <= x < x_dim:
            boundary_array[z, y, x] = 1  # Mark the boundary vertex in the array
        else:
            print(f"Index out of bounds: z={z}, y={y}, x={x}")

    # Keep only the top-most value closest to the face in each column perpendicular to the face
    if face == 'x':
        for y in range(y_dim):
            for z in range(z_dim):
                found = False
                for x in range(x_dim):
                    if boundary_array[z, y, x] == 1:
                        boundary_array[z, y, :x] = 0  # Set all values below to 0
                        found = True
                        # break
                if not found:
                    boundary_array[z, y, :] = 0  # No values found in this column

    elif face == 'y':
        for x in range(x_dim):
            for z in range(z_dim):
                found = False
                for y in range(y_dim):
                    if boundary_array[z, y, x] == 1:
                        boundary_array[z, :y, x] = 0  # Set all values below to 0
                        found = True
                        # break
                if not found:
                    boundary_array[z, :, x] = 0  # No values found in this column

    elif face == 'z':
        for x in range(x_dim):
            for y in range(y_dim):
                found = False
                for z in range(z_dim):
                    if boundary_array[z, y, x] == 1:
                        boundary_array[:z, y, x] = 0  # Set all values below to 0
                        found = True
                        # break
                if not found:
                    boundary_array[:, y, x] = 0  # No values found in this column

    return boundary_array

def seam_voxels_to_graph_indices(boundary):
    """
    Transforms a 3D array of seam voxels into a list of graph indices.
    
    Parameters:
    boundary (np.ndarray): A 3D array where non-zero values specify the location of the selected seam voxels.
    
    Returns:
    list: A list of vertex indices.
    """
    # Get the dimensions of the 3D array
    z, y, x = boundary.shape
    
    # Function to convert 3D coordinates to 1D graph index
    index = lambda i, j, k: i * y * x + j * x + k
    
    # Find the non-zero voxel coordinates
    seam_voxel_coords = np.argwhere(boundary > 0)
    
    # Transform the coordinates to graph indices
    vertex_indices = [index(i, j, k) for i, j, k in seam_voxel_coords]
    
    return vertex_indices

# def visualize_min_cut(graph, boundary_vertices, cut_edges, src, tgt, output_filename):
#     # Position nodes using a layout algorithm
#     pos = gt.sfdp_layout(graph)

#     # Vertex color and size properties
#     vertex_color = graph.new_vertex_property("vector<double>")
#     vertex_size = graph.new_vertex_property("double")

#     # Edge color property
#     edge_color = graph.new_edge_property("vector<double>")
    
#     # Set properties for vertices and edges
#     for v in graph.vertices():
#         if v in boundary_vertices:
#             vertex_color[v] = [1, 0, 0, 1]  # Red for boundary vertices
#             vertex_size[v] = 10
#         else:
#             vertex_color[v] = [0.6, 0.6, 0.6, 1]  # Grey for others
#             vertex_size[v] = 5

#         if v == src or v == tgt:
#             vertex_color[v] = [0, 1, 0, 1]  # Green for source and sink
#             vertex_size[v] = 15

#     for e in graph.edges():
#         if e in cut_edges:
#             edge_color[e] = [1, 0, 0, 1]  # Red for cut edges
#         else:
#             edge_color[e] = [0.6, 0.6, 0.6, 1]  # Grey for other edges

#     # Draw the graph
#     gt.graph_draw(graph, pos=pos, vertex_fill_color=vertex_color,
#                   vertex_size=vertex_size, edge_color=edge_color,
#                   output_size=(1000, 1000), output=output_filename, vertex_text=tdg.vertex_index, edge_text=weights, vertex_font_size=18)

def seam_voxels_to_graph_indices(boundary):
    """
    Transforms a 3D array of seam voxels into a list of graph indices.
    
    Parameters:
    boundary (np.ndarray): A 3D array where non-zero values specify the location of the selected seam voxels.
    
    Returns:
    list: A list of vertex indices.
    """
    # Get the dimensions of the 3D array
    z, y, x = boundary.shape
    
    # Function to convert 3D coordinates to 1D graph index
    index = lambda i, j, k: i * y * x + j * x + k
    
    # Find the non-zero voxel coordinates
    seam_voxel_coords = np.argwhere(boundary > 0)
    
    # Transform the coordinates to graph indices
    vertex_indices = [index(i, j, k) for i, j, k in seam_voxel_coords]
    
    return vertex_indices

def update_outgoing_edges(graph, vertices, new_weight):
    """
    Updates the weights of all outgoing edges of the given vertices in a directed graph and returns the updated weights.
    
    Parameters:
    graph (Graph): A directed graph from graph_tool.
    vertices (list): A list of vertices whose outgoing edge weights need to be updated.
    new_weight (float): The new weight value to be set for the outgoing edges.
    
    Returns:
    PropertyMap: The updated edge weight property map.
    """

    # Ensure the graph has an edge weight property map
    if 'weight' in graph.edge_properties:
        weights = graph.edge_properties['weight']
    else:
        weights = graph.new_edge_property("int")
        graph.edge_properties['weight'] = weights

    # Convert the list of vertex indices to vertex objects
    vertex_objects = [graph.vertex(v) for v in vertices]

    stime = time.time()
    # Update the weight of all outgoing edges for each vertex
    for vertex in vertex_objects:
        for edge in vertex.in_edges():
            weights[edge] = new_weight
    print("Time taken to update weights: ", time.time()-stime)
    # Attach the updated weights to the graph
    graph.edge_properties['weight'] = weights

    return weights

def add_directed_source_sink(graph, z, y, x, source_face='front', sink_face='back', large_weight=1e8):
    # Add source and sink nodes to the graph
    source = graph.add_vertex()
    sink = graph.add_vertex()

    # Edge weight property
    if 'weight' in graph.edge_properties:
        weights = graph.edge_properties['weight']
    else:
        weights = graph.new_edge_property("int")

    # Helper function to get vertex indices for a face
    def get_face_vertices(face):
        indices = []
        if face == 'left':
            indices = [(0, j, k) for j in range(y) for k in range(x)]
        elif face == 'right':
            indices = [(z-1, j, k) for j in range(y) for k in range(x)]
        elif face == 'top':
            indices = [(i, 0, k) for i in range(z) for k in range(x)]
        elif face == 'bottom':
            indices = [(i, y-1, k) for i in range(z) for k in range(x)]
        elif face == 'front':
            indices = [(i, j, 0) for i in range(z) for j in range(y)]
        elif face == 'back':
            indices = [(i, j, x-1) for i in range(z) for j in range(y)]
        return [graph.vertex(i * y * x + j * x + k) for i, j, k in indices]

    # Get vertices for the specified faces
    source_vertices = get_face_vertices(source_face)
    sink_vertices = get_face_vertices(sink_face)

    # Connect the source to the corresponding face vertices
    for v in source_vertices:
        e = graph.add_edge(source, v)
        weights[e] = large_weight

    # Connect the corresponding face vertices to the sink
    for v in sink_vertices:
        e = graph.add_edge(v, sink)
        weights[e] = large_weight

    # Attach the weights to the graph
    graph.edge_properties['weight'] = weights

    return graph, int(source), int(sink), weights

def add_edge_with_weight(g, weight_prop, vertices, src_idx, tgt_idx, weight):
    e = g.add_edge(vertices[src_idx], vertices[tgt_idx])
    weight_prop[e] = weight

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

# works only in 1 direction and is the slowest part of the code
# but it works and has proven difficult to optimise or parallelise
def create_directed_energy_graph_from_mask(mask_data, direction='left', large_weight=1e8):
    z,y,x = mask_data.shape  # Dimensions of the 3D mask array
    g = gt.Graph(directed=True)
    weight_prop = g.new_edge_property("int")  # Edge property for weights

    # Function to get linear index from 3D coordinates, assuming C-style row-major order
    index = lambda i, j, k: i * y * x + j * x + k

    # Add vertices
    num_vertices = z * y * x
    g.add_vertex(num_vertices)
    vertices = list(g.vertices())  # Get the list of vertices

    # Define neighbor offsets based on directionality
    # Source to sink propagation direction - positive axes directions
    directions = {
        'left': [(0, 0, 1)],  # propagate right
        'right': [(0, 0, -1)],  # propagate left
        'top': [(0, 1, 0)],  # propagate downwards
        'bottom': [(0, -1, 0)],  # propagate upwards
        'front': [(1, 0, 0)],  # propagate back
        'back': [(-1, 0, 0)]  # propagate front
    }

    neighbors = directions[direction]
    print(x,y,z)
    
    for i in range(z):
        for j in range(y):
            for k in range(x):
                current_index = index(i, j, k)
                # print(current_index, len(vertices))
                current_vertex = vertices[current_index]

                # Check each neighbor direction for valid connections
                for di, dj, dk in neighbors:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < z and 0 <= nj < y and 0 <= nk < x:
                        neighbor_index = index(ni, nj, nk)
                        neighbor_vertex = vertices[neighbor_index]

                        # print(current_index, neighbor_index)

                        # Determine edge weight
                        weight = 10 if mask_data[ni, nj, nk] != 0 or mask_data[i, j, k] != 0 else 1

                        # Add edge and assign weight
                        e = g.add_edge(current_vertex, neighbor_vertex) #forward edge with energy value
                        e2 = g.add_edge(neighbor_vertex, current_vertex) #backward edge with large energy value
                        weight_prop[e] = weight
                        weight_prop[e2] = large_weight

                # Add each diagonal backwards neighbor inf edge, ie x-1, y-1 and x-1, y+1 for YX plane
                if k > 0 and j > 0:
                    neighbor_index = index(i, j-1, k-1)
                    neighbor_vertex = vertices[neighbor_index]
                    # print(current_index, neighbor_index)
                    # print(i,j,k, " y-1, x-1 ", i, j-1, k-1) 
                    e = g.add_edge(current_vertex, neighbor_vertex)
                    weight_prop[e] = large_weight+1
                if k > 0 and j < y-1:
                    neighbor_index = index(i, j+1, k-1)
                    neighbor_vertex = vertices[neighbor_index]
                    # print(current_index, neighbor_index)
                    # print(i,j,k, "  y+1 x-1 ", i, j+1, k-1)
                    e = g.add_edge(current_vertex, neighbor_vertex)
                    weight_prop[e] = large_weight+1

                # print(i,j,k, current_index)
                    
                # Add each diagonal backwards neighbor inf edge for ik plane
                if k > 0 and i > 0:
                    neighbor_index = index(i-1, j, k-1)
                    neighbor_vertex = vertices[neighbor_index]
                    # print(current_index, neighbor_index)
                    e = g.add_edge(current_vertex, neighbor_vertex)
                    weight_prop[e] = large_weight+2
                if k > 0 and i < z-1:
                    neighbor_index = index(i+1, j, k-1)
                    neighbor_vertex = vertices[neighbor_index]
                    # print(current_index, neighbor_index)
                    e = g.add_edge(current_vertex, neighbor_vertex)
                    weight_prop[e] = large_weight+2

                # Add each diagonal backwards neighbor inf edge for ij plane
                # if j > 0 and i > 0:
                #     neighbor_index = index(i-1, j-1, k)
                #     neighbor_vertex = vertices[neighbor_index]
                #     print(current_index, neighbor_index)
                #     e = g.add_edge(current_vertex, neighbor_vertex)
                #     weight_prop[e] = large_weight+3
                # if j > 0 and i < z-1:
                #     neighbor_index = index(i+1, j-1, k)
                #     neighbor_vertex = vertices[neighbor_index]
                #     print(current_index, neighbor_index)
                #     e = g.add_edge(current_vertex, neighbor_vertex)
                #     weight_prop[e] = large_weight+3



    g.edge_properties["weight"] = weight_prop
    return g, weight_prop

def create_graph(mask_data):
    graph, _ = create_directed_energy_graph_from_mask(mask_data)
    graph, src, sink, weights = add_directed_source_sink(graph, mask_data.shape[0], mask_data.shape[1], mask_data.shape[2])
    src, sink = graph.vertex(src), graph.vertex(sink)
    return graph, src, sink, weights

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

def reduce_array(data, block_size=(2, 2, 2), method='mean'):
    # Ensure that the dimensions are divisible by the block size
    assert data.shape[0] % block_size[0] == 0
    assert data.shape[1] % block_size[1] == 0
    assert data.shape[2] % block_size[2] == 0

    # Reshape the array to create blocks
    z, y, x = data.shape
    new_shape = (z // block_size[0], block_size[0], 
                 y // block_size[1], block_size[1], 
                 x // block_size[2], block_size[2])
    reshaped_data = data.reshape(new_shape)

    # Apply the reduction method
    if method == 'max':
        reduced_data = reshaped_data.max(axis=(1, 3, 5))
    elif method == 'min':
        reduced_data = reshaped_data.min(axis=(1, 3, 5))
    elif method == 'mean':
        reduced_data = reshaped_data.mean(axis=(1, 3, 5)).astype(data.dtype)
    else:
        raise ValueError('Unknown method: {}'.format(method))

    return reduced_data

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