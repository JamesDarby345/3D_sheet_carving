import numpy as np
import nrrd
import os
import trimesh
import pyvista as pv
from scipy.spatial import cKDTree

def array_to_thin_sheet_obj(array, filename, max_distance=1.8, smoothing_iterations=0):
    """
    Convert a 3D numpy array to a thin sheet-like mesh file, connecting only nearby voxels.
    
    Parameters:
    array (numpy.ndarray): 3D numpy array representing the structure.
    filename (str): Name of the output file.
    max_distance (float): Maximum distance for connecting voxels.
    smoothing_iterations (int): Number of smoothing iterations to apply.
    
    Returns:
    pyvista.PolyData: The resulting mesh.
    """
    
    # Find the indices of non-zero elements
    indices = np.argwhere(array != 0)
    
    # Create vertices from these indices
    vertices = indices.astype(float)
    print(f"Found {len(vertices)} non-zero elements.")
    
    # Create a KD-tree for efficient nearest neighbor search
    tree = cKDTree(vertices)
    
    # Find pairs of points within max_distance
    pairs = tree.query_pairs(r=max_distance)
    
    # Create edges from these pairs
    edges = np.array(list(pairs))
    
    # Create a PyVista PolyData object
    mesh = pv.PolyData(vertices, lines=edges)
    
    # Convert lines to surface
    surf = mesh.delaunay_2d()
    
    print(f"Created surface with {surf.n_points} points and {surf.n_cells} cells.")
    
    # Apply smoothing if requested
    if smoothing_iterations > 0:
        surf = surf.smooth(n_iter=smoothing_iterations)
    
    # Save as OBJ
    # surf.save(filename)
    # print(f"OBJ file '{filename}' has been created.")
    
    return surf

def visualize_mesh(mesh):
    """
    Visualize the mesh using PyVista.

    Parameters:
    mesh (pyvista.PolyData): The mesh to visualize.
    """
    print("Visualizing mesh...")
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='red', show_edges=True, opacity=0.7)
    plotter.add_points(mesh.points, color='blue', point_size=5)
    plotter.show_axes()
    plotter.show()

current_directory = os.getcwd()
input_nrrd_path = f'{current_directory}/output/09936_03280_04560_zyx_256_chunk_s1_vol_label_thinned.nrrd'
output_obj_path = f'{current_directory}/output/09936_03280_04560_thin_sheet.obj'

array, _ = nrrd.read(input_nrrd_path)
array[array!=6] = 0
mesh = array_to_thin_sheet_obj(array, output_obj_path, max_distance=1.8, smoothing_iterations=1)
visualize_mesh(mesh)