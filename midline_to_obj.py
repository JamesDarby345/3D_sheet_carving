import numpy as np
import nrrd
import os
import trimesh
import pyvista as pv
from scipy.spatial import cKDTree
import multiprocessing
import time

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

def pyvista_to_trimesh(pv_mesh):
    """
    Convert a PyVista mesh to a Trimesh object, preserving UV coordinates.
    
    Parameters:
    pv_mesh (pyvista.PolyData): The input PyVista mesh.
    
    Returns:
    trimesh.Trimesh: The converted Trimesh object.
    """
    vertices = pv_mesh.points
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:4]
    
    # Create the Trimesh object
    tm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Add UV coordinates if they exist
    if 'UV' in pv_mesh.point_data:
        uv_coords = pv_mesh.point_data['UV']
        tm_mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords)
    
    return tm_mesh

def add_uv_mapping(mesh):
    """
    Add UV coordinates to the mesh using a simple planar projection.
    
    Parameters:
    mesh (pyvista.PolyData): The input mesh.
    
    Returns:
    pyvista.PolyData: The mesh with UV coordinates added.
    """
    # Get the mesh points
    points = mesh.points
    
    # Normalize X and Y coordinates to [0, 1] range for UV mapping
    min_xy = np.min(points[:, :2], axis=0)
    max_xy = np.max(points[:, :2], axis=0)
    uv_coords = (points[:, :2] - min_xy) / (max_xy - min_xy)
    
    # Add the UV coordinates to the mesh
    mesh.point_data['UV'] = uv_coords
    
    # print("Added UV mapping to the mesh.")
    return mesh

def filter_disconnected_parts(mesh, min_vertices):
    """
    Filter out disconnected parts of the mesh with fewer than min_vertices.
    
    Parameters:
    mesh (pyvista.PolyData): The input mesh.
    min_vertices (int): The minimum number of vertices a part should have to be kept.
    
    Returns:
    pyvista.PolyData: The filtered mesh.
    """
    # Get connected regions
    labeled = mesh.connectivity(largest=False)
    
    # Count vertices in each region
    unique_labels, counts = np.unique(labeled.cell_data['RegionId'], return_counts=True)
    
    # Create a mask for regions to keep
    keep_mask = np.isin(labeled.cell_data['RegionId'], unique_labels[counts >= min_vertices])
    
    # Extract the kept regions
    filtered_mesh = labeled.extract_cells(keep_mask)
    
    # print(f"Filtered out {len(unique_labels) - np.sum(counts >= min_vertices)} disconnected parts")
    # print(f"Remaining parts: {np.sum(counts >= min_vertices)}")
    
    return filtered_mesh

def array_to_thin_sheet_obj(array, filename, max_distance=1.8, min_vertices=1000):
    """
    Convert a 3D numpy array to a thin sheet-like mesh file, connecting only nearby voxels.
    
    Parameters:
    array (numpy.ndarray): 3D numpy array representing the structure.
    filename (str): Name of the output file.
    max_distance (float): Maximum distance for connecting voxels.
    min_vertices (int): Minimum number of vertices for a disconnected part to be kept.
    
    Returns:
    pyvista.PolyData: The resulting mesh.
    """
    
    # Find the indices of non-zero elements
    indices = np.argwhere(array != 0)
    
    # Create vertices from these indices
    vertices = indices.astype(float)
    # print(f"Found {len(vertices)} non-zero elements.")
    
    if len(vertices) <= 1:
        print("Not enough points to create a mesh, skipping.")
        return None

    # Create a KD-tree for efficient nearest neighbor search
    tree = cKDTree(vertices)
    
    # Find pairs of points within max_distance
    pairs = tree.query_pairs(r=max_distance)
    
    # Create edges from these pairs
    edges = np.array(list(pairs))
    
    # Create a PyVista PolyData object
    mesh = pv.PolyData(vertices, lines=edges)
    
    # Convert lines to surface
    surf = mesh.delaunay_2d(alpha=max_distance)
    
    # print(f"Created surface with {surf.n_points} points and {surf.n_cells} cells.")

    # Filter out disconnected parts
    surf = filter_disconnected_parts(surf, min_vertices=min_vertices)

    if surf.n_points == 0:
        print("No points left after filtering, skipping mesh creation.")
        return None
    
    # print(f"After filtering: surface has {surf.n_points} points and {surf.n_cells} cells.")

    surf = surf.delaunay_2d(alpha=max_distance*3)

    surf = add_uv_mapping(surf)
    
    # Convert to Trimesh
    tm_mesh = pyvista_to_trimesh(surf)
    
    # Save as OBJ
    with open(filename, 'w') as f:
        f.write("# OBJ file\n")
        for v in tm_mesh.vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        if tm_mesh.visual.uv is not None:
            for uv in tm_mesh.visual.uv:
                f.write(f"vt {uv[0]} {uv[1]}\n")
            
            for face in tm_mesh.faces:
                f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
        else:
            for face in tm_mesh.faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    # print(f"OBJ file '{filename}' has been created.")
    
    return surf

def process_single_value(args):
    value, original_array, output_obj_path, max_distance, min_vertices, visualise = args
    # print(f"Processing value {value}...")
    if np.sum(original_array == value) == 0:
        print(f"Value {value} not found in the input array, skipping.")
        return
    array = original_array.copy()
    array[original_array!=value] = 0
    temp_output_obj_path = f'{output_obj_path}_{value}.obj'
    mesh = array_to_thin_sheet_obj(array, temp_output_obj_path, max_distance=max_distance, min_vertices=min_vertices)
    if visualise:
        visualize_mesh(mesh)

def midline_labels_to_obj(input_nrrd_path, output_obj_path, array_values=None, max_distance=1.5, min_vertices=1000, visualise=False):
    original_array, _ = nrrd.read(input_nrrd_path)
    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
    if not array_values:
        array_values = np.unique(original_array)
    array_values = [v for v in array_values if v != 0]
    
    # Prepare arguments for multiprocessing
    args_list = [(value, original_array, output_obj_path, max_distance, min_vertices, visualise) for value in array_values]
    
    # Use multiprocessing to process label values in parallel
    with multiprocessing.Pool() as pool:
        pool.map(process_single_value, args_list)

# Main execution
if __name__ == "__main__":
    current_directory = os.getcwd()
    zyx = '09936_03280_04560'
    scroll_name = 's1'
    input_nrrd_path = f'{current_directory}/output/{zyx}_zyx_256_chunk_s1_vol_label_thinned.nrrd'
    output_obj_path = f'{current_directory}/output/objs/{scroll_name}/{zyx}/{zyx}'
    array_values = []  # If empty, it will process all unique values in the array
    stime = time.time()
    midline_labels_to_obj(input_nrrd_path, output_obj_path, array_values=array_values, max_distance=1.5, min_vertices=500, visualise=False)
    print(f"Time taken: {time.time() - stime:.2f} seconds.")