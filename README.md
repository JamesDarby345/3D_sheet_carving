# 3D_sheet_carving
A implementation of seam carving generalized to 3D volumes, thus referred to as sheet carving.
Originally inspired by this paper: https://www.researchgate.net/publication/215721610_Seam_Carving_for_Content-Aware_Image_Resizing, the 3d implementation is based off of the forward energy specified in this paper: https://people.csail.mit.edu/mrub/papers/vidret.pdf though does not follow it exactly.

It also uses multi-banded graph cuts as specified in this paper: https://www.researchgate.net/publication/4193861_A_multilevel_banded_graph_cuts_method_for_fast_image_segmentation

## Purpose of this repository
The goal of this repository is to provide an efficient, open source implementation of 3D sheet carving in python. Creating this repository was further motivated by using 3D sheet carving as a data augmentation to take instance labelled 3D cubes from the Vesuvius Scroll challenge and 'densify' them. Thus helping with domain transfer from the labels which include very few examples of touching sheets to the common situation in the dataset where many sheets are densly packed and touch eachother. Sheet carving allows for this densification to take place while trying to maintain visual coherence, thus not introducing to many odd or out of place artifacts.

## Example Results with varying iterations of seam removals
![Screenshot 2024-06-25 at 6 35 35 PM](https://github.com/JamesDarby345/3D_sheet_carving/assets/49734270/6ac44301-0daa-48d8-866e-feb8b79a16e2)
![Screenshot 2024-06-25 at 6 35 44 PM](https://github.com/JamesDarby345/3D_sheet_carving/assets/49734270/fd4fd010-e2d4-4fee-adeb-feeb11011f48)
![Screenshot 2024-06-25 at 6 35 53 PM](https://github.com/JamesDarby345/3D_sheet_carving/assets/49734270/1717bb92-0688-4dd8-8dca-62356c3a9d17)

## Technical details and file descriptions
Currently it takes ~20 seconds to calculate a 3d sheet/seam to remove on a 256^3 volume on a M2 Max Macbook Pro, larger volumes will scale superlinearily, so larger volumes may not be practicle without further optimization. To acheive a densified label, ~80 sheets are calculated and removed from a volume, thus taking a signifigant amount of time, though far better than the original ~15 minutes per sheet before optimization efforts and multi banded graph cuts were implemented. The new_dev_graphTool_multires_densification notebook in old_dev_notebooks has print statements for the timing of different parts of the process.

The graphTool_multires_densification.ipynb jupyter notebook calculates and removes seams from one volume with some visulisations before and after to verify the correctness.

The parallel_multires_densificaiton.py file takes in a path to a folder, assuming it is in the working directory, and assumes the folder has a label and raw subfolder containing paired xyz_label.nrrd and xyz_raw.nrrd files, which it will then rotate on the 3 different axis and calculate seam removals on all the different data cubes and rotations in parrallel. It will also save the densified cubes every 20 iterations until it hits the maximum number of iterations, 120 by default.

The resulting densified .nrrd cubes are of the same dimensions of the original, but the right hand side of the cube is filled with -1 values, while the left hand side has the densified data. They are saved to a output/densified_cubes folder.
