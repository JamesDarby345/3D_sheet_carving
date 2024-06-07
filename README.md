# 3D_sheet_carving
A implementation of seam carving generalized to 3D volumes, thus sheet carving.
Originally inspired by this paper: https://www.researchgate.net/publication/215721610_Seam_Carving_for_Content-Aware_Image_Resizing, the 3d implementation is based off of the forward energy specified in this paper: https://people.csail.mit.edu/mrub/papers/vidret.pdf though does not follow it exactly.

It also uses multi-banded graph cuts as specified in this paper: https://www.researchgate.net/publication/4193861_A_multilevel_banded_graph_cuts_method_for_fast_image_segmentation

The goal of this repository is to provide an effcient, open source implementation of 3D sheet carving in python. Creating this repository was further motivated by using 3D sheet carving as a data augmentation to take instance labelled 3D cubes from the Vesuvius Scroll challenge and 'densify' them. Thus helping with domain transfer from the labels which include very few examples of touching sheets to the common situation in the dataset where many sheets are densly packed and touch eachother. Sheet carving allows for this densification to take place while maintaining visual coherence, and not introducing any odd or out of place artifacts.

Currently it takes ~20 seconds to calculate a 3d sheet/seam to remove on a 256^3 volume on a M2 Max Macbook Pro, larger volumes will scale superlinearily, so larger volumes may not be practicle without further optimization. To acheive a densified label, ~80 sheets are calculated and removed from a volume, thus taking a signifigant amount of time, though far better than the original ~15 minutes per sheet before optimization efforts and multi banded graph cuts were implemented. The new_dev_graphTool_multires_densification notebook in old_dev_notebooks has print statements for the timing of different parts of the process.

The graphTool_multires_densification.ipynb jupyter notebook calculates and removes seams from one volume with some visulisations before and after to verify the correctness.

The parallel_multires_densificaiton.py file takes in a path to a folder, assuming it is in the working directory, and assumes the folder has a label and raw subfolder containing paired xyz_label.nrrd and xyz_raw.nrrd files, which it will then rotate on the 3 different axis and calculate seam removals on all the different data cubes and rotations in parrallel. It will also save the densified cubes every 20 iterations until it hits the maximum number of iterations, 120 by default.

The resulting densified .nrrd cubes are of the same dimensions of the original, but the right hand side of the cube is filled with -1 values, while the left hand side has the densified data. They are saved to a output/densified_cubes folder.
