# 3D_sheet_carving
A implementation of seam carving generalized to 3D volumes.
The implementation is based of off this paper though does not follow it exactly: https://people.csail.mit.edu/mrub/papers/vidret.pdf

The goal of this repository is to provide an effcient, open source implementation of 3D sheet carving in python. Creating this repository was further motivated by using 3D sheet carving as a data augmentation to take instance labelled 3D cubes from the Vesuvius Scroll challenge and 'densify' them. Thus helping with domain transfer from the labels which include very few examples of touching sheets to teh common situation in the dataset where many sheets are densly packed and touch eachother. 
