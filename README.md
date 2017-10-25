# BSSP
Code for my BSSP research project involving morphological cell profiling with deep learning.

## Abstract
This project involves testing two novel methods of cell profiling and comparing them to each other and a baseline method, with the objective of determining the most robust method. The baseline method uses CellProfiler’s traditional computer vision algorithms for nucleus segmentation and feature extraction, the second method uses a U-Net for nucleus segmentation, and the third method uses an Inception-ResNet for convolutional feature extraction. Images of human MCF7 cells from an experiment designed to predict the mechanism of action (MoA) for unknown treatments were profiled and analyzed using each method. The resulting similarity matrices and prediction accuracies from the three methods show that the deep learning methods have a beneficial effect on downstream analysis and that the convolutional features yielded almost 10% more prediction accuracy over the baseline, while the CNN segmentation yielded an improvement of approximately 4%. If combined, these two methods could further improve prediction accuracy. These results suggest that using deep learning and convolutional features instead of classical features in cell profiling is much more robust for MoA classification.

## Files
*condenser.ipynb - Condense feature matrices from each stain into one matrix per image<br>
*aggregator.ipynb - Aggregate cell level DMSO data to plate level data, computing mean and std for normalization<br>
*normalizer.ipynb - Normalize all cell level data using means and stds from DMSO<br>
*collapser.ipynb - Collapse cell level data to well level data<br>
*treatments.ipynb - Collapse well level data to treatment level data<br>
*predictor.ipynb - Evaluate MoA prediction using treatment level data<br>
labels2.ipynb - Produce labeled matrices from CNN segmentation<br>
nuclei_coords.ipynb - Extract nucleus coordinates from feature matrices<br>
visualize_differences.py - Generate difference maps for CP/CNN segmentations<br>
whitening.ipynb, whitening2.ipynb - Whiten data with 1 or 2 passes<br>
*commands.txt - Commands to run in parallel for CellProfiler pipeline<br>
*.cppipe - CellProfiler pipeline

## Prefixes
cp - baseline CellProfiler<br>
cnn - CNN segmentation<br>
irn - Inception-ResNet<br>
w - with whitening
