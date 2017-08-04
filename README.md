# BSSP
Code for my BSSP research project involving morphological cell profiling with deep learning.

## Files
*condenser.ipynb - Condense feature matrices from each stain into one matrix per image
*aggregator.ipynb - Aggregate cell level DMSO data to plate level data, computing mean and std for normalization
*normalizer.ipynb - Normalize all cell level data using means and stds from DMSO
*collapser.ipynb - Collapse cell level data to well level data
*treatments.ipynb - Collapse well level data to treatment level data
*predictor.ipynb - Evaluate MoA prediction using treatment level data
labels2.ipynb - Produce labeled matrices from CNN segmentation
nuclei_coords.ipynb - Extract nucleus coordinates from feature matrices
visualize_differences.py - Generate difference maps for CP/CNN segmentations
*commands.txt - Commands to run in parallel for CellProfiler pipeline
*.cppipe - CellProfiler pipeline

## Prefixes
cp - baseline CellProfiler
cnn - CNN segmentation
irn - Inception-ResNet
