# BSSP
Code for my BSSP research project involving morphological cell profiling with deep learning.

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
