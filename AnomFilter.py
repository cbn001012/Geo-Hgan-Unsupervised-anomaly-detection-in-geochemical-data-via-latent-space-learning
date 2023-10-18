import os
import numpy as np
from scipy import stats
from skimage import io
import shutil

# Step 1: Select all the samples in the target domain to form the sample set X
data_folder = 'path/to/data/folder'
samples = os.listdir(data_folder)

# Step 2: Calculate the average feature vector M for X
n_channels = 39  # Replace with the actual number of data channels
sample_center = np.zeros(n_channels)

for sample_file in samples:
    sample_path = os.path.join(data_folder, sample_file)
    sample_data = io.imread(sample_path)
    sample_center += np.mean(sample_data, axis=(0, 1))

sample_center /= len(samples)

# Step 3: Calculate Q3 and IQR based on file names
source_outliers_folder = 'path/to/source/outliers/folder'
source_outliers = os.listdir(source_outliers_folder)

outlier_scores = [float(outlier_file.split(".")[0]) for outlier_file in source_outliers]

Q3 = np.percentile(outlier_scores, 75)
IQR = stats.iqr(outlier_scores)

# Step 4: Keep outlier samples greater than Q3 + 1.5 * IQR
X_c_ast = []
for outlier_file in source_outliers:
    outlier_score = float(outlier_file.split(".")[0])

    if outlier_score > Q3 + 1.5 * IQR:
        X_c_ast.append(outlier_file)

# Step 5: Calculate the distance set D using Euclidean distance
D = []
for outlier_file in X_c_ast:
    outlier_path = os.path.join(source_outliers_folder, outlier_file)
    outlier_data = io.imread(outlier_path)
    distance = np.linalg.norm(sample_center - outlier_data)
    D.append(distance)

# Step 6: Calculate the filtering radius R
R = (np.min(D) + np.max(D)) / 2

# Step 7: Keep only the outlier samples within the distance range of M within the filtering radius R
X_f_ast = []
for outlier_file, distance in zip(X_c_ast, D):
    if distance <= R:
        X_f_ast.append(outlier_file)

# Specify the folder to save the final outlier samples
output_folder = 'path/to/save/outliers'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Copy the outlier samples to the output folder
for outlier_file in X_f_ast:
    outlier_path = os.path.join(source_outliers_folder, outlier_file)
    output_path = os.path.join(output_folder, outlier_file)
    shutil.copy2(outlier_path, output_path)

print("Outlier samples saved to:", output_folder)