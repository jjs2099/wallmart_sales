import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_heatmap(data_path, output_dir):
    merged_data = pd.read_csv(data_path)
    plt.figure(figsize=(14, 10))
    heatmap = sns.heatmap(merged_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap with Engineered Features')
    
    heatmap_path = os.path.join(output_dir, 'heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    
    return heatmap_path
