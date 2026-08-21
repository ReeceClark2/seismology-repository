import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 1. Load your data
# Replace 'your_data.csv' with your actual filename
df = pd.read_csv('BATS_model_results_16.csv')

# 1. Group by function, then find the row index of the max probability for each group
best_indices = df.groupby('function')['probability'].idxmax()

# 2. Extract those exact rows from the original dataframe
best_rows = df.loc[best_indices]

# 3. Print the results clearly
print("Highest Probabilities per Function:")
print("-" * 40)
for index, row in best_rows.iterrows():
    func_id = int(row['function'])
    freq = row['frequency']
    decay = row['decay_rate']
    prob = row['probability']
    
    print(freq, ',')
for index, row in best_rows.iterrows():
    func_id = int(row['function'])
    freq = row['frequency']
    decay = row['decay_rate']
    prob = row['probability']
    
    print(decay, ',')

# 2. Get a list of all unique function IDs
function_ids = df['function'].unique()

for func_id in function_ids:
    subset = df[df['function'] == func_id]
    
    # Get the specific max point for THIS function from our best_rows dataframe
    max_row = best_rows[best_rows['function'] == func_id].iloc[0]
    max_x = max_row['frequency']
    max_y = max_row['decay_rate']
    max_p = max_row['probability']

    x = subset['frequency'].values
    y = subset['decay_rate'].values
    z = subset['probability'].values
    
    grid_x, grid_y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    
    plt.figure(figsize=(8, 6))
    heatmap = plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap='Blues')
    
    # 1. Overlay the sampling points
    plt.scatter(x, y, c='red', s=0.4, alpha=0.2, label='Samples')
    
    # 2. Add the "Best" point (Max Probability)
    plt.scatter(x[0], y[0], color='white', marker='o', s=100, 
                edgecolors='black', linewidths=1, label='Initial Probability', zorder=5)
    plt.scatter(max_x, max_y, color='white', marker='*', s=150, 
                edgecolors='black', linewidths=1, label='Highest Probability', zorder=5)

    # Formatting
    plt.colorbar(heatmap, label='Probability')
    plt.title(f'Interpolated Probability Distribution for Function {func_id}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Decay Rate ($\gamma$)')
    plt.legend(loc='upper right')
    
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0), useOffset=False)
    plt.tight_layout()
    plt.savefig(f"{func_id}_heatmap.png", dpi=300)
    plt.close() # Recommended to close plots in a loop to save memory