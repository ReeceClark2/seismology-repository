import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_covariance_matrices(csv_filepath):
    # --- 0. Load the Covariance Matrix ---
    # Read the CSV. 
    # Note: If your CSV has column names or an index column, you might need to adjust 
    # this to pd.read_csv(csv_filepath, index_col=0) instead.
    df = pd.read_csv(csv_filepath, header=None)
    cov_matrix = df.to_numpy()

    # --- 1. Prepare Block-Normalized Log Matrix ---
    n_rows = cov_matrix.shape[0]
    mid = n_rows // 2

    def log_normalize_block(block):
        # Prevent log of negative numbers or zero if this was intended to use np.log
        # (Keeping your original logic as written)
        shifted_block = block - np.min(block) + 1

        min_log = np.min(block)
        max_log = np.max(block)
        if max_log == min_log:
            return np.zeros_like(block)
        return (block - min_log) / (max_log - min_log)

    normalized_log_mat = np.block([
        [log_normalize_block(cov_matrix[:mid, :mid]), log_normalize_block(cov_matrix[:mid, mid:])],
        [log_normalize_block(cov_matrix[mid:, :mid]), log_normalize_block(cov_matrix[mid:, mid:])]
    ])

    # --- 2. Calculate Correlation Matrix ---
    # Extract the standard deviations (square root of the diagonal variances)
    std_dev = np.sqrt(np.diag(cov_matrix))

    # Avoid division by zero warning by temporarily replacing 0s with 1s
    std_dev_safe = np.where(std_dev == 0, 1e-10, std_dev)

    # Compute correlation matrix 
    correlation_matrix = cov_matrix / np.outer(std_dev_safe, std_dev_safe)

    # Force any perfect 0/0 NaNs that slipped through to become 0
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

    # --- 3. Plot All Three Matrices Side-by-Side ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Plot A: Raw Covariance Matrix
    im0 = axes[0].imshow(cov_matrix, cmap='viridis', interpolation='nearest')
    axes[0].set_title('Raw Covariance Matrix')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot B: Block-Normalized Log Matrix
    im1 = axes[1].imshow(normalized_log_mat, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Block-Normalized Covariance')
    axes[1].axhline(mid - 0.5, color='white', linewidth=2)
    axes[1].axvline(mid - 0.5, color='white', linewidth=2)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot C: Correlation Matrix
    im2 = axes[2].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
    axes[2].set_title('Correlation Matrix')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Add uniform labels to all subplots
    for ax in axes:
        ax.set_xlabel(f'Variable Index\n(0-{mid-1}: Freq, {mid}-{n_rows-1}: Decay)')
        ax.set_ylabel('Variable Index')

    # Adjust layout so nothing overlaps
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace this string with the actual path to your CSV file
    csv_file_path = "C:/Users/starb/Downloads/matrix_output.csv"
    
    print(f"Loading covariance matrix from {csv_file_path} and generating plots...")
    plot_covariance_matrices(csv_file_path)