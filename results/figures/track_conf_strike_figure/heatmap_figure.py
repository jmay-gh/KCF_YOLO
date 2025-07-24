import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

# Example values between 30 and 40
data = np.array([
[0.34756, 0.35474, 0.35772, 0.36031, 0.36119],
[0.36443, 0.36642, 0.36718, 0.36690, 0.36661],
[0.36561, 0.36256, 0.36041, 0.35790, 0.35691]
])

fig, ax = plt.subplots(figsize=(4, 6))  # Square figure

# Get a lighter subset of the Blues colormap
cmap = cm.get_cmap('Blues', 256)
lighter_blues = cmap(np.linspace(0.4, 1.0, 256))  # Skip darkest ~30%

# Apply the new colormap
im = ax.imshow(data, cmap=cm.colors.ListedColormap(lighter_blues))
ax.set_aspect('equal')  # Ensure square grid cells

# Add labels into center of cells
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        text = f"{data[i, j]:.3f}"  # Format with 2 decimal places
        ax.text(j, i, text, ha='center', va='center', color='black', fontsize=10)

# Add colorbar using separate axis
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("HOTA", rotation=270, labelpad=15)
cbar.ax.tick_params(labelsize=8)

# Set ticks to center of each square
num_rows, num_cols = data.shape
ax.set_xticks(np.arange(num_cols))
ax.set_yticks(np.arange(num_rows))

# Optional: add tick labels
ax.set_xticklabels([f'{i:.2f}' for i in np.arange(5, 18, 3)])
ax.set_yticklabels([f'{i:.2f}' for i in np.arange(0.45, 0.24, -0.10)])

# Show gridlines between cells
ax.set_xticks(np.arange(num_cols+1)-0.5, minor=True)
ax.set_yticks(np.arange(num_rows+1)-0.5, minor=True)
ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
ax.tick_params(which='minor', bottom=False, left=False)

ax.set_xlabel("Tracker Confidence")
ax.set_ylabel("Detection Confidence")

fig.suptitle("Tracker and Detection HOTA for Removal Thresholds", fontsize=10, y=0.85)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
