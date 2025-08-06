import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

# Example values between 30 and 40
data = np.array([
[0.26507, 0.32389, 0.35275, 0.35815, 0.36691],
[0.33922, 0.36815, 0.38659, 0.38937, 0.38935],
[0.38537, 0.38537, 0.38060, 0.37618, 0.37552],
[0.37540, 0.36101, 0.34210, 0.34207, 0.34204],
[0.34384, 0.34212, 0.34210, 0.34207, 0.34204]
])

fig, ax = plt.subplots(figsize=(6, 5))  # Square figure

# Get a lighter subset of the Blues colormap
cmap = cm.get_cmap('Blues', 256)
lighter_blues = cmap(np.linspace(0.0, 1.0, 256))  # Skip darkest ~40%

# Apply the new colormap
im = ax.imshow(data, cmap='Blues', vmin=0.26, vmax=0.40)
ax.set_aspect('equal')  # Ensure square grid cells

# Add labels into center of cells
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        text = f"{data[i, j]:.4f}"  # Format with 2 decimal places
        ax.text(j, i, text, ha='center', va='center', color='black', fontsize=10)

# Add colorbar using separate axis
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("HOTA", rotation=270, labelpad=15)
cbar.ax.tick_params(labelsize=8)

# Set ticks at both ends
ticks = np.linspace(0.26, 0.40, num=5)  # 3 evenly spaced ticks
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{t:.3f}' for t in ticks])

# Set ticks to center of each square
num_rows, num_cols = data.shape
ax.set_xticks(np.arange(num_cols))
ax.set_yticks(np.arange(num_rows))

# Optional: add tick labels
ax.set_xticklabels([f'{i}' for i in np.arange(0, 13, 3)])
ax.set_yticklabels([f'{i:.1f}' for i in np.arange(0.50, 0.09, -0.10)])

# Show gridlines between cells
ax.set_xticks(np.arange(num_cols+1)-0.5, minor=True)
ax.set_yticks(np.arange(num_rows+1)-0.5, minor=True)
ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
ax.tick_params(which='minor', bottom=False, left=False)

ax.set_xlabel("Strike Threshold Value")
ax.set_ylabel("Confidence Threshold Value")

fig.suptitle("HOTA Values for Tracker Conf and Strike Removal Thresholds", fontsize=10, y=0.98)
plt.tight_layout()
# plt.show()

plt.savefig('Desktop/KCF_YOLO/results/figures/threshold_scan_tracker/threshold_scan_hungarian_tracker_conf.png', dpi=300, bbox_inches='tight')
