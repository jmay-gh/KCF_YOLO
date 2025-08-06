import matplotlib.pyplot as plt
import numpy as np

# Categories (e.g., threshold values or evaluation bins)
categories = ['0.2', '0.3', '0.4', '0.5', '0.6']
num_categories = len(categories)
num_groups = 2  # e.g., 4 different methods or variants

# Dummy data: shape (num_groups, num_bars)
data = np.array([
    [0.37682, 0.46055, 0.45327, 0.43790, 0.40985],  # IoU Singly Thresholded
    [0.45049, 0.46094, 0.46083, 0.45945, 0.45676],  # IoS Singly Thresholded
])

group_labels = ['IoU Singly Thresholded', 'IoS Singly Thresholded']
colors = [ '#888888', '#222222']

# Plotting setup
bar_width = 0.2
intra_group_gap = 0.02
group_width = num_groups * bar_width + (num_groups - 1) * intra_group_gap

x_centers = np.arange(num_categories) * (group_width + 0.2)  # 0.2 is gap between groups

fig, ax = plt.subplots(figsize=(10, 5))

# Plot each method
for i in range(num_groups):
    offsets = i * (bar_width + intra_group_gap)
    ax.bar(x_centers + offsets - group_width / 2 + bar_width / 2, data[i], width=bar_width,
           label=group_labels[i], color=colors[i])

# Set x-ticks in the center of each group
ax.set_xticks(x_centers)
ax.set_xticklabels(categories)
ax.set_xlabel('Threshold Value')
ax.set_ylabel('HOTA Score')
ax.set_title('HOTA Scores for Matching Thresholds with Hungarian Algorithm')
ax.set_ylim(0, 0.6)
ax.legend()

plt.tight_layout()
# plt.show()

plt.savefig('Desktop/KCF_YOLO/results/figures/threshold_scan_matching/hungarian_2_matching_types.png', dpi=300, bbox_inches='tight')
