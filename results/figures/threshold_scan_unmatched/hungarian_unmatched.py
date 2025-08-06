import matplotlib.pyplot as plt
import numpy as np

# Categories (e.g., threshold values or evaluation bins)
categories = ['0', '3', '6', '9', '12']
num_categories = len(categories)
num_groups = 1  # Only one group in your case

# Data
data = np.array([
    [0.31815, 0.39933, 0.40825, 0.41026, 0.40742],
])

# Plotting setup
bar_width = 0.2
intra_group_gap = 0.02
group_width = num_groups * bar_width + (num_groups - 1) * intra_group_gap

x_centers = np.arange(num_categories) * (group_width + 0.2)

fig, ax = plt.subplots(figsize=(8, 4))

# Get index of max value
best_idx = np.argmax(data[0])
highlight_color = '#2E8B57'

# Plot bars
for i in range(num_groups):
    offsets = i * (bar_width + intra_group_gap)
    x_positions = x_centers + offsets - group_width / 2 + bar_width / 2

    # Set colors: default grey, best gets green
    colors = ['#222222'] * num_categories
    colors[best_idx] = highlight_color

    bars = ax.bar(x_positions, data[i], width=bar_width, color=colors)

    # Add values above each bar
    for x, height in zip(x_positions, data[i]):
        ax.text(x, height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=10)

# Set x-ticks in the center of each group
ax.set_xticks(x_centers)
ax.set_xticklabels(categories)
ax.set_xlabel('Strike Threshold Value')
ax.set_ylabel('HOTA Score')
ax.set_title('HOTA Scores for Unmatched Thresholds with Hungarian Algorithm')
ax.set_ylim(0, 0.6)

plt.tight_layout()
# plt.show()

plt.savefig('Desktop/KCF_YOLO/results/figures/threshold_scan_unmatched/hungarian_unmatched.png', dpi=300, bbox_inches='tight')


