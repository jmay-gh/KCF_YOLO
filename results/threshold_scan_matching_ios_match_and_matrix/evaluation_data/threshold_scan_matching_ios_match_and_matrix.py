import matplotlib.pyplot as plt
import numpy as np

# Sample thresholds
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

# Sample HOTA values for two models/variants
# Plot 1
hota_model1_plot1 = [0.42288, 0.42625, 0.43119, 0.43411, 0.43287]
hota_model2_plot1 = [0.42288, 0.42625, 0.43119, 0.43411, 0.43287]

# Plot 2
hota_model1_plot2 = [0.41612, 0.40791, 0.41469, 0.41632, 0.42354]
hota_model2_plot2 = [0.41612, 0.40791, 0.41469, 0.41632, 0.42354]

# Bar width and spacing
bar_width = 0.35
gap = 0.03
x = np.arange(len(thresholds))  # base x-tick locations

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plot 1
axes[0].bar(x - bar_width/2 - gap, hota_model1_plot1, bar_width, label='Singly Thresholded', color = 'grey')
axes[0].bar(x + bar_width/2 + gap, hota_model2_plot1, bar_width, label='Doubly Thresholded', color = 'black')
axes[0].set_title('Hungarian Algorithm Matching')
axes[0].set_xlabel('Threshold Value')
axes[0].set_ylabel('HOTA Score')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'{t:.1f}' for t in thresholds])
axes[0].legend()
axes[0].set_ylim(0, 0.5)

# Plot 2
axes[1].bar(x - bar_width/2 - gap, hota_model1_plot2, bar_width, label='Singly Thresholded', color = 'grey')
axes[1].bar(x + bar_width/2 + gap, hota_model2_plot2, bar_width, label='Doubly Thresholded', color = 'black')
axes[1].set_title('EMD Matching')
axes[1].set_xlabel('Threshold Value')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'{t:.1f}' for t in thresholds])
axes[1].legend()
axes[1].set_ylim(0, 0.5)

plt.tight_layout()
plt.show()
