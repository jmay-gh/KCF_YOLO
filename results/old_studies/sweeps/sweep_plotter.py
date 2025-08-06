import matplotlib.pyplot as plt

# Common IoU thresholds (used across all methods)
ious = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

# Dictionary of methods -> metrics
results = {
    "Nearest Neighbour": [0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.96, 0.95, 0.92, 0.86, 0.77, 0.62, 0.46, 0.28, 0.12, 0.02, 0.00, 0.00],

    "Hungarian IOU": [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.98, 0.97, 0.96, 0.93, 0.87, 0.77, 0.62, 0.46, 0.28, 0.12, 0.02, 0.00, 0.00],

    # "Hungarian Distance": [],

    "EMD Distance": [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.98, 0.97, 0.96, 0.92, 0.87, 0.77, 0.62, 0.46, 0.28, 0.12, 0.02, 0.00, 0.00],

    "EMD Area": [0.99, 0.99, 1.00, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.97, 0.96, 0.93, 0.87, 0.77, 0.62, 0.46, 0.28, 0.12, 0.02, 0.00, 0.00],

    # "EMD HOG": [],
}


# Plot precision curves
plt.figure(figsize=(10, 6))
for method, precisions in results.items():
    plt.plot(ious, precisions, label=f'{method}')

plt.title('Precision vs IoU Threshold')
plt.xlabel('IoU Threshold')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.ylabel('Precision')
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("precision_vs_iou.png")
plt.show()

