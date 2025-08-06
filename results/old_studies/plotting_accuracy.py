import matplotlib.pyplot as plt
import pandas as pd

# Data
methods = ['Nearest Neighbour', 'Hungarian Algorithm', 'EMD - Distance', 'EMD - Area', 'EMD - HOG']

## WITH NOISE
# tp = [4233, 4246, 4175, 4252, 4245]
# fp = [718, 705, 776, 699, 706]
# fn = [61, 53, 86, 37, 57]

## WITHOUT NOISE
tp = [4897, 4897, 4894, 4897, 4897]
fp = [54, 54, 57, 54, 54]
fn = [43, 43, 45, 43, 43]

# Create DataFrame
df = pd.DataFrame({
    'Method': methods,
    'True Positives': tp,
    'False Positives': fp,
    'False Negatives': fn
})

# Plotting
bar_width = 0.6
fig, ax = plt.subplots(figsize=(10, 6))

# Stack bars
bottom = [0] * len(methods)
colors = ['#4CAF50', '#FF9800', '#F44336']  # TP = green, FP = orange, FN = red

for i, category in enumerate(['True Positives', 'False Positives', 'False Negatives']):
    ax.bar(df['Method'], df[category], bottom=bottom, label=category, color=colors[i])
    bottom = [sum(x) for x in zip(bottom, df[category])]

# Annotate values
for i, method in enumerate(methods):
    cumulative = 0
    for cat in ['True Positives', 'False Positives', 'False Negatives']:
        value = df[cat][i]
        if value > 0:
            ax.text(i, cumulative + value / 2, str(value), ha='center', va='center', color='black', fontsize=9)
        cumulative += value

# Customize plot
ax.set_ylabel('Count')
ax.set_title('Comparison of TP, FP, FN Across Matching Algorithms')
ax.legend()
plt.xticks(rotation=15)
plt.tight_layout()

# Show
plt.show()
