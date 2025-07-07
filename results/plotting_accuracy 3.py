import matplotlib.pyplot as plt
import pandas as pd

# Data
methods = ['Nearest Neighbour', 'Hungarian Algorithm', 'Earth Movers Distance']
tp = [762, 767, 608]
fp = [20, 15, 174]
fn = [10, 4, 11]

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
            ax.text(i, cumulative + value / 2, str(value), ha='center', va='center', color='white', fontsize=9)
        cumulative += value

# Customize plot
ax.set_ylabel('Count')
ax.set_title('Comparison of TP, FP, FN Across Matching Algorithms')
ax.legend()
plt.xticks(rotation=15)
plt.tight_layout()

# Show
plt.show()
