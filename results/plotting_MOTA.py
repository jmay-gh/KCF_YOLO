import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the data
data = {
    'Method': ['Nearest Neighbour', 'Hungarian Algorithm', 'Earth Movers Distance'],
    'Precision': [0.974425, 0.980818, 0.777494],
    'Recall': [0.987047, 0.994812, 0.982229],
    'F1 Score': [0.980695, 0.987766, 0.867951]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame to long format for Seaborn
df_melted = df.melt(id_vars='Method', var_name='Metric', value_name='Score')

# Set up the plot
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x='Method', y='Score', hue='Metric')

# Add value labels
for p in plt.gca().patches:
    height = p.get_height()
    plt.gca().annotate(f'{height:.3f}', 
                       (p.get_x() + p.get_width() / 2., height), 
                       ha='center', va='bottom', fontsize=9)

# Customize the plot
plt.title('Performance Comparison of Matching Algorithms')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.legend(title='Metric', loc = 'lower right')
plt.xticks(rotation=15)
plt.tight_layout()

# Show the plot
plt.show()
