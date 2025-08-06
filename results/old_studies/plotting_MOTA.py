import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the data
data = {
    'Method': ['Nearest Neighbour', 'Hungarian Algorithm', 'EMD - Distance', 'EMD - Area', 'EMD - HOG'],

    # WITH NOISE
    # 'Precision': [0.854979, 0.857605, 0.843264, 0.858816, 0.857403],
    # 'Recall': [0.985794, 0.987672, 0.979817, 0.991373, 0.98675],
    # 'F1 Score': [0.915738, 0.918054, 0.906426, 0.920346, 0.91754]

    ## WITHOUT NOISE
    'Precision': [0.989093, 0.989093, 0.988487, 0.989093, 0.989093],
    'Recall': [0.991296, 0.991296, 0.990889, 0.991296, 0.991296],
    'F1 Score': [0.990193, 0.990193, 0.989687, 0.990193, 0.990193]

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
