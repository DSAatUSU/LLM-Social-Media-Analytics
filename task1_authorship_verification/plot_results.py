"""Figures for Task I (authorship verification) from results/task1/weighted_metrics_all_models.csv.

Produces (in ./figures):
  * class-specific precision/recall bar plot per model (final_class_specific_precision_recall.eps)
  * F1(class 0) vs F1(class 1) scatter with per-model centroids (classwise_f1_comparison_with_centroids.eps)

Run from this folder:  python plot_results.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "task1", "weighted_metrics_all_models.csv")
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

rename_mapping = {
    'bert': 'BERT', 'Bert': 'BERT',
    'deepseek': 'DeepSeek',
    'gemini': 'Gemini',
    'gpt-3.5-turbo': 'GPT-3.5-Turbo',
    'gpt-4': 'GPT-4',
    'Llama3.2': 'Llama',
    'RandomForest': 'RF',
    'Gzip': 'Compression-NCD',
    'tfidf_cosine': 'TF-IDF',
    'use_cosine': 'USE',
    'SIAMESE_SBERT': 'SIAMESE + SBERT',
    'Siamese_GloVe': 'SIAMESE + GloVe'
}

# ----------------------------------------------------------------------
# Figure 1: class-specific precision / recall
# ----------------------------------------------------------------------
df = pd.read_csv(RESULTS_CSV)

df['model_name'] = df['model_name'].replace(rename_mapping)

# Set seaborn style for professional clarity
sns.set(style="whitegrid", context="talk", font_scale=1.3)

# Aggregate precision and recall for each class and model
class_metrics = df.groupby('model_name').agg({
    'precision_0': 'mean',
    'recall_0': 'mean',
    'precision_1': 'mean',
    'recall_1': 'mean'
}).reset_index()

# Save aggregated data to Excel
excel_file_path = os.path.join(output_dir, 'precision_recall_values.xlsx')
class_metrics.to_excel(excel_file_path, index=False, float_format='%.4f')

# Reshape data for plotting
class_metrics_melted = class_metrics.melt(
    id_vars='model_name',
    value_vars=['precision_0', 'recall_0', 'precision_1', 'recall_1'],
    var_name='Metric_Class',
    value_name='Score'
)

# Extract metric type and class, renaming classes for clarity
class_metrics_melted['Metric'] = class_metrics_melted['Metric_Class'].apply(lambda x: x.split('_')[0].capitalize())
class_metrics_melted['Class'] = class_metrics_melted['Metric_Class'].apply(lambda x: 'Negative Posts' if x.split('_')[1] == '0' else 'Positive Posts')

# Define color palette
palette = {"Precision": "#1f77b4", "Recall": "#ff7f0e"}

# Create bar plot
plot = sns.catplot(
    data=class_metrics_melted, x='model_name', y='Score', hue='Metric', col='Class',
    kind='bar', palette=palette, height=6.5, aspect=1.4, ci=None, legend=False
)

# Update subplot titles and apply hatches to recall bars
for ax in plot.axes.flat:
    class_title = ax.get_title().split('=')[1].strip()
    ax.set_title(class_title, fontsize=24, fontweight='bold')

    bars = ax.patches
    class_data = class_metrics_melted[class_metrics_melted['Class'] == class_title]
    for bar, metric in zip(bars, class_data['Metric']):
        if metric == 'Recall':
            bar.set_hatch('///')

    # Larger, bold tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=22, rotation=90, fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(axis='y', labelsize=24)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

# Bold and clearly formatted main title and axis labels
plot.fig.subplots_adjust(top=0.85, right=0.83)
#plot.fig.suptitle('Precision and Recall by Model and Post Type', fontsize=34, fontweight='bold')
plot.set_axis_labels('Model', 'Score', fontsize=28, fontweight='bold')

# Explicitly define and add legend
handles = [
    plt.Rectangle((0,0),1,1, facecolor=palette["Precision"]),
    plt.Rectangle((0,0),1,1, facecolor=palette["Recall"], hatch='///')
]
labels = ['Precision', 'Recall']
legend = plot.fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.89, 0.5), fontsize=20, frameon=False)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Save figure professionally
figure_path = os.path.join(output_dir, 'final_class_specific_precision_recall.eps')
plot.savefig(figure_path, bbox_inches='tight', format='eps')
plt.close()

# Print confirmation paths
print(f"Figure saved at: {figure_path}")
print(f"Numeric data saved at: {excel_file_path}")

# ----------------------------------------------------------------------
# Figure 2: F1 class 0 vs class 1 scatter
# ----------------------------------------------------------------------
df = pd.read_csv(RESULTS_CSV)

# Professional seaborn style
sns.set(style="whitegrid", context="talk", font_scale=1.2)

df['model_name'] = df['model_name'].replace(rename_mapping)

# Unique models and palette
unique_models = df['model_name'].unique()
palette = sns.color_palette("husl", len(unique_models))

# Marker styles
markers = ['o', 's', 'X', 'P', 'D', '^', '<', '>', 'v', '*', 'H', '$\\bowtie$', 'x', 'd', '|', '_']

color_marker_map = {
    model: {'color': palette[i], 'marker': markers[i % len(markers)]}
    for i, model in enumerate(unique_models)
}

# Plot setup
plt.figure(figsize=(60, 35))

# Individual scatter points
for model in unique_models:
    model_data = df[df['model_name'] == model]
    plt.scatter(model_data['f1_score_0'], model_data['f1_score_1'],
                label=model, alpha=0.3, s=1200,
                color=color_marker_map[model]['color'],
                marker=color_marker_map[model]['marker'])

# Centroids
centroids = df.groupby('model_name')[['f1_score_0', 'f1_score_1']].mean().reset_index()

# Print centroid coordinates
print("Centroid coordinates:")
for _, row in centroids.iterrows():
    print(f"{row['model_name']}: (Neg: {row['f1_score_0']:.3f}, Pos: {row['f1_score_1']:.3f})")

# Plot centroids
for _, row in centroids.iterrows():
    model = row['model_name']
    plt.scatter(row['f1_score_0'], row['f1_score_1'],
                s=7000, alpha=0.9, color=color_marker_map[model]['color'],
                marker=color_marker_map[model]['marker'],
                edgecolor='black', linewidth=1.5, label='_nolegend_')

# Diagonal reference
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

# Titles and axis labels
plt.xlabel('Weighted F1-score (Neg. Posts)', fontsize=110, fontweight='bold', labelpad=10)
plt.ylabel('Weighted F1-score (Pos. Posts)', fontsize=110, fontweight='bold', labelpad=10)
plt.title('Class-wise Weighted F1-score Comparison', fontsize=110, fontweight='bold', pad=15)

# Enhanced legend
legend = plt.legend(
    bbox_to_anchor=(1.05, 1.0),   # Adjusted for better positioning outside the figure
    loc='upper left', 
    fontsize=80,                  # Clearly larger legend text
    markerscale=1.5,              # Clearly larger legend markers
    frameon=False
)

# Bold legend text
for text in legend.get_texts():
    text.set_fontweight('bold')

# Bold and enlarge tick labels
plt.xticks(fontsize=100, fontweight='bold')
plt.yticks(fontsize=100, fontweight='bold')

plt.tight_layout()

# Save EPS figure
figure_path = os.path.join(output_dir, 'classwise_f1_comparison_with_centroids.eps')
plt.savefig(figure_path, bbox_inches='tight', format='eps')
plt.close()

print(f"Figure successfully saved at: {figure_path}")
