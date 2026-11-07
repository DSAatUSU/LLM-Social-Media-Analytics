import matplotlib.pyplot as plt
import numpy as np

labels = ['DeepSeek', 'Gemini 1.5 Pro', 'GPT-4o', 'Llama 3.2']
categories = ['Definitely me', 'Probably me', 'Unsure', 'Probably not', 'Definitely not me']
data = np.array([
    [21, 18, 7, 9, 5],
    [27, 17, 6, 6, 4],
    [21, 20, 5, 6, 8],
    [23, 21, 8, 6, 2]
])

x = np.arange(len(labels))
width = 0.15  # Width of each bar

patterns = ['/', '\\', 'x', '-', '+']
colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99']

fig, ax = plt.subplots(figsize=(12, 7))

for i, category in enumerate(categories):
    ax.bar(x + (i - 2)*width, data[:, i], width, label=category,
           color=colors[i], hatch=patterns[i], edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=18)
ax.set_ylabel('Number of Responses', fontsize=22)
ax.set_xlabel('LLM Models', fontsize=22)
ax.set_title('User Survey Responses by LLM', fontsize=24, pad=15)

ax.legend(title='Responses', title_fontsize=20, fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left')

ax.tick_params(axis='y', labelsize=20)

plt.tight_layout()
plt.savefig('user_survey_responses_barchart.eps', format='eps', bbox_inches='tight')
plt.show()
