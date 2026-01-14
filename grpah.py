import matplotlib.pyplot as plt
import numpy as np

# Data from the table
stages = ['Normal', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']

metrics = {
    "Precision (PPV)": {
        "Testing": [0.43, 0.22, 0.40, 0.92, 0.80],
        "Final": [0.50, 0.19, 0.29, 1.00, 0.95]
    },
    "Recall (Sensitivity)": {
        "Testing": [0.54, 0.25, 0.33, 0.50, 1.00],
        "Final": [0.35, 0.25, 0.40, 0.55, 1.00]
    },
    "Specificity": {
        "Testing": [0.82, 0.78, 0.88, 0.99, 0.94],
        "Final": [0.91, 0.74, 0.75, 1.00, 0.99]
    },
    "F1-score": {
        "Testing": [0.48, 0.24, 0.36, 0.65, 0.89],
        "Final": [0.41, 0.22, 0.33, 0.71, 0.98]
    },
    "Accuracy": {
        "Testing": [0.77, 0.68, 0.77, 0.89, 0.95],
        "Final": [0.80, 0.64, 0.68, 0.91, 0.99]
    }
}

# Plot setup
plt.figure(figsize=(12, 8))
x = np.arange(len(stages))
width = 0.35

# Plot each metric separately
for i, (metric, values) in enumerate(metrics.items(), 1):
    plt.subplot(3, 2, i)
    plt.bar(x - width/2, values["Testing"], width, label='Testing', alpha=0.7)
    plt.bar(x + width/2, values["Final"], width, label='Final', alpha=0.7)
    plt.xticks(x, stages, rotation=45)
    plt.ylim(0, 1.1)
    plt.title(metric)
    plt.xlabel("Disease Stage")
    plt.ylabel("Score")
    plt.legend()

plt.tight_layout()
plt.suptitle("Performance Metrics of the Two-Stage CNN on Panoramic Radiographs", fontsize=14, y=1.03)
plt.show()
