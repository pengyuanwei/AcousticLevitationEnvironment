import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    'Naive - CBS': [0.07, 0.36, 0.68, 0.84],
    'Naive - S2M2': [0.04, 0.38, 0.66, 0.89],
    'Naive - AcoustoReinforce': [0.30, 0.57, 0.84, 0.96],
    'TWGS - CBS': [0.26, 0.58, 0.78, 0.94],
    'TWGS - S2M2': [0.20, 0.60, 0.81, 0.96],
    'TWGS - AcoustoReinforce': [0.46, 0.74, 0.91, 0.99]
}

labels = ['0.2', '0.15', '0.1', '0.05']
x = np.arange(len(labels))  # label locations
width = 0.13  # width of the bars

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

for i, (key, values) in enumerate(data.items()):
    ax.bar(x + i * width, values, width, label=key)

# Formatting
ax.set_xlabel('Max Velocity')
ax.set_ylabel('Real System Success Rate')
ax.set_title('Success Rate With Different Acoustic Hologram Solvers')
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(labels)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()