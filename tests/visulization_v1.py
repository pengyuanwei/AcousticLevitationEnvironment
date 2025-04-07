import matplotlib.pyplot as plt

def plot_stability_rates():
    velocities = [0.2, 0.15, 0.1, 0.05]
    marker = ['o', 'x']
    
    data = {
        'Naive - CBS': [0.07, 0.36, 0.68, 0.84],
        'Naive - S2M2': [0.04, 0.38, 0.66, 0.89],
        'Naive - AcoustoReinforce': [0.30, 0.57, 0.84, 0.96],
        'TWGS - CBS': [0.26, 0.58, 0.78, 0.94],
        'TWGS - S2M2': [0.20, 0.60, 0.81, 0.96],
        'TWGS - AcoustoReinforce': [0.46, 0.74, 0.91, 0.99]
    }
    
    plt.figure(figsize=(8, 6))
    for label, values in data.items():
        plt.plot(velocities, values, marker='o', linestyle='-', label=label)
    
    plt.xlabel('Vmax (m/s)')
    plt.ylabel('Stability Rate')
    plt.title('8-Particle Solution Stability Rate at Different Vmax')
    plt.xticks(velocities)
    plt.legend()
    plt.grid(True)
    plt.savefig("figure_1.png")
    plt.show()

# 调用函数绘制图表
plot_stability_rates()


# Data extracted from the table
particle_counts = [4, 6, 8, 10]
methods = ['CBS', 'S2M2', 'AcoustoReinforce']

# Data for each method and Vmax value
CBS_0_05 = [1.000, 1.000, 0.9400, 0.5200]
S2M2_0_05 = [1.000, 0.9700, 0.9600, 0.4900]
AcoustoReinforce_0_05 = [1.000, 0.9900, 0.9900, 0.6600]

CBS_0_10 = [1.000, 0.9000, 0.7800, 0.2400]
S2M2_0_10 = [1.000, 0.9200, 0.8100, 0.2300]
AcoustoReinforce_0_10 = [1.000, 0.9400, 0.9100, 0.3400]

CBS_0_15 = [1.000, 0.7600, 0.5800, 0.0100]
S2M2_0_15 = [1.000, 0.7900, 0.6000, 0.0000]
AcoustoReinforce_0_15 = [1.000, 0.8400, 0.7400, 0.0800]

CBS_0_20 = [0.7900, 0.3700, 0.2600, 0.0000]
S2M2_0_20 = [0.7400, 0.3800, 0.2000, 0.0000]
AcoustoReinforce_0_20 = [0.9200, 0.6700, 0.4600, 0.0000]


# Plotting
plt.figure(figsize=(10, 6))

# Define line styles and colors for each method
line_styles = ['--', '--', '-']
colors = ['blue', 'green', 'red']

# Plot lines for each method at different Vmax values
plt.plot(particle_counts, CBS_0_05, color=colors[0], marker='o', linestyle='--', label=f'CBS - Vmax=0.05m/s')
plt.plot(particle_counts, S2M2_0_05, color=colors[1], marker='o', linestyle='--', label=f'S2M2 - Vmax=0.05m/s')
plt.plot(particle_counts, AcoustoReinforce_0_05, color=colors[2], marker='o', linestyle='-', label='A.R. - Vmax=0.05m/s')

plt.plot(particle_counts, CBS_0_10, color=colors[0], marker='s', linestyle='--', label='CBS - Vmax=0.10m/s')
plt.plot(particle_counts, S2M2_0_10, color=colors[1], marker='s', linestyle='--', label='S2M2 - Vmax=0.10m/s')
plt.plot(particle_counts, AcoustoReinforce_0_10, color=colors[2], marker='s', linestyle='-', label='A.R. - Vmax=0.10m/s')

plt.plot(particle_counts, CBS_0_15, color=colors[0], marker='^', linestyle='--', label='CBS - Vmax=0.15m/s')
plt.plot(particle_counts, S2M2_0_15, color=colors[1], marker='^', linestyle='--', label='S2M2 - Vmax=0.15m/s')
plt.plot(particle_counts, AcoustoReinforce_0_15, color=colors[2], marker='^', linestyle='-', label='A.R. - Vmax=0.15m/s')

plt.plot(particle_counts, CBS_0_20, color=colors[0], marker='x', linestyle='--', label='CBS - Vmax=0.20m/s')
plt.plot(particle_counts, S2M2_0_20, color=colors[1], marker='x', linestyle='--', label='S2M2 - Vmax=0.20m/s')
plt.plot(particle_counts, AcoustoReinforce_0_20, color=colors[2], marker='x', linestyle='-', label='A.R. - Vmax=0.20m/s')


# Customizing the plot
plt.title('Solution Stability Rate at Different Vmax and Particle Numbers')
plt.xlabel('Particle Number')
plt.ylabel('Stability Rate')
plt.xticks(particle_counts)
plt.legend()
# Display the plot
plt.grid(True)
plt.tight_layout()
plt.savefig("figure_2.png")
plt.show()
