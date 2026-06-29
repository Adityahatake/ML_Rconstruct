import numpy as np
import matplotlib.pyplot as plt

# Discrete Random Variable — 2 coin flips
outcomes = {0: 1/4, 1: 2/4, 2: 1/4}

print("X = number of heads in 2 flips")
print("-" * 30)
for value, prob in outcomes.items():
    print(f"P(X={value}) = {prob:.2f}")
print(f"Sum = {sum(outcomes.values()):.2f}  ← must be 1")

# Plot the PMF
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(outcomes.keys(), outcomes.values(), color='steelblue', edgecolor='black', width=0.4)
plt.title("PMF — Discrete (Coin Flips)")
plt.xlabel("Number of Heads")
plt.ylabel("Probability")
plt.xticks([0, 1, 2])

# Continuous Random Variable — heights
plt.subplot(1, 2, 2)
heights = np.random.normal(loc=170, scale=10, size=10000)
plt.hist(heights, bins=50, density=True, color='coral', edgecolor='black', alpha=0.7)
plt.title("PDF — Continuous (Heights)")
plt.xlabel("Height (cm)")
plt.ylabel("Density")

plt.tight_layout()
plt.show()