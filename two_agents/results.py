import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

results = np.array([
    0.169,
    0.325,
    0.263,
    0.279,
    0.388,
    0.311,
    0.327,
    0.529,
    0.529,
    0.309,
    0.512,
    0.684,
    0.451,
    0.42,
    0.468,
    0.561,
    0.452,
    0.671,
    0.577,
    0.781,
    0.562,
    0.656,
    0.749,
    0.702,
    0.812,
    0.781,
    0.812,
    0.938,
    1.032,
    0.891,
    0.891,
    1.235,
    1.141,
    1.72,
    1.86,
    1.704,
    1.813,
    1.782,
    1.797,
    2.329,
    1.922,
    2.172,
    2.422,
    2.969,
    2.938,
    2.547,
    2.86,
    2.938,
    2.938,
    2.938,
    3.188,
    3.157,
    2.797,
    3.094,
    3.094,
    3.109,
    3.516,
    3.469,
    3.266,
    3.453,
    3.0,
    3.281,
    3.187,
    3.016,
    3.359,
    3.625,
    3.578,
    3.562,
    3.375,
    3.922,
    3.688,
    3.797,
    3.859,
    3.813,
    3.875,
    3.422,
    3.812,
    3.75,
    3.766,
    3.563,
    3.828,
    3.484,
    3.688,
    3.86,
    3.86,
    3.75,
    3.703,
    3.782,
    3.75,
    3.594,
    3.656,
    3.797,
    3.828,
    3.734,
    3.766,
    3.719,
    3.953,
    3.891,
    3.828,
    3.687,
    3.578,
    3.75,
    3.719,
    3.688,
    3.594,
    3.656,
    3.625,
    3.781,
    3.938,
    4.063,
    3.781,
    3.782,
    4.0,
    3.782, 3.485, 4.376, 3.922, 4.016, 4.126, 4.501, 3.907, 4.298, 4.172, 4.266, 3.969, 4.235, 4.188, 4.298, 4.266, 4.157, 3.766, 4.47, 4.141, 4.157, 4.282, 4.376, 4.313, 4.235, 4.313, 4.423, 4.126, 4.392, 4.298, 4.329, 4.501, 4.267, 4.501, 4.36, 4.079, 4.407, 4.392, 4.564, 4.283, 4.533, 4.314, 4.626, 4.361, 4.392, 4.579, 4.407, 4.204, 4.486, 4.47, 4.314, 4.345, 4.595, 4.564, 4.767, 4.548, 4.392, 4.798, 4.58, 4.658, 4.454, 4.361, 4.501, 4.845, 4.58, 5.064, 4.783, 4.548, 4.611, 4.892, 4.58, 5.127, 4.658, 4.658, 4.893, 4.689, 4.658, 4.986, 5.017, 4.72, 4.877, 4.752, 5.158, 4.814, 4.939,
])

sns.set()
plt.plot(results)
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('Average Reward')
plt.show()
