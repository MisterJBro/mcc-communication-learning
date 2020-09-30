import numpy as np


def random_tunnel_columns() -> np.array:
    columns = []
    ranges = [(0, 22)]

    for _ in range(4):
        if len(ranges) == 0:
            return random_tunnel_columns()
        rand_index = np.random.randint(0, len(ranges))
        cur_range = ranges[rand_index]

        col = np.random.randint(cur_range[0], cur_range[1])
        columns.append(col)

        left_range = (cur_range[0], (col-3))
        right_range = ((col + 3), cur_range[1])

        if left_range[0] < left_range[1]:
            ranges.append(left_range)

        if right_range[0] < right_range[1]:
            ranges.append(right_range)

        del ranges[rand_index]

    return np.sort(np.array(columns, dtype=np.uint8))


test = np.zeros(22)

for _ in range(1_000_000):
    a = random_tunnel_columns()
    b = np.zeros((4, 22))
    b[np.arange(4), a] = 1
    b = b.sum(0)
    test += b

print(np.round(test/100_000, 2))

# 1_000_000 With bias 0:
# [3.52 2.4  1.88 1.96 1.94 1.92 1.9  1.72 1.64 1.61 1.57 1.52 1.47 1.44
# 1.45 1.51 1.5  1.47 1.47 1.66 1.94 2.5 ]

# 1_000_000 With random, still biased?:
# [2.74 2.01 1.65 1.71 1.71 1.7  1.7  1.66 1.64 1.64 1.65 1.64 1.62 1.63
# 1.65 1.71 1.7  1.65 1.56 1.8  2.19 3.03]

