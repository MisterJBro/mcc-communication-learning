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


for _ in range(1_000_000):
    a = random_tunnel_columns()
    


