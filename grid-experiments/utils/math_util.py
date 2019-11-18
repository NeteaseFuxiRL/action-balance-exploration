import numpy as np

eps = 1e-10


def handle_all_zeros(x):
    """

    :param x:
    :return:
    """
    absc = np.all(x == 0, axis=-1)
    x[absc, :] = 1. / x.shape[-1]

    return x


def normalize(x):
    """
    x / sum(x)
    :param x:
    :return:
    """
    x = x_tmp = np.array(x)

    if x.size == 0:
        return x

    if len(x.shape) == 1:
        x_tmp = x.reshape(1, -1)

    # handle negative number
    neg_index = np.any(x_tmp < 0, axis=-1)
    x_tmp[neg_index] = x_tmp[neg_index] + abs(np.min(x_tmp[neg_index], axis=-1).reshape(-1, 1))

    x_sum = np.sum(x_tmp, axis=1).reshape(-1, 1)

    # add eps to sum zeros
    absc = np.all(x_sum == 0, axis=-1)
    x_sum[absc] = np.add(x_sum[absc], eps)

    normalized = x_tmp / x_sum
    normalized = normalized.reshape(x.shape)
    normalized = handle_all_zeros(normalized)
    return normalized


def normalize2(x):
    x = x_tmp = np.array(x)
    if len(x.shape) == 1:
        x_tmp = x.reshape(1, -1)

    min_x = np.min(x_tmp, axis=-1).reshape(-1, 1)
    max_x = np.max(x_tmp, axis=-1).reshape(-1, 1)
    normalized = (x_tmp - min_x) / np.add(max_x - min_x, eps)
    normalized = normalized.reshape(x.shape)
    normalized = handle_all_zeros(normalized)
    return normalized


def normalize_v3(x):
    x = x_tmp = np.array(x)
    if len(x.shape) == 1:
        x_tmp = x.reshape(1, -1)

    x_exp = np.exp(x_tmp)
    normalized = x_exp / np.sum(x_exp, axis=-1)

    normalized = normalized.reshape(x.shape)
    normalized = handle_all_zeros(normalized)
    return normalized


def scale_neg1_to_pos1(x):
    x = np.array(x)
    scaled = np.zeros_like(x, dtype=np.float64)

    # x -= np.sum(x, axis=-1, keepdims=True)

    neg_index = (x < 0)

    scaled[neg_index] = -normalize(np.abs(np.where(neg_index, x, 0)))[neg_index]
    scaled[~neg_index] = normalize(np.where(~neg_index, x, 0))[~neg_index]

    return scaled


def outliers_iqr(ys, tolerance_factor=1.5):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * tolerance_factor)
    upper_bound = quartile_3 + (iqr * tolerance_factor)
    # return np.where((ys > upper_bound) | (ys < lower_bound))
    return ys < lower_bound


def entropy(logits, temperature=1.):
    a0 = logits - np.max(logits, axis=-1, keepdims=True)

    a0 = a0 / temperature
    ea0 = np.exp(a0)
    z0 = np.sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return np.sum(p0 * (np.log(z0) - a0), axis=-1)

def test():
    ones = np.ones([10])
    ones[-1] = 0
    print(outliers_iqr(ys=ones))

    x = [[0.1, 0.3, -0.1], [0.2, 0.3, 0.1], [0,0,0], [0.1, 0.4, 0.5]]

    y = [0, 0, 0]
    print(normalize(x))
    # print(normalize2(x))
    print(x)

    # a = [0.51, 0.5, 0.49, 0.4, 0.2]
    a = [0.8]
    print(outliers_iqr(a, tolerance_factor=1.0))

    # print(scale_neg1_to_pos1([146,76,-61, -161]))


if __name__ == "__main__":
    test()
