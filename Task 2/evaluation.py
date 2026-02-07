def mean_absolute_error(y_true, y_pred):
    error = 0.0
    for yt, yp in zip(y_true, y_pred):
        error += abs(yt - yp)
    return error / len(y_true)


def mean_squared_error(y_true, y_pred):
    error = 0.0
    for yt, yp in zip(y_true, y_pred):
        error += (yt - yp) ** 2
    return error / len(y_true)


def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)

    ss_res = 0.0
    ss_tot = 0.0

    for yt, yp in zip(y_true, y_pred):
        ss_res += (yt - yp) ** 2
        ss_tot += (yt - mean_y) ** 2

    return 1 - (ss_res / ss_tot)