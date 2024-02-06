import numpy as np
from classes.kalman import Kalman

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

def confidence_ellipse(x, y, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    # Taken from matplotlib documentation https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

if __name__ == "__main__":


    mean_wind = 0
    std_dev_wind = 1
    # Question 1.1
    # Minimal state vector x = [x, v]^T

    # Question 1.2 
    # The state transition p(xt|ut, xt−1)
    # [x, v]^T = [[1, deltaT], [0, 1]][x, v]^T + [1/2deltaT, deltaT] * [a] + [[1/4, 1/2], [1/2, 1]] [std_dev_wind]
    deltaT = 1
    A = np.array([[1, deltaT], [0, 1]])
    B = np.array([[1/2 * (deltaT**2)], [deltaT]])
    R = std_dev_wind * (B @ B.T)

    # Question 1.3
    initial_x = 0
    initial_v = 0
    initial_state = np.array([[initial_x], [initial_v]])
    initial_variance = np.array([[0,0], [0,0]])
    initial_a = 0
    initial_control = np.array([[initial_a]])
    question1_kalman = Kalman(A, B, R, None, None, initial_state, initial_variance)

    history_of_means = [initial_state]
    history_of_variances = [initial_variance]
    for t in range(5):
        mean, variance = question1_kalman.predict(initial_control)
        question1_kalman.mean = mean
        question1_kalman.covariance = variance
        print(mean)
        print(variance)
        history_of_means.append(mean)
        history_of_variances.append(variance)

    # Question 1.4
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.set_xlim([-20, 20])
    ax.set_ylim([-8, 8])
    for state in range(len(history_of_means)):
        x = history_of_means[state][0]
        y = history_of_means[state][1]
        cov = history_of_variances[state]

        confidence_ellipse(x, y, cov, ax, edgecolor='red')
    plt.show()
    print("Pause to see graph")


