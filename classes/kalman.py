import numpy as np

class Kalman():

    def __init__(self, A, B, R, C, Q, mean, covariance) -> None:
        self.A = A
        self.B = B
        self.R = R
        self.C = C
        self.Q = Q

        self.mean = mean
        self.covariance = covariance

    def predict(self, control_signal):
        x = self.A @ self.mean + self.B @ control_signal
        p = self.A @ self.covariance @ self.A.T + self.R
        return x, p
    
    def measurement_update(self, x, p, z):
        kalman_gain = p @ self.C.T * np.linalg.inv(self.C @ p @ self.C.T + self.Q)
        self.mean = x + kalman_gain @ (z - self.C @ x)
        self.covariance = (np.identity((kalman_gain @ self.C).shape[0]) -  kalman_gain @ self.C) @ p
        return self.mean, self.covariance
    
    def filter(self, control_signal, measurement):
        x, p = self.predict(control_signal)
        mean, covariance = self.measurement_update(x, p, measurement)
        return mean, covariance