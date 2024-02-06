import numpy as np
from classes.kalman import Kalman
from classes.simulation import Simulation

mean_wind = 0
std_dev_wind = 1

gps_measurement_noise = 8
deltaT = 1
A = np.array([[1, deltaT], [0, 1]])
B = np.array([[1/2 * (deltaT**2)], [deltaT]])
R = std_dev_wind * (B @ B.T)

C = np.array([1, 0])
C = np.reshape(C, (1, 2))
Q = np.array([gps_measurement_noise])

# Question 3.1
B = np.array([[1/2 * (deltaT**2)], [deltaT]]) # Because acceleration of 1 will increase velocity by 1 and position by 1/2
initial_x = 5.0
initial_v = 1.0
initial_state = np.array([[initial_x], [initial_v]])
initial_variance = np.array([[0,0], [0,0]])
initial_a = 1
initial_control = np.array([[initial_a]]) 

# Ax + BU + N([0, 0]^T, R)

question3_kalman = Kalman(A, B, R, C, Q, initial_state, initial_variance)
mean, covariance = question3_kalman.predict(initial_control)
print(mean)
print(covariance)
noise = np.random.multivariate_normal(np.array([0,0]), R, 1)
print(noise)