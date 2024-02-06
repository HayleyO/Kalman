import random
import numpy as np
import matplotlib.pyplot as plt
from classes.kalman import Kalman
from classes.simulation import Simulation

if __name__ == "__main__":

    mean_wind = 0
    std_dev_wind = 1

    gps_measurement_noise = 8
    deltaT = 1
    A = np.array([[1, deltaT], [0, 1]])
    B = np.array([[1/2 * (deltaT**2)], [deltaT]])
    R = std_dev_wind * (B @ B.T)
    
    # Question 2.1
    C = np.array([1, 0])
    C = np.reshape(C, (1, 2))
    Q = np.array([gps_measurement_noise])

    # Question 2.2 
    initial_state = np.array([[0], [0]])
    initial_covariance = np.array([[41.25, 12.5], [12.5, 5]]) # Covariance at time t = 5
    initial_a = 0
    initial_control = np.array([[initial_a]])
    measurement = np.array([10])
    print(initial_state)
    print(initial_covariance)
    question2_kalman = Kalman(A, B, R, C, Q, initial_state, initial_covariance)
    mean, covariance = question2_kalman.measurement_update(question2_kalman.mean, question2_kalman.covariance, measurement)
    print(mean)
    print(covariance)

    true_position = 0
    simulator = Simulation(true_position, gps_measurement_noise)
    gps_measure = simulator.sample(1)
    print(gps_measure)

    # Question 2.3
    T = 20-5 #I'm at timestep 5 so 15 more times would put me at 20
    N = 1000
    p_gps_fail = [0.1, 0.5, 0.9]
    final_predictions = np.zeros((len(p_gps_fail), N))
    for n in range(N):
        for p in range(len(p_gps_fail)):
            for t in range(T):
                if random.random() > p_gps_fail[p]:
                    # Sucessfully simulate   
                    gps_measure = simulator.sample(1)
                    mean, covariance = question2_kalman.predict(initial_control)
                    mean, covariance = question2_kalman.measurement_update(mean, covariance, np.array([measurement]))
                else:
                    # Fail to measure, just predict
                    mean, covariance = question2_kalman.predict(initial_control)
                question2_kalman.mean = mean
                question2_kalman.covariance = covariance
            final_predictions[p, n] = question2_kalman.mean[0]
    error = np.zeros((len(p_gps_fail), N)) - final_predictions

    error = np.mean(error, axis=1)


    plt.plot(p_gps_fail, error)
    plt.ylabel('Error')
    plt.xlabel('Probability of GPS Failure')
    plt.show()
    print("Done")

