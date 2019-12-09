import numpy as np
import itertools as it
from matplotlib import pyplot as plt
import random

class GaussianProcess:

    def __init__(self, space_dim):
        # param_values[i] is a vector with all possible values for i-th parameter
        self.known_points = np.empty((0, space_dim)) # Matrix with known points on the rows
        self.known_values = np.empty(0) # Array with known values 

    def add_point(self, point, val):
        # Takes the function value at the given point and updates the whole gaussian process
        # point: vector
        # val: scalar value correspondent to f(point)
        # TODO: Improve dynamic increment of known_points and known_values
        self.known_points = np.concatenate((self.known_points,np.matrix(point)))
        self.known_values = np.concatenate((self.known_values, [val]))
        self.K = self._kernel_mat()
        self.K_inv = np.linalg.inv(self.K)

        
    def predict(self, points):
        # points: matrix with points on the rows
        # Returns: matrix with predicted mean and covariance for the given points on the rows
        n_known_points = self.known_points.shape[0]
        n_predict_points = points.shape[0]
        prediction = np.empty((n_predict_points, 2))
        for point_idx in range(n_predict_points): 
            point = points[point_idx]
            # Build vector k = kernel(x*, x_n)
            k = np.empty(n_known_points)
            for i in range(n_known_points):
                k[i] = self._kernel_func(point, self.known_points[i])
            c = self._kernel_func(point, point) 
            mu = k.dot(self.K_inv).dot(self.known_values)
            sigma = c - k.T.dot(self.K_inv).dot(k)
            prediction[point_idx] = [mu, sigma]
        return prediction

    def _kernel_mat(self):
        # Computes the kernel matrix K for the known points
        K = np.empty((self.known_points.shape[0], self.known_points.shape[0]))
        for i, point_i in enumerate(self.known_points):
            for j, point_j in enumerate(self.known_points):
                K[i,j] = self._kernel_func(point_i, point_j)
        return K

    def _kernel_func(self, point_i, point_j):
        # TODO: (Federico) vectorization
        # TODO: (Federico) how do we set/decide these parameters?
        sigma_k = 1
        l = 1
        return sigma_k**2 * np.exp(-np.linalg.norm(point_i - point_j)**2/(2*l**2))






if __name__ == "__main__":
    x = np.linspace(-2, 2, 1000)
    y = np.exp(x)


    gp = GaussianProcess(1)
    for index in random.sample(range(x.shape[0]), 10):
        gp.add_point(x[index], y[index])

    prediction = gp.predict(x)
    plt.plot(x, prediction[:,0], label="predicted")   
    plt.plot(x, y, label="real")   
    plt.show()
