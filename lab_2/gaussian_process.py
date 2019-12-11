import numpy as np
import itertools as it
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import time
import scipy.stats
from pathlib import Path

class GaussianProcess:

    def __init__(self, space_dim=None, length_scale=1, noise=0.1, standardize=True):
        self.known_points = None # Matrix with known points on the rows
        self.known_values = np.empty(0) # Array with known values 
        self.noise_kernel = 0.1
        self.length_scale = length_scale
        self.standardize = standardize
        if space_dim is not None:
            self.known_points = np.empty((0, space_dim)) 

    def add_points(self, points, values):
        # Add the given point and values and update the Gaussian Process
        # points: matrix with points on the rows
        # values: array with f(points)
        if self.known_points is None:
            self.known_points = np.array(points)
            self.known_values = np.array(values)
        else:
            self.known_points = np.concatenate((self.known_points, points))
            self.known_values = np.concatenate((self.known_values, values))
        print("Computing kernel matrix")
        self.K = self._kernel_mat()
        print("Inverting Kernel")
        self.K_inv = np.linalg.inv(self.K)

        
    def predict(self, points):
        print("Predicting")
        # points: matrix with points on the rows
        # Returns: matrix with predicted mean and covariance for the given points on the rows
        n_known_points = self.known_points.shape[0]
        n_predict_points = points.shape[0]
        scaled_points = self._scale_points(points) if self.standardize else points
        scaled_known_points = self._scale_points(self.known_points) if self.standardize\
                              else self.known_points
        scaled_targets = self._scale_targets()
        prediction = np.empty((n_predict_points, 2))
        for point_idx in tqdm(range(n_predict_points)): 
            point = scaled_points[point_idx]
            # Build vector k = kernel(x*, x_n)
            k = np.array([self._kernel_func(point, x) for x in scaled_known_points])
            c = self._kernel_func(point, point) + self.noise_kernel**2
            mu = k.dot(self.K_inv).dot(scaled_targets)
            sigma = c - k.dot(self.K_inv).dot(k)
            prediction[point_idx] = [mu, sigma]
        return prediction

    def predict_unscaled(self, points):
        # Always return the real (not scaled) prediction for the given points
        # note that the covariance can't be used anymore
        prediction = self.predict(points)
        prediction[:,0] *= np.max(self.known_values)
        prediction[:,0] += np.mean(self.known_values)
        return prediction

    def most_likely_max(self, space):
        # space: list of lists with all possible values for each dimension of the search space.
        #        Predicted maximum will be searched in the cartesian product of all these lists
        # return: the parameters that have the highest probability of being a max
        scaled_known_points = self._scale_points(self.known_points) if self.standardize\
                              else self.known_points
        scaled_targets = self._scale_targets()
        cur_max_val = np.max(scaled_targets)
        combinations = list(it.product(*space)) # The whole search space
        for point in self.known_points: # We're not going to search in known points
            combinations.remove(tuple(point))
        combinations = np.array(combinations)
        scaled_combinations = combinations / np.max(combinations, axis=0) # scaling 

        predicted = self.predict(scaled_combinations)
        max_idx = None
        max_p = -float('inf')
        for index, row in enumerate(predicted):
            mu, sigma = row
            p_over_max = 1 - scipy.stats.norm.cdf(cur_max_val, loc=mu, scale=sigma)
            if p_over_max > max_p:
                max_p = p_over_max
                max_idx = index
            elif p_over_max == max_p: # Can happen bc. of precision errors in cdf
                if predicted[index, 0] > predicted[max_idx, 0]: # Take max mu
                    max_idx = index
        return combinations[max_idx]

    def get_max(self):
        max_idx = np.argmax(self.known_values)
        return self.known_points[max_idx], self.known_values[max_idx]

    def _scale_points(self, points):
        # Returns the known_point matrix in which points are scaled to [-1, 1]
        # in each dimension
        return points / np.max(np.abs(points), axis=0)

    def _scale_targets(self):
        # Returns the known_values as a zero mean array
        scaled = self.known_values
        if self.standardize:
            scaled = self.known_values / np.max(self.known_values)
        return scaled - np.mean(scaled) # This must be done anyway
 
    def _kernel_mat(self):
        # Computes the kernel matrix K for the known points
        scaled_points = self._scale_points(self.known_points) if self.standardize\
                        else self.known_points
        K = np.empty((scaled_points.shape[0], scaled_points.shape[0]))
        for i, point_i in enumerate(scaled_points):
            for j, point_j in enumerate(scaled_points):
                K[i,j] = self._kernel_func(point_i, point_j)
        K += self.noise_kernel**2 * np.identity(K.shape[0])
        return K

    def _kernel_func(self, point_i, point_j):
        # TODO: (Federico) vectorization ?
        # TODO: (Federico) how do we set sigma?
        sigma_k = 1
        return sigma_k**2 * np.exp(-np.linalg.norm(point_i - point_j)**2
                                   /(2*self.length_scale**2))


def save(dirname, known_points, known_values):
    # Save the known points and values in the given directory.
    Path(dirname+"/known_points.npy").touch()
    Path(dirname+"/known_values.npy").touch()
    np.save(dirname+"/known_points.npy", known_points)
    np.save(dirname+"/known_values.npy", known_values)

def load(dirname):
    known_points = np.load(dirname+"/known_points.npy")
    known_values = np.load(dirname+"/known_values.npy")
    return known_points, known_values


# Testing with a known function 
if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    y = np.exp(x)
    gp = GaussianProcess(length_scale=0.5, noise=0.1)

    p = [random.choice(x)]
    while True:
        # Take random point
        v = np.exp(p)
        print('Adding '+str(p)+', '+str(v))
        gp.add_points([p], v)
        p = gp.most_likely_max([x])
        print('Most likely max at '+str(p))
        
        prediction = gp.predict_unscaled(x)
        plt.plot(x, prediction[:,0], label='Prediction', c='k')
        plt.plot(x, prediction[:,0] - 2 * prediction[:,1], c='r')
        plt.plot(x, prediction[:,0] + 2 * prediction[:,1], c='r')
        plt.scatter(gp.known_points[:,0], gp.known_values)
        plt.show()

