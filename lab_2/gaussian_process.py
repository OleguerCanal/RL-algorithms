import numpy as np
import itertools as it
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import time
import scipy.stats

class GaussianProcess:

    def __init__(self, space_dim=None):
        self.known_points = None # Matrix with known points on the rows
        self.known_values = np.empty(0) # Array with known values 
        self.noise_kernel = 0
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
        prediction = np.empty((n_predict_points, 2))
        for point_idx in tqdm(range(n_predict_points)): 
            point = points[point_idx]
            # Build vector k = kernel(x*, x_n)
            k = np.empty(n_known_points)
            for i in range(n_known_points):
                k[i] = self._kernel_func(point, self.known_points[i])
            c = self._kernel_func(point, point) + self.noise_kernel**2
            mu = k.dot(self.K_inv).dot(self.known_values)
            sigma = c - k.T.dot(self.K_inv).dot(k)
            prediction[point_idx] = [mu, sigma]
        return prediction

    def most_likely_max(self, space):
        # space: list of lists with all possible values for each dimension of the search space.
        #        Predicted maximum will be searched in the cartesian product of all these lists
        # return: the parameters that have the highest probability of being a max
        combinations = np.array(list(it.product(*space))) # matrix with all possible combinations
                                                          # on the rows
        predicted = self.predict(combinations)
        cur_max_val = np.argmax(self.known_values)
        max_idx = None
        max_p = -float('inf')
        for index, row in enumerate(predicted):
            mu, sigma = row
            p_over_max = 1 - scipy.stats.norm.cdf(cur_max_val, loc=mu, scale=sigma)
            if p_over_max > max_p:
                max_p = p_over_max
                max_idx = index
        cur_max_point, cur_max_val = self.get_max()
        print("Current max is at x: "+str(cur_max_point)+" with value: "+str(cur_max_val))
        print("best "+str(predicted[max_idx])+" with p "+str(max_p))
        return combinations[max_idx]

    def get_max(self):
        max_idx = np.argmax(self.known_values)
        return self.known_points[max_idx], self.known_values[max_idx]
            
    def _kernel_mat(self):
        # Computes the kernel matrix K for the known points
        K = np.empty((self.known_points.shape[0], self.known_points.shape[0]))
        for i, point_i in enumerate(self.known_points):
            for j, point_j in enumerate(self.known_points):
                K[i,j] = self._kernel_func(point_i, point_j)
        K += self.noise_kernel**2 + np.identity(K.shape[0])
        return K

    def _kernel_func(self, point_i, point_j):
        # TODO: (Federico) vectorization
        # TODO: (Federico) how do we set/decide these parameters?
        sigma_k = 1000
        l = 10
        return sigma_k**2 * np.exp(-np.linalg.norm(point_i - point_j)**2/(2*l**2))

  


def init():
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 22000)

def update(frame):
    idx = random.randint(0, x.shape[0]-1)
    point = x[idx]
    target = y[idx]
    gp.add_points([point], [target])
    prediction = gp.predict(x)
    ln.set_data(x, prediction[:,0])
    return ln


if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    y = np.exp(x)
    gp = GaussianProcess()


    p = [random.choice(x)]
    while True:
        # Take random point
        v = np.exp(p)
        print("p = "+str(p)+", v = "+str(v))

        gp.add_points([p], v)
        p = gp.most_likely_max([np.linspace(-10,10,1000)])
        print("Most likely max is at "+str(p))
        
        prediction = gp.predict(x)
        plt.plot(x, prediction[:,0], label='Prediction', c='k')
        plt.scatter(gp.known_points[:,0], gp.known_values)
        plt.ylim(0, 22000)
        plt.show()

