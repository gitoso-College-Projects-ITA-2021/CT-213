import numpy as np
import time

class SimpleEvolutionStrategy:
    """
    Represents a simple evolution strategy optimization algorithm.
    The mean and covariance of a gaussian distribution are evolved at each generation.
    """
    def __init__(self, m0, C0, mu, population_size):
        """
        Constructs the simple evolution strategy algorithm.

        :param m0: initial mean of the gaussian distribution.
        :type m0: numpy array of floats.
        :param C0: initial covariance of the gaussian distribution.
        :type C0: numpy matrix of floats.
        :param mu: number of parents used to evolve the distribution.
        :type mu: int.
        :param population_size: number of samples at each generation.
        :type population_size: int.
        """
        self.m = m0
        self.C = C0
        self.mu = mu
        self.population_size = population_size
        self.samples = np.random.multivariate_normal(self.m, self.C, self.population_size)

    def ask(self):
        """
        Obtains the samples of this generation to be evaluated.
        The returned matrix has dimension (population_size, n), where n is the problem dimension.

        :return: samples to be evaluated.
        :rtype: numpy array of floats.
        """
        return self.samples

    def tell(self, fitnesses):
        """
        Tells the algorithm the evaluated fitnesses. The order of the fitnesses in this array
        must respect the order of the samples.

        :param fitnesses: array containing the value of fitness of each sample.
        :type fitnesses: numpy array of floats.
        """
        # [DONE] Todo: implement this method
        
        # Rank
        indices = np.argsort(fitnesses)
        best_samples = self.samples[indices[0:self.mu], :]

        # Sigma in m update
        Sigma = np.zeros(np.size(self.m, 0))
        for i in range(self.mu):
                Sigma += best_samples[i]

        # Update m
        m = (1 / self.mu) * Sigma

        # Sigma in C update
        Sigma = np.zeros((np.size(self.C, 0), np.size(self.C, 1)))
        for i in range(self.mu):
                vect = best_samples[i] - self.m
                vect2d = vect[np.newaxis]
                vect2d_T = vect2d.transpose()

                Sigma += vect2d * vect2d_T


        # Update C
        C = (1 / self.mu) * Sigma

        # Update m and C values
        self.m = m
        self.C = C
        
        # Get new samples
        self.samples = np.random.multivariate_normal(self.m, self.C, self.population_size)

