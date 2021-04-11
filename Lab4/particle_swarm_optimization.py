import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        # [DONE] Todo: implement

        # Position
        self.position = np.random.uniform(lower_bound, upper_bound)
        
        # J()
        self.j = -inf
        
        # Best Position
        self.my_best_position = self.position
        self.my_best_j = self.j

        # Velocity
        delta = upper_bound - lower_bound
        self.velocity = np.random.uniform(-delta, delta)


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):
        # [DONE] Todo: implement
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.hyperparams = hyperparams
        
        # Initialize particles
        self.num_particles = hyperparams.num_particles
        self.particles = [Particle(lower_bound, upper_bound) for i in range(self.num_particles)]

        # Initial best global
        self.best_position = None
        self.best_j = -inf

        # Actual position index
        self.actual_index = -1

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        # [DONE] Todo: implement
        return self.best_position

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        # [DONE] Todo: implement
        return self.best_j

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        # [DONE] Todo: implement

        # Update index
        self.actual_index = (self.actual_index + 1) % self.num_particles

        # Return
        return self.particles[self.actual_index].position

    def advance_generation(self):
        """
        Advances the generation of particles. Auxiliary method to be used by notify_evaluation().
        """
        # [DONE] Todo: implement
        iw = self.hyperparams.inertia_weight
        cp = self.hyperparams.cognitive_parameter
        sp = self.hyperparams.social_parameter

        best_iteraction = None
        for particle in self.particles:
                rp = random.uniform(0.0, 1.0)
                rg = random.uniform(0.0, 1.0)
                
                # Update Velocity
                particle.velocity = iw * particle.velocity + cp * rp * (particle.my_best_position - particle.position) + sp * rg * (self.best_position - particle.position)
                
                # Update Position
                particle.position = particle.position + particle.velocity

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        # [DONE] Todo: implement
        actual_particle = self.particles[self.actual_index]

        # Update particle value
        actual_particle.j = value

        # Update "bests"
        if value > actual_particle.my_best_j:
                actual_particle.my_best_j = value
                actual_particle.my_best_position = actual_particle.position
        if value > self.best_j:
                self.best_j = value
                self.best_position = actual_particle.position

        # Check if must advance generation
        if self.actual_index == self.num_particles - 1: # Last Particle
                self.advance_generation()

