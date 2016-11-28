import os
import numpy as np
import utils as algo_utils


''' Gradients '''
class DiscreteGradient:

	def __init__(self, energy, delta = 1, \
				 components_to_change = None, \
				 positive_cutoff = 0.75):

		self.energy = energy
		self.delta = delta
		self.c2c = components_to_change
		self.positive_cutoff = positive_cutoff

	def pcomp_grad(self, state, E0):
		dims = state.size

		gradv = np.zeros(dims)

		c_idxs = np.random.choice(np.arange(dims), self.c2c)
		state_delta = np.ones((dims, 1))*state
		state_delta[c_idxs] += self.delta*np.eye(dims)[c_idxs]

		for c in c_idxs:
			gradv[c] = self.energy(state_delta[c]) - E0

		return gradv

	def ncomp_grad(self, state, E0):
		dims = state.size

		gradv = np.zeros(dims)

		nc_idxs = np.random.choice(np.arange(dims), dims - self.c2c)
		c_idxs = np.setdiff1d(np.arange(dims), nc_idxs)
		state_delta = np.ones((dims, 1))*state
		state_delta[c_idxs] += self.delta*np.eye(dims)[c_idxs]

		for c in c_idxs:
			gradv[c] = self.energy(state_delta[c]) - E0

		return gradv

	def all_components_grad(self, state, E0):
		dims = state.size

		gradv = np.zeros(dims)

		state_delta = np.ones((dims, 1))*state + self.delta*np.eye(dims)
		for i in xrange(dims):
			gradv[i] = self.energy(state_delta[i]) - E0

		return gradv

	def __call__(self, state, E0):
		if self.c2c is None or self.c2c == state.size:
			return self.all_components_grad(state, E0)
		elif self.c2c >= state.size*self.positive_cutoff:
			return self.ncomp_grad(state, E0)
		else:
			return self.pcomp_grad(state, E0)

"""
The gain controlling the norm of the velocity.
ContantGainUpdate -- a constant gain of g, so |v| = g always.

BoldGainUpdate -- there are three methods of controlling gain here
	- Linear: increase the gain by delta each time, g <- g + delta
	- Exponential: change the gain by a factor of alpha, g <- g*alpha
	- Harmonic: starting with gain g0, the gain at timestep n gn = g0/n
In the expansion phase, the gain increases exponentially until a suboptimal
change in energy occurs. Then, each optimal change in energy increases the gain
by one of the above methods, while each suboptimal change in energy decreases
the gain by one of the above methods.
"""
class ConstantGainUpdate:
	def __init__(self):
		self.g = 0

	def reset(self, g):
		self.g = g

	def __call__(self, dE):
		return self.g

class BoldGainUpdate:

	class Linear:
		def __init__(self, delta, lower = 0, upper = np.inf):
			self.delta = delta
			self.lower = lower
			self.upper = upper

		def __call__(self, g):
			g += self.delta
			return self.lower if g < self.lower else \
				   self.upper if g > self.upper else g

	class Exponential:
		def __init__(self, alpha, lower = 0, upper = np.inf):
			self.alpha = alpha
			self.lower = lower
			self.upper = upper

		def __call__(self, g):
			g *= self.alpha
			return self.lower if g < self.lower else \
				   self.upper if g > self.upper else g

	class Harmonic:
		def __init__(self, lower = 0, upper = np.inf):
			self.lower = lower
			self.upper = upper
			self.n = float(2)

		def __call__(self, g):
			g = g*(self.n - 1)/self.n
			return self.lower if g < self.lower else \
				   self.upper if g > self.upper else g

	def __init__(self, expansion_method = Exponential(1.5, upper = 100),
					   increment_method = Linear(0.01),
					   decrement_method = Exponential(0.66)):
		self.expn_meth = expansion_method
		self.incr_meth = increment_method
		self.decr_meth = decrement_method

		self.g = 0

		self.in_expansion_phase = True

	def reset(self, g):
		self.in_expansion_phase = True
		self.g = g

	def __call__(self, dE):

		if dE < 0:
			if self.in_expansion_phase:
				self.g = self.expn_meth(self.g)
			else:
				self.g = self.incr_meth(self.g)
		else:
			self.in_expansion_phase = False
			self.g = self.decr_meth(self.g)
		return self.g

"""
Standard move: change the current state in the direction dirn (the optimal direction determined
			   by the gradient) for distance given by the gain.
Momentum move: set the new direction as a combination of the old direction and dirn (as above),
			   with weights given by the mass and the gain.
"""
def StandardMove(state, dirn, gain, epsilon = 1e-3):
	dirn_norm = np.linalg.norm(dirn)
	dx = dirn if dirn_norm < epsilon else dirn/dirn_norm

	state -= gain*dx
	return state

class MomentumMove:

	def __init__(self, state_size, mass = 0.5):
		self.mass = mass
		self.velocity = np.zeros(state_size)

	def __call__(self, dirn, gain, epsilon = 1e-3):
		dirn_norm = np.linalg.norm(dirn)
		dx = dirn if dirn_norm < epsilon else dirn/dirn_norm

		self.velocity = self.mass*self.velocity + gain*self.dx
		return state - self.velocity

''' MEAT 'N' POTATOES '''
class GradientDescent:

	LoggingDefaults = {
		'track_time_taken': False,
		'track_func_calls': False,
		'log_names'		  : [],
		'base_path'		  : '.',
		'debug'			  : False
	}

	Defaults = {
		# State and gain updates
		'gain'	  : BoldGainUpdate(),
		'move'	  : StandardMove,
		'state_or_shape': 16,

		# GradDesc search params
		'min_iters'    : 10,
		'max_iters'    : 500,
		'starting_gain': 0.2,
		'epsilon'	   : 1,
		'avg_weight'   : 0.6,
		'restart'	   : 0,
		'opt_thresh'   : np.NINF
	}

	def __init__(self, raw_energy, gradient, base_path = '.', **kwargs):

		self.raw_energy = raw_energy
		self.gradient = gradient

		for param in GradientDescent.LoggingDefaults:
			value = kwargs[param] if param in kwargs \
								  else GradientDescent.LoggingDefaults[param]
			setattr(self, param, value)

		for param in GradientDescent.Defaults:
			value = kwargs[param] if param in kwargs \
								  else GradientDescent.Defaults[param]
			setattr(self, param, value)

		self.reset(self.state_or_shape)

		if self.track_time_taken:
			self.time_taken = 0

		if self.track_func_calls:
			self.func_calls = 0

		self.logs_id = algo_utils.gen_time_id()
		self.logs = algo_utils.open_logs(self.base_path, self.log_names, self.logs_id)

	def write_log(self, log_name, msg):
		if log_name in self.logs:
			self.logs[log_name].write(msg)

	def energy(self, state):
		if self.track_func_calls:
			self.func_calls += 1
		return self.raw_energy(state)

	def reset(self, state_or_shape):
		if type(state_or_shape) == np.ndarray:
			self.state_shape = state_or_shape.shape
			self.state0 = state_or_shape.reshape(state_or_shape.size)
		elif type(state_or_shape) == tuple:
			self.state_shape = state_or_shape
			self.state0 = algo_utils.random_state(state_or_shape.size)

		self.E0 = self.raw_energy(self.state0)

		self.gain.reset(self.starting_gain)

	def run(self):

		state = np.copy(self.state0)

		E = np.array([self.E0, self.E0])
		avg_E = np.array([self.E0, self.E0])

		g = self.starting_gain
		self.gain.reset(g)

		s, t0 = 0, timer()
		while s < self.max_iters:

			self.write_log('time_x_E', '{0},{1},{2}\n'.format(timer()-t0, E[0], avg_E[0]))
			self.write_log('time_x_gain', '{0},{1}\n'.format(timer()-t0, g))

 			state = self.move(state, self.gradient(state, E[0]), g)

 			E[-1] = E[0]
			E[0] = self.energy(state)

			dE = E[0] - E[-1]
			g = self.gain(dE)

			if self.debug:
				print "S: {0} / G: {1: .2f} / E: {2:.2f}".format(s, g, E[0])
			self.write_log('time_x_dE', '{0},{1}\n'.format(timer()-t0, dE))

			# Exponential weighted average with alpha = self.avg_weight
			avg_E[-1] = avg_E[0]
			avg_E[0] = self.avg_weight*avg_E[-1] + (1 - self.avg_weight)*E[0]

			if (s > self.min_iters) and (abs(avg_E[0] - avg_E[-1]) < self.epsilon):
				break

			if E[0] < self.opt_thresh:
				break

			s += 1

		if self.track_time_taken:
			self.time_taken = timer() - t0

		for log in self.logs:
			log.close()

		return state.reshape(self.state_shape), E[0]
