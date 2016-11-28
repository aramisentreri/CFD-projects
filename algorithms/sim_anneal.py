from __future__ import print_function

import utils as algo_utils
import numpy as np
import os
import pickle
import sys
from timeit import default_timer as timer
from math import sqrt, exp, log

''' STATE MOVEMENTS '''
"""
Each state movement controls how a new trial point is picked based on the current trial point.
"""
class UniformMove:
	def __init__(self, widths = 10, bounds = 0):
		self.widths = widths
		self.bounds = bounds

	def __call__(self, size, T):
		return self.bounds + self.widths*np.random.random(size)

class NeighborhoodMove:
	def __init__(self, center = 0, radius = 10):
		self.center = center
		self.radius = radius

	def __call__(self, size, T):
		rn = np.random.normal(0, 1, size + 2)
		return self.center + self.radius*rn[:-2].reshape(state.shape)/np.linalg.norm(rn)

class GaussianMove:
	def __init__(self, learning = 60):
		self.learning = learning

	def __call__(self, size, T):
		std = min(360/(3*self.learning), sqrt(T))
		return self.learning*np.random.normal(0, std, size)

''' ACCEPTS '''
def standard_accept(dE, T):
	return (dE < 0) or exp(-dE/T) > np.random.rand()


''' THE MEAT 'N' POTATOES '''
def run_at_fixed_T(T, states, Es, best_E0,
				   energy_f, move_f, accept_f,
				   improve_callback = None, opt_callback = None,
				   cur_step = 0, max_steps = 0, kill_thresh = -np.inf,
				   max_accepts = np.inf, start_time = 0, log_writer = None):

	best_E = best_E0

	n_steps = 0
	n_accepts, n_improves = 0, 0

	while (n_steps < max_steps) and (n_accepts < max_accepts):
		try_state = move_f(states[0], T)

		try_E = energy_f(try_state)

 		if log_writer is not None:
			log_writer.write('trial_states', try_state)
			log_writer.write('trial_energies', try_E)

		dE = try_E - Es[0]

		if accept_f(dE, T):
			states[-1] = np.copy(states[0])
			states[0]  = try_state

			Es[-1] = Es[0]
			Es[0]  = try_E

			print("\tCur E: {0:.3f} // Prv E: {1:.3f}".format(Es[0], Es[-1]))

			n_accepts += 1
			if dE < 0:
				n_improves += 1
				if improve_callback is not None:
					self.improve_callback()

			if Es[0] < best_E and opt_callback is not None:
				best_E = Es[0]
				opt_callback(states[0], Es[0], cur_step, log_writer = log_writer)
				if best_E < kill_thresh:
					break

			if log_writer is not None:
				log_writer.write('time_x_dE', timer()-start_time, dE, 1)
				log_writer.write('time_x_E',  timer()-start_time, Es[0], best_E)
		else:
			if log_writer is not None:
				log_writer.write('time_x_dE', timer()-start_time, dE, 0)

		n_steps += 1

	return n_steps, n_accepts, n_improves

class Annealer(object):

	LoggingDefaults = {
		'base_path'		  : '.',
		'use_frozen_sys'  : False,
		'frozen_sys_name' : '',
		'track_time_taken': False,
		'time_taken'	  : 0,
		'track_func_calls': False,
		'func_calls'	  : 0,
		'logs'		  	  : 'none',
		'debug'			  : False
	}

	SystemDefaults = {

		#State settings
		'state_shape'  : (),
		'state0'  	   : None,
		'E0'	  	   : np.inf,
		'best_state'   : None,
		'best_E'	   : np.inf,
		'raw_move'	   : GaussianMove(),
		'move_relative': False,
		'accept'	   : standard_accept,
		'cooling_rate' : -1,

		#Temperature settings
		'T0'			: None,
		'Tf'			: None,
		'T0_accept_rate': 0.95,
		'T0_search_frac': 1.5,
		'Tf_accept_rate': 0.00,
		'Tf_search_frac': 2,
		'T_initial_temp': 100,
		'T_search_iters': 500,

		#iteration settings
		'n_iters'		  : 1000,
		'max_iters_at_T'  : 500,
		'max_accepts_at_T': 100,
		'quench_thresh'   : 50,
		'optimal_thresh'  : -np.inf
	}

	def __init__(self, raw_energy, **kwargs):
		self.raw_energy = raw_energy

		self.setup_system(kwargs)
		if self.state0 is not None:
			self.state_shape = self.state0.shape
		else:
			self.state0 = np.random.random(self.state_shape)

		self.best_upd_step = -1

		if self.logs == 'none':
			self.logs = []
		elif self.logs == 'all':
			self.logs = ['time_x_temp', 'time_x_rate', 'temp_x_rate', 'time_x_E', 'time_x_dE',
				  	     'trial_states', 'trial_energies', 'best_states', 'best_energies']

		arr_types = []
		chunk_sizes = []
		for name in self.logs:
			if name == 'trial_states' or name == 'best_states':
				arr_types.append(True)
				chunk_sizes.append(100)
			else:
				arr_types.append(False)
				chunk_sizes.append(500)

		self.log_writer = None
		#self.log_writer = algo_utils.LogWriter(self.logs, arr_types, chunk_sizes)
		#self.log_writer.open(self.base_path)

	'''
	SET UP PARAMS AND LOGGING
	'''
	def setup_system(self, kwargs):
		for param in self.__class__.LoggingDefaults:
			value = kwargs[param] if param in kwargs else \
					self.__class__.LoggingDefaults[param]
			setattr(self, param, value)

		thawed = self.thaw_system()
		for param in self.__class__.SystemDefaults:
			value = kwargs[param] if param in kwargs else \
					thawed[param] if thawed is not None else \
					self.__class__.SystemDefaults[param]
			setattr(self, param, value)

	def freeze_system(self):
		if not self.use_frozen_sys:
			return

		if len(self.frozen_sys_name) == 0:
			self.frozen_sys_name = '{0}_{1}.state'.format(self.__class__.__name__, \
											 			  algo_utils.gen_time_id())

		fz_path = os.path.join(self.base_path, 'frozen_states')
		if not os.path.exists(fz_path):
			os.makedirs(fz_path)
		frozen_states_log = os.path.join(fz_path, 'frozen_states.log')

		with open(os.path.join(fz_path, self.frozen_sys_name), 'wb') as fzf:
			frozen_state = {}
			for param in self.__class__.SystemDefaults:
				frozen_state[param] = getattr(self, param)

			pickle.dump(frozen_state, fzf)
		with open(frozen_states_log, 'a+') as fsl:
			fsl.write(self.frozen_sys_name + '\n')

	def thaw_system(self):
		if not self.use_frozen_sys:
			return

		if len(self.frozen_sys_name) == 0:
			frozen_states_log = os.path.join(self.base_path, 'frozen_states', 'frozen_states.log')
			try:
				with open(frozen_states_log, 'r') as fsl:
					fsl.seek(-3, 2)
					while fsl.tell() > 1 and fsl.read(1) != '\n':
						fsl.seek(-2, 1)

					if fsl.tell() == 1:
						fsl.seek(0)

					self.frozen_sys_name = fsl.readline().strip('\n')
			except IOError:
				print("Error: Could not thaw system", self.frozen_sys_name, file = sys.stderr)
				return

		fz_path = os.path.join(self.base_path, 'frozen_states', self.frozen_sys_name)
		try:
			with open(fz_path, 'rb') as fzf:
				frozen_state = pickle.load(fzf)
		except IOError:
			return

		return frozen_state

	'''
	SEARCH FOR TEMPERATURE BOUNDS
	'''
	def T0_search(self, move_f, accept_f):
		if self.debug:
		    print('Starting T0 search...', end = " ")


		states = np.ones((2,1))*self.state0
		Es     = np.array([self.E0, self.E0])

		T0_accept_thresh = self.T0_accept_rate*self.T_search_iters
		if self.debug:
		    print('T0 threshold: {}'.format(T0_accept_thresh))

		T0_upper = float(self.T_initial_temp)
		T0 		 = float(self.T_initial_temp)
		T0_lower = float(self.T_initial_temp)

		_, na, _ = run_at_fixed_T(T0_lower, states, Es, self.best_E, self.raw_energy, move_f, self.accept, 	 \
								  max_steps = self.T_search_iters)

		if na < T0_accept_thresh:
			while na < T0_accept_thresh:
				states = np.ones((2,1))*self.state0
				Es     = np.array([self.E0, self.E0])


				T0_lower = T0_upper
				T0_upper *= self.T0_search_frac
				if self.debug:
					print('Accepts = {0:.2f} / T0_lower = {1:.2f} / T0_upper = {2:.2f}'.format(na, T0_lower, T0_upper))
				_, na, _ = run_at_fixed_T(T0_upper, states, Es, self.best_E, self.raw_energy, move_f, accept_f, 	 \
								  		  max_steps = self.T_search_iters)

		else:
			while na > T0_accept_thresh:
				states = np.ones((2,1))*self.state0
				Es     = np.array([self.E0, self.E0])

				T0_upper = T0_lower
				T0_lower /= self.T0_search_frac
				if self.debug:
					print('Accepts = {0:.2f} / T0_lower = {1:.2f} / T0_upper = {2:.2f}'.format(na, T0_lower, T0_upper))
				_, na, _ = run_at_fixed_T(T0_lower, states, Es, self.best_E, self.raw_energy, move_f, accept_f, 	 \
								  max_steps = self.T_search_iters)

		states = np.ones((2,1))*self.state0
		Es     = np.array([self.E0, self.E0])

		T0 = (T0_lower + T0_upper)/2
		_, na, _ = run_at_fixed_T(T0, states, Es, self.best_E, self.raw_energy, move_f, accept_f, 	 \
								  max_steps = self.T_search_iters)
		while na < T0_accept_thresh and (T0_upper - T0) > 1e-3:
			states = np.ones((2,1))*self.state0
			Es     = np.array([self.E0, self.E0])

			if self.debug:
				print('Accepts: {0:.2f} / T0 = {1:.2f}'.format(na, T0))
			T0 = (T0 + T0_upper)/2
			_, na, _ = run_at_fixed_T(T0, states, Es, self.best_E, self.raw_energy, move_f, accept_f, 	 \
								  	  max_steps = self.T_search_iters)
		if self.debug:
			print('T0 found: {}'.format(T0))
		return T0

	def Tf_search(self, move_f, accept_f):
		if self.debug:
			print('Starting Tf search...')

		states = np.ones((2,1))*self.state0
		Es     = np.array([self.E0, self.E0])

		Tf_accept_thresh = self.T_search_iters*self.Tf_accept_rate
		if self.debug:
			print('Tf threshold: {}'.format(Tf_accept_thresh))

		Tf = float(self.T_initial_temp)

		_, na, _ = run_at_fixed_T(Tf, states, Es, self.best_E, self.raw_energy, move_f, accept_f, 	 \
								  max_steps = self.T_search_iters)
		while na > Tf_accept_thresh and Tf > 1e-1:
			state = np.ones((2,1))*self.state0
			E     = np.array([self.E0, self.E0])

			if self.debug:
				print('Accepts = {0:.2f} / Tf = {1:.2f}'.format(na, Tf))
			Tf /= self.Tf_search_frac
			_, na, _ = run_at_fixed_T(Tf, states, Es, self.best_E, self.raw_energy, move_f, accept_f, 	 \
								 	  max_steps = self.T_search_iters)
		if self.debug:
			print('Tf found: {}'.format(Tf))
		return Tf

	def init_T0_Tf(self):
		if self.T0 is None:
			self.T0 = self.T0_search(self.move, self.accept)

		if self.Tf is None:
			self.Tf = self.Tf_search(self.move, self.accept)

		self.freeze_system()

	'''
	RUN THE SYSTEM
	'''

	"""
	Get the current energy
	"""
	def energy(self, state):
		if self.track_func_calls:
			self.func_calls += 1
		return self.raw_energy(state)

	"""
	Move the point. If move_relative is True, the movement is from the previous
	state, otherwise the movement is from the last best state.
	"""
	def move(self, state, T):
		if self.move_relative:
			dx = self.raw_move(state.size, T)
			return np.mod(state + dx, 360)
		else:
			dx = self.raw_move(self.best_state.size, T)
			return np.mod(self.best_state + dx, 360)

	"""
	Determine whether to stop the annealing process: if the number of iterations
	so far is less than n_iters, or the best_energy so far is above the optimal
	threshold, or the number of steps from the last update is less than the
	quench_thresh, continue the process.
	"""
	def continue_annealing(self, step):
		return (step < self.n_iters) and \
			   (self.best_E > self.optimal_thresh) and \
			   (step - self.best_upd_step < self.quench_thresh)

	def update_best_energy(self, cur_state, cur_E, cur_step, log_writer = None):
		if self.debug and self.best_state is not None:
			print('New opt found: step: {0} E: {1:.5f}'.format(cur_step, cur_E))

		if log_writer is not None:
			log_writer.write('best_states', cur_state)
			log_writer.write('best_energies', cur_E)

		self.best_upd_step = cur_step
		self.best_state = np.copy(cur_state)
		self.best_E = cur_E

		self.freeze_system()

	def anneal(self):
		states = np.ones((2,1))*self.state0
		Es     = np.array([self.E0, self.E0])
		T = self.T0

		s = 0
		while self.continue_annealing(s):
			n_steps, n_accepts, n_improves = run_at_fixed_T(T, states, Es, self.best_E, self.energy, self.move, self.accept, \
								  	  						opt_callback = self.update_best_energy, cur_step = s, kill_thresh = self.optimal_thresh, \
								  	  						max_steps = self.max_iters_at_T, max_accepts = self.max_accepts_at_T, \
								  	  						start_time = self.start_time, log_writer = self.log_writer)

			frac_accepted = float(n_accepts)/n_steps
			frac_improved = float(n_improves)/n_steps

			delta_t = timer() - self.start_time
			if self.log_writer is not None:
				self.log_writer.write('time_x_temp', delta_t, T)
				self.log_writer.write('time_x_rate', delta_t, frac_accepted, frac_improved)
				self.log_writer.write('temp_x_rate', T, frac_accepted, frac_improved)

			self.print_run_result(s, T, n_steps, frac_accepted, frac_improved)

			T *= self.cooling_rate
			s += 1

	def run(self):
		self.E0 = self.raw_energy(self.state0)
		self.update_best_energy(self.state0, self.E0, 0)

		self.init_T0_Tf()
		iterated_cooling_rate = exp(log(self.Tf/self.T0)/self.n_iters)
		self.cooling_rate =  iterated_cooling_rate if self.cooling_rate < 0 \
												   else self.cooling_rate

		self.start_time = timer()
		self.anneal()

		if self.log_writer is not None:
			self.log_writer.flush_all()

		if self.track_time_taken:
			self.time_taken = timer() - self.start_time

		return self.best_state, self.best_E


	def print_run_result(self, step, T, n_steps, frac_accepted, frac_improved):
		print('step: {0} / temp: {1:.2f}'.format(step, T), end = " ")
		print('nsteps: {0:.2f} / ar: {1:.2f} / ar: {2:.2f}'.format(n_steps, frac_accepted, frac_improved))



''' TREE ANNEALING '''

"""
Tree Annealing builds a tree where each node in the tree represents a subspace
to search within. Each new point is gotten by building a random walk down the tree
and choosing a random point within the subspace represented by the end of the walk.

A probability distribution over each depth of the tree governs how the random walk
is generated. The distribution is initially uniform. Each time a new best point is
found in a subspace at the leaves of the tree, the weight of that leaf is increased, and
that change is propagated up the tree to the root.

SpatialNode represents a subspace represented by an axis aligned box. The subspace
represented by each SpatialNode is contained within the subspace represented by its
parent, except the root which represents the whole space.

gen_walk -- generates a random walk down the tree of spatial nodes. Returns the last node of the
			walk, along with the walk itself.

upd_hist -- update the probability distribution governing the walks down the tree,
			as described above.

upd_rect -- compute the subspace given by the walk down the tree.
"""

class SpatialNode:

	def __init__(self, parent = None, depth = 0):

		self.parent = parent
		self.depth = depth
		self.children = []
		self.isa_leaf = True

	def split(self, n):
		self.isa_leaf = False
		for k in xrange(n):
			self.children.append(SpatialNode(parent = self, depth = self.depth + 1))

def gen_walk(root, n_splits, tree_prob):
	walk = []

	node = root
	child_choices = np.arange(n_splits)
	while not node.isa_leaf:
		step = np.random.choice(child_choices, p = tree_prob)

		walk.append(step)
		node = node.children[step]

	return node, walk

def upd_hist(hist, walk):
	for step in walk:
		hist[step] += 1
	return hist

def upd_rect(size, bounds, widths, walk, n_splits_at_d):
	dim = 0
	for step in walk:
		widths[dim] /= n_splits_at_d
		bounds[dim] += step*widths[dim]

		dim = (dim + 1) % size

	return bounds, widths

def calc_node_prob(walk, tree_prob, widths):
	n_widths = widths / 360
	return np.prod(tree_prob[walk])/np.prod(n_widths)


class TreeAnnealer(Annealer):
	SystemDefaults = {
		#Tree settings
		'n_splits_at_d' 	: 2,
		'min_tree_width'	: 1,
		'explore_tree_depth': 10,
		'min_tree_depth'	: 2,
		'max_tree_depth'	: 160,
		'bounds0' 			: np.zeros(16),
		'widths0'			: np.ones(16)*360,

		#State settings
		'n_iters' : np.inf,
		'raw_move': UniformMove()
	}
	SystemDefaults.update(dict([(k, v) for (k,v) in Annealer.SystemDefaults.items() \
												 if k not in SystemDefaults]))

	def __init__(self, raw_energy, **kwargs):
		self.tree_root = SpatialNode()
		self.tree_node = self.tree_root

		self.tree_walk = [0]
		self.depth = 0

		super(TreeAnnealer, self).__init__(raw_energy, **kwargs)
		self.raw_move = UniformMove()

		self.tree_hist = np.ones(self.n_splits_at_d)
		self.tree_prob = self.tree_hist/np.sum(self.tree_hist)
		self.node_probs = [float(1), float(1)]

	def init_T0_Tf(self):


		if self.T0 is None:
			self.T0 = self.T0_search(lambda state, T: self.raw_move(state.size, T), self.accept)

		if self.Tf is None:
			self.Tf = self.Tf_search(lambda state, T: self.raw_move(state.size, T), self.accept)

		self.freeze_system()

	def move(self, state, T):
		self.tree_node, self.tree_walk = gen_walk(self.tree_root, self.n_splits_at_d, \
												  self.tree_prob)

		bounds, widths = upd_rect(state.size, self.bounds0, self.widths0, self.tree_walk, self.n_splits_at_d)
		self.raw_move.widths = widths
		self.raw_move.bounds = bounds

		self.node_probs[0] = calc_node_prob(self.tree_walk, self.tree_prob, widths)

		return self.raw_move(state.size, T)

	def accept(self, dE, T):
		if dE < 0:
			self.node_probs[-1] = self.node_probs[0]
			return True
		elif self.node_probs[0]/self.node_probs[-1]*exp(-dE/T) > np.random.rand():
			self.node_probs[-1] = self.node_probs[0]
			return True
		else:
			return False

	def update_best_energy(self, cur_state, cur_E, cur_step, log_writer = None):
		if self.depth < self.max_tree_depth:
			self.tree_node.split(self.n_splits_at_d)
			self.depth = self.tree_node.depth

		#if self.depth > self.explore_tree_depth:
		#self.tree_hist = upd_hist(self.tree_hist, self.tree_walk)
		#self.tree_prob = self.tree_hist / np.sum(self.tree_hist)

		super(TreeAnnealer, self).update_best_energy(cur_state, cur_E, cur_step, log_writer)

		if self.debug:
			print('\tTree depth: {}'.format(self.depth))
			print('\tTree hist: \t' + \
				'\n\t           \t'.join('{0}: {1}'.format(i, self.tree_prob[i])
													       for i in xrange(self.n_splits_at_d)))
