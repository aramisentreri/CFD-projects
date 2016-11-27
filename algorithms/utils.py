import os, numpy as np
from calendar import timegm
from datetime import datetime

AP_space_bounds = [0, 360]
IQ_space_bounds = [0, 1]

def feasible(state, bounds):
	return np.all(state < bounds[0]) and np.all(state > bounds[1])

def randomstate(dims, bounds):
	rn = np.random.random(size = dims)
	return bounds[0] + (bounds[1] - bounds[0])*rn

''' IQ <-> PHASE TRANSFORMATIONS '''
def fixed_amplitude_AP_2_IQ(phs_state, amplitude = 1):
	return amplitude*np.exp(1j*np.radians(phs_state))

def varied_amplitude_AP_2_IQ(ap_state):
	return ap_state[...,0]*np.exp(1j*np.radians(ap_state[...,1]))

def AP_2_IQ(ap_state):
	if ap_state.shape[-1] == 2:
		return varied_amplitude_AP_2_IQ(ap_state)
	else:
		return fixed_amplitude_AP_2_IQ(ap_state)

def fixed_amplitude_IQ_2_AP(iq_state, amplitude = 1):
	theta = np.mod(np.angle(iq_state, deg = True), 360)
	return theta

def varied_amplitude_IQ_2_AP(iq_state):
	mod = np.linalg.norm(iq_state)
	ang = np.mod(np.angle(iq_state, deg = True), 360)

	return np.dstack((mod, ang))

def IQ_2_AP(iq_state):
	if iq_state.shape[-1] == 2:
		return varied_amplitude_IQ_2_AP(iq_state)
	else:
		return fixed_amplitude_IQ_2_AP(iq_state)

def maybe_outer_subtract(p, q):
	if len(p.shape) > 1 and len(q.shape) > 1:
		return np.subtract.outer(p, q)
	else:
		return np.atleast_2d(p - q)

def IQ_distance(p, q):
	d = maybe_outer_subtract(p, q)
	return np.linalg.norm(d, axis = 1)

def AP_distance(p, q):
	d = maybe_outer_subtract(p, q)
	d_min = np.minimum(d, d + 360, d - 360)
	return np.linalg.norm(d, axis = 1)
'''

LOGGING AND TRACKING
'''
def gen_time_id():
	now = datetime.now()
	return timegm(now.timetuple()) + now.microsecond

class LogInfo:
	def __init__(self, data_type, chunk_size):

		self.arr_type = None
		if isinstance(data_type, tuple) and data_type[0] == np.ndarray:
			self.data_type, self.arr_type = data_type
		else:
			self.data_type = data_type

		if data_type is np.ndarray:
			self.save_as_arr = True
		elif data_type is int or np.issubdtype(data_type, int):
			self.save_as_arr = True
			self.arr_type = np.int
		elif data_type is float or np.issubdtype(data_type, float):
			self.save_as_arr = True
			self.arr_type = np.float

		self.deduce_save_fmt()

		self.chunk_size = chunk_size
		self.chunk_idx = 0

	def deduce_save_fmt(self):
		if np.issubdtype(self.arr_type, int):
			self.save_fmt = "%i"
		elif np.issubdtype(self.arr_type, float):
			self.save_fmt = "%.4e"
		else:
			self.save_fmt = None


class LogWriter:
	def __init__(self, logs_info, base_path = "."):
		self.logs_fd = os.path.join(base_path, "logs")

		self.log_files = {}

		self.logs_info_map = dict((info[0], LogInfo(*info[1:])) for info in logs_info)
		self.chunks = {}

		self.write_counter = dict((info[0], 0) for info in logs_info)

	def __enter__(self):
		self.t_id = gen_time_id()

		if not os.path.isdir(self.logs_fd):
			os.makedirs(self.logs_fd)

		for name, fn in self.get_file_names().items():
			self.log_files[name] = open(fn, "a+")

	def get_file_names(self):
		return dict((name, os.path.join(self.logs_fd, "%s_%d.log" % (name, self.t_id))) \
					for name in self.logs_info_map)

	def _write_file(self, name, cutoff):
		log_info = self.logs_info_map.get(name)
		chunk = self.chunks.get(name)

		if log_info is not None and chunk is not None:
			self.write_counter[name] += cutoff

			if log_info.save_as_arr:
				np.savetxt(self.log_files[name], chunk[:cutoff], fmt = log_info.save_fmt)
			else:
				self.log_files[name].writelines(chunk[:cutoff])

	def _flush_all(self):
		for name in self.logs_info_map:
			info = self.logs_info_map[name]
			self._write_file(name, info.chunk_idx)

	def write(self, name, *data):
		if name in self.logs_info_map:
			info = self.logs_info_map[name]

			if name in self.chunks:
				chunk = self.chunks[name]
			else:
				if info.save_as_arr:
					if info.data_type is np.ndarray:
						chunk = np.ndarray((info.chunk_size,) + data[0].shape, dtype = info.arr_type)
					else:
						chunk = np.ndarray((info.chunk_size, len(data)), dtype = info.arr_type)
				else:
					chunk = [""]*info.chunk_size
				self.chunks[name] = chunk

			idx = info.chunk_idx
			if info.data_type is np.ndarray:
				chunk[idx] = data[0]
			elif info.save_as_arr:
				chunk[idx] = data
			else:
				chunk[idx] = ",".join(map(str, data)) + "\n"

			idx = (idx + 1) % info.chunk_size
			if idx == 0:
				self._write_file(name, info.chunk_size)
			info.chunk_idx = idx

	def __exit__(self):
		for name in self.logs_info_map:
			self.write_counter[name] = 0
		self._flush_all()
		for name, fh in self.log_files.items():
			fh.close()
