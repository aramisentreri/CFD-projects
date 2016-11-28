from firmware import n_antennas, nx_antennas, ny_antennas
import time
import numpy as np

min_beam_angle = -30
max_beam_angle = 30

beam_angle_map = [0, 6, 12, 18, 24, 30, 36, 41, 47, 53, 59, 65, 71, 76, 82, 88, 94, 99, 105, 111, 116, 122, 127, \
				  133, 138, 144, 149, 154, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 209, 214, 219, 223, \
                  228, 232, 236, 240, 245, 249, 253, 257, 260, 264, 268, 272, 275, 279, 282, 285, 288, 291, 294]

"""
Mapping a beam angle `ba` to the beam phase offset used by beam_angle_to_phases.
"""
def frac_beam_angle(ba):
	lb = int(ba)
	ub = lb + 1

	return (ba - lb)*beam_angle_map[ub] + \
		   (ub - ba)*beam_angle_map[lb]

"""
Get the phases corresponding to the beam at angle (theta, phi)
If a = frac_beam_angle(theta) and b = frac_beam_angle(phi), then the antenna
at x-y position (i, j) is given phase i*a + j*b
"""
def beam_angle_to_phases(theta, phi):
	phases = np.zeros(n_antennas, dtype = 'float')

	for y in xrange(ny_antennas):
		yM = ny_antennas - y - 1 if phi < 0 else y

		for x in xrange(nx_antennas):
			xM = nx_antennas - x - 1 if theta < 0 else x

			phase = xM*frac_beam_angle(abs(theta)) + \
					yM*frac_beam_angle(abs(phi))
			phases[x + ny_antennas*y] = phase % 360

	return phases


"""
Set the basestation to transmit a certain single beam (theta, phi)
"""
def fixed_beam(energy_f, theta, phi):
	phases = beam_angle_to_phases(theta,phi)
	energy_f(phases)


"""

"""
def beam_sweep_tester(energy_f, n_steps, bounds = (min_beam_angle, max_beam_angle), delay = 0.01):
	lower, upper = bounds

	if type(n_steps) == int:
		steps_x, steps_y = n_steps, 1
	elif type(n_steps) == tuple:
		steps_x, steps_y = n_steps

	x_mesh = [0] if steps_x == 1 else np.linspace(lower, upper, steps_x)
	y_mesh = [0] if steps_y == 1 else np.linspace(lower, upper, steps_y)

	for x in xrange(steps_x):
		for y in xrange(steps_y):
			phases = beam_angle_to_phases(x_mesh[x], y_mesh[y])
			val = energy_f(phases)

			print 'theta: {0} / phi: {1}'.format(x_mesh[x], y_mesh[y])
			print "val: ", val
			if delay > 0:
				time.sleep(delay)

	return x_mesh, y_mesh

"""
Runs a beam sweem in one dimension.
"""
def oneD_beam_sweep(energy_f, steps, thresh = -np.inf, delays = (0, 0), axis = 0, \
		    		angle0 = None, E0 = np.inf, bounds = None, system_delay = 0,
                    debug = False):

    if bounds is None:
        bounds = [min_beam_angle, max_beam_angle]

    if angle0 is None:
        angle0 = [0,0]

    coarse, fine = steps
    delay_coarse, delay_fine = delays

    min_angle = angle0[axis]
    min_E = E0

    beam = angle0
    lower = bounds[0]
    upper = bounds[1]

    mesh, coarse_d = np.linspace(lower, upper, coarse, retstep = True)
    lower = max(min_angle - coarse_d, min_beam_angle)
    upper = min(min_angle + coarse_d, max_beam_angle)
    for angle in mesh:
        time.sleep(delay_coarse)
        beam[axis] = angle

        phases = beam_angle_to_phases(*beam)
        E = energy_f(phases)

        if debug:
            print 'theta = ' + str(beam[0]),
            print 'phi   = ' + str(beam[1]),
            print 'E  = ' + str(E)

        if E < min_E:
            min_E = E
            min_angle = angle

        if min_E < thresh:
            break

    if debug:
        print 'coarse min: ' + str(min_angle),
        print 'E = ' + str(min_E)

    beam[axis] = min_angle - system_delay*coarse_d
    if min_E < thresh or fine == 1:
        return beam, min_E

    lower = max(min_angle - coarse_d, min_beam_angle)
    upper = min(min_angle + coarse_d, max_beam_angle)

    beam = angle0

    min_E = None
    mesh, fine_d = np.linspace(lower, upper, fine, retstep = True)
    for angle in mesh:
        beam[axis] = angle

        phases = beam_angle_to_phases(*beam)
        E = energy_f(phases)
        time.sleep(delay_fine)

        if debug:
            print 'theta = ' + str(beam[0]),
            print 'phi   = ' + str(beam[1]),
            print 'rssi  = ' + str(E)

        if min_E is None or E < min_E:
            min_E = E
            min_angle = angle

        if min_E < thresh:
            beam[axis] = min_angle
            return beam, min_E

    if debug:
        print 'fine min: ' + str(min_angle),
        print 'E = ' + str(min_E)

    beam[axis] = min_angle - system_delay*fine_d
    return beam, min_E

"""
"""
def twoD_beam_sweep(energy_f, steps, thresh = -np.inf, delays = (0, 0), axis = 0, \
		    angle0 = None, E0 = np.inf, bounds = (None, None), system_delay = 0, debug = False):

    bounds0, bounds1 = bounds
    oneD_beam, oneD_min_E = oneD_beam_sweep(energy_f, steps, thresh = thresh, \
                                            delays = delays, axis = axis, \
                                            angle0 = angle0, E0 = E0, bounds = bounds0, \
                                            system_delay = system_delay, debug = debug)

    if oneD_min_E < thresh:
        return oneD_beam, oneD_min_E

    twoD_beam, twoD_min_E = oneD_beam_sweep(energy_f, steps, thresh = thresh, \
                                            delays = delays, axis = 1 - axis, \
                                            angle0 = oneD_beam, E0 = E0, bounds = bounds1, \
                                            system_delay = system_delay, debug = debug)

    return twoD_beam, twoD_min_E
