from __future__ import print_function

import os
import sys
import numpy as np
import time

from collections import namedtuple

sys.path.append(os.path.realpath(".."))
from firmware import basestation_connection, get_energy
from algorithms import beam_sweep, sim_anneal, grad_desc

Thresholds = namedtuple(\
    "Thresholds", "lock_thresh_hi lock_thresh_lo re_find_thresh"
)

"""
First potentially runs a tree-annealing algorithm to get a best guess for a
first state. Then runs a "vanilla" annealing algorithm, using a Gaussian step
with progressively increasing variance if a "good enough" state is not found.
Finally runs an optional final gradient descent step.
"""

def run_annealing_optimization(energy_f, cur_state, cur_energy, thresholds,
                               final_gradient_step = False, debug = False):

    """
    If the current energy is above half the goal, discard the current state and
    instead search for a starting point using tree annealing
    """
    if cur_energy > 0.5*thresholds.lock_thresh_hi:
        tsa = sim_anneal.TreeAnnealer(energy_f, frozen_sys_name = "tree",
                                      state0 = cur_state, n_iters = 500,
                                      optimal_thresh = thresholds.lock_thresh_hi,
                                      quench_thresh = 50, use_frozen_sys = True,
                                      debug = debug)

        sa_state, sa_energy = tsa.run()
        if sa_energy < cur_energy:
            sa_energy = cur_energy
            sa_state = cur_state

    """
    Run a simulated annealing algorithm... if there are no update within the
    steps given by `quench_thresh`, kill it and increase the step size to search
    a larger area. If a good enough state is found, stop the process
    """

    vsa = sim_anneal.Annealer(energy_f, move_relative = True,
                              n_iters = 500, quench_thresh = 50,
                              optimal_thresh = thresholds.lock_thresh_hi,
                              use_frozen_sys = True, debug = debug)

    for l_param in [0.5, 2, 5, 10]:

        vsa.state0 = cur_state
        vsa.raw_move = sim_anneal.GaussianMove(learning = l_param)
        sa_state, sa_energy = vsa.run()

        if sa_energy < cur_energy:
            cur_energy = sa_energy
            cur_state = sa_state

        if cur_energy > 0.8*thresholds.lock_thresh_hi:
            break

    if final_gradient_step:
        gd = grad_desc.GradientDescent(energy_f, grad_desc.DiscreteGradient(energy_f), \
									   state_or_shape = best_state, debug = debug)

        gd_state, gd_energy = gd.run()
        if gd_energy > cur_energy:
            cur_state = gd_state
            cur_energy = gd_energy

    return cur_state, cur_energy

"""
Beam-based optimization. Run a beamsweep to find the best beamform solution,
possibly incorporating previous information for a more informed search.
"""

def steps_from_bounds(bounds):
    if bounds is None:
        return (13, 11)
    else:
        w = bounds[1] - bounds[0]
        coarse = min(w, 13)
        fine = min(2*coarse, 11)
        return (coarse, fine)

def run_beam_optimization(energy_f, thresholds, prv_beam, debug = False, logfile = sys.stdout):
    best_energy = np.inf

    """
    If we have previous beam data, search first in a small area around that
    previous beam, under the assumption that the phone has significantly
    changed its position.
    """
    if prv_beam is not None:
        boundsX = [prv_beam[0] - 10, prv_beam[0] + 10]
        boundsY = [prv_beam[1] - 10, prv_beam[1] + 10]
    else:
        boundsX, boundsY = None, None

    stepsX = steps_from_bounds(boundsX)
    stepsY = steps_from_bounds(boundsY)

    """
    Run sweeps, first theta-phi, then phi-theta
    """

    print("RUNNING BS X")
    print("BOUNDS: ", boundsX, boundsY, file = logfile)
    beamX, beam_energX = beam_sweep.twoD_beam_sweep(energy_f, stepsX,
                                                     thresh = thresholds.lock_thresh_hi,
                                                     delays = (0.02, 0.02), axis = 0,
                                                     bounds = (boundsX, boundsY),
                                                     debug = debug)

    print("BEAM X: ", beamX, file = logfile)
    time.sleep(0.02)
    cur_energy = energy_f(beam_sweep.beam_angle_to_phases(*beamX))
    print("ENERGY: ", cur_energy, file = logfile)

    if cur_energy < best_energy:
        best_state = beam_sweep.beam_angle_to_phases(*beamX)
        best_energy = cur_energy

    if best_energy < thresholds.lock_thresh_hi:
        return beamX, best_state, best_energy

    print("RUNNING BS Y")
    beamY, beam_energY = beam_sweep.twoD_beam_sweep(energy_f, stepsY,
                                                    thresh = thresholds.lock_thresh_hi,
                                                    delays = (0.02, 0.02), axis = 1,
                                                    bounds = (boundsY, boundsX),
                                                    debug = debug)

    print("BEAM Y: ", beamY, file= logfile)
    time.sleep(0.02)
    cur_energy = energy_f(beam_sweep.beam_angle_to_phases(*beamY))
    print("ENERGY:", cur_energy, file = logfile)

    if cur_energy < best_energy:
        best_state = beam_sweep.beam_angle_to_phases(*beamY)
        best_energy = cur_energy

    if best_energy < thresholds.lock_thresh_hi:
        return beamY, best_state, best_energy

    """
    If we haven't found a beam with good enough energy, return `None` as the
    found beam.
    """
    return None, best_state, best_energy

"""
The main loop which ensures the phone is always either found or being searched for
"""
def main_loop(energy_f, run_annealing = False, use_gradient_step = False, debug = False, logfile = sys.stdout):
    cur_state = None
    cur_energy = np.inf

    thresholds = Thresholds(-500, -375, -0x01)
    poll_time  = 0.1

    beam = None

    poll_count = 0
    low2_count = 0
    window_size = 10

    while True:
        while cur_state is None or \
              cur_energy > thresholds.re_find_thresh:

            beam, cur_state, cur_energy = run_beam_optimization(energy_f,
                                                                  thresholds, beam,
                                                                  debug = debug)

            if beam is None:
                beam, cur_state, cur_energy = run_beam_optimization(energy_f,
                                                                    thresholds, beam,
                                                                    debug = debug)

            if beam is None and run_annealing:
                run_annealing_optimization(energy_f, cur_state, cur_energy, thresholds,
                                           final_gradient_step = use_gradient_step,
                                           debug = debug)

        """
        If we're below the the low phone threshold, continue the loop to trigger
        the optimization step again
        """
        if cur_energy > thresholds.lock_thresh_lo:
            cur_state = None
            continue

        """
        Otherwise, start the polling process
        """
        cur_energy = energy_f(cur_state)
        poll_count += 1

        if cur_energy > thresholds.lock_thresh_lo:
            low2_count += 1
            print("POLL: ", poll_count, " LO2: ", low2_count, file = sys.stdout)

        if poll_count > 2 and float(low2_count)/poll_count > 0.75:
            low2_count = 0
            poll_count = 0
            cur_state = None

        print("energy: ", cur_energy, " polls: ", poll_count, " lows: ", low2_count, file = sys.stdout)
        if poll_count == window_size:
            poll_count = 0
            low2_count = 0
        time.sleep(poll_time/4)

if __name__ == "__main__":
    args = sys.argv[1:]

    should_log = False
    if "log" is args:
        should_log = True

    should_anneal = False
    if "anneal" in args:
        should_anneal = True

    with basestation_connection() as bs_connect:
        energy_f = get_energy(bs_connect)
        try:
            if should_log:
                with open("run_main.log") as log_fh:
                    main_loop(energy_f,run_annealing = should_anneal,
                              debug = True, logfile = log_fh)
            else:
                main_loop(energy_f, run_annealing = should_anneal, debug = True)
        except KeyboardInterrupt:
            pass
