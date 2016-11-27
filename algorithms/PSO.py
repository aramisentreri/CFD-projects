import numpy as np

"""
An implementation of basic particle swarm optimization.

v_c -- the weight of the previous direction
locv_c -- the weight of the direction towards a given particle's best point
glbv_c -- the weight of the direction towards the current best point among the
          particles

max_iters, min_dX, min_dE -- thresholds to stop iteration, either the maximum
                             number of iterations, the minimum change in best
                             position, or the minimum change in best energy

trigger_restart_or_kill -- the number of iterations without update after which
                           to either restart or stop the process.
restart_not_kill -- restart instead of stopping the process
"""

def pso_vanilla(energy_f, bounds = None,
                swarm_count = 10, starting_x = None, starting_v = None,
                v_c = 1/np.sqrt(2), locv_c = np.sqrt(2), glbv_c = np.sqrt(2),
                max_iters = 100, min_dX = 1e-2, min_dE = 1e-3,
                trigger_restart_or_kill = None, restart_not_kill = False,
                return_swarm = False, debug = False):

    if trigger_restart_or_kill is None:
        trigger_restart_or_kill = float(max_iters)/3

    if bounds is None:
        lb = np.zeros(state_dims)
        ub = np.ones(state_dims)*360
    else:
        lb, ub = bounds
        lb = np.array(lb)
        ub = np.array(ub)
    spread = ub - lb

    if starting_x is None:
        starting_x = lb + spread*np.random.random((swarm_count, state_dims))

    x = starting_x
    E = np.empty(swarm_count)

    if starting_v is None:
        v = spread/4*(-1 + 2*np.random.random((swarm_count, state_dims)))
    else:
        v = starting_v

    opt_x = np.empty((swarm_count, state_dims))
    opt_E = np.ones(swarm_count)*np.inf

    g_opt_x = None
    g_opt_E = np.inf

    it = 0
    last_update = -1
    while it < max_iters:

        # update energy
        for i in xrange(swarm_count):
            E[i] = energy_func(x[i])

        #update optima
        improved = E < opt_E
        opt_x[improved] = x[improved].copy()
        opt_E[improved] = E[improved]

        g_improved = opt_E.argmin()
        if opt_E[g_improved] < g_opt_E:
            if g_opt_x is not None:
                dE = g_opt_E - opt_E[g_improved]
                dx = utils.AP_distance(g_opt_x, opt_x[g_improved])
            else:
                dE = np.inf
                dx = np.inf

            last_update = it

            if debug:
                print """NEW OPT FOUND @ iter {0:}: {1:} with energy {2:}
                            delta X = {3:} -- delta E = {4:}""".format(\
                            it, opt_x[g_improved], opt_E[g_improved], dx, dE)

            if dx < min_dX:
                if debug:
                    print "Minimal change in optimal position -- stopping"
                break

            elif dE < min_dE:
                if debug:
                    print "Minimal change in energy value -- stopping"
                break
            else:
                g_opt_x = opt_x[g_improved].copy()
                g_opt_E = opt_E[g_improved]

        loc_weight = locv_c*np.random.random((swarm_count, state_dims))
        glb_weight = glbv_c*np.random.random((swarm_count, state_dims))

        v = v_c*v + loc_weight*(x - opt_x) + glb_weight*(x - g_opt_x)
        x += v

        out_of_bounds = np.logical_or(x < lb, x > ub)
        x *= (~out_of_bounds)
        x += out_of_bounds*(lb + np.random.random()*spread)

        #update iteration
        if debug:
            print "Best at iter {0:} is {1:} with energy {2:}".format(\
                it, g_opt_x, g_opt_E)

        if it - last_update > trigger_restart_or_kill:
            if restart_not_kill:
                x = starting_x
            else:
                break

        it += 1

    if return_swarm:
        return it, g_opt_x, g_opt_E, opt_x, opt_E
    else:
        return it, g_opt_x, g_opt_E
