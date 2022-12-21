from collections import defaultdict
import numpy as np
from scipy.stats import uniform

from surpyval.renewal import NonParametricCounting

class GeneralizedRenewal():
    def __init__(self, model):
        self.model = model

    def simulate_over_interval(self, T, q, N=100_000, tol=1e-5, return_xi=False):
        running = 0
        # x = model.random(2 * N).tolist()
        U = uniform.rvs(size=100_000).tolist()
        aggregate_timeline = defaultdict(lambda: 0)
        aggregate_timeline[0] = 0
        # aggregate_timeline[T] = 0
        max_life = defaultdict(lambda: 0)
        counted = 0
        alpha, beta = self.model.params
        life = np.copy(alpha)

        xicn = {
            'x' : [],
            'i' : []
        }

        # Need to do N simulations to 2 x T
        # pbar = tqdm.tqdm(total=N)
        while counted < N:
            try:
                ui = U.pop()
            except:
                U = uniform.rvs(size=100_000).tolist()
                ui = U.pop()
        
            new_life = life * q
            xi = self.model.dist.qf(ui, life, beta)
            running += xi

            
            delta = new_life / alpha
            if running > T:
                running = 0
                counted += 1
                life = np.copy(alpha)
                max_life[T] += 1
            elif delta < tol:
                max_life[running] += 1
                running = 0
                counted += 1
                life = np.copy(alpha)
            else:
                xicn['x'].append(running)
                xicn['i'].append(counted)
                life = new_life
                aggregate_timeline[running] += 1
        
        if return_xi:
            return xicn['x'], xicn['i']

        at_risk = np.array(list(max_life.items()))
        at_risk = at_risk[at_risk[:, 0].argsort()]
        at_risk[:, 1] = N - at_risk[:, 1].cumsum() + at_risk[:, 1]

        timeline_arr = np.array(list(aggregate_timeline.items()))
        timeline_arr = timeline_arr[timeline_arr[:, 0].argsort()]

        idx = np.searchsorted(at_risk[:, 0], timeline_arr[:, 0], side='right')
        timeline = timeline_arr[:, 0]
        hazard_rate = timeline_arr[:, 1] / at_risk[idx, 1]
        cum_hazard_rate = hazard_rate.cumsum()

        expected_events = lambda x: np.interp(x, timeline, cum_hazard_rate, left=0, right=np.nan)
        self._expected_events = expected_events

    def reset(self):
        del self._expected_events

    def expected_events(self, x):
        return self._expected_events(x)

    # @classmethod
    # def fit(self, x, i, c, n, dist=Weibull):
        # np_count_model = NonParametricCounting.fit(x, i, c, n)
        # groupby i, find min x
        # if len == 1 then init=[x, 1.]
        # create function that takes parameters to generate MC curve estimate
        # within the generation:
        ## keep a list of all but the life parameter
        ## use either KI or KII logic to modify the life parameter
        ## generate next xi
        # create np.interp function
        # return (func(np_count_model.x) - np_count_model.mcf_hat)**2
        # use minimize to solve


        # Might need to solve for 1 in [0, 1) and [1, inf)