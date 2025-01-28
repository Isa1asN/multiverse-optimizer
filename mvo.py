import random
import numpy as np
from sklearn.preprocessing import normalize
from solution import Solution
import time
import math

class MultiverseOptimizer:
    def __init__(self, obj_func, lb, ub, dim, num_universes, max_iter, 
                 wep_max=1.0, wep_min=0.2):
        self.obj_func = obj_func
        self.lb = [lb] * dim if not isinstance(lb, list) else lb
        self.ub = [ub] * dim if not isinstance(ub, list) else ub
        self.dim = dim
        self.num_universes = num_universes
        self.max_iter = max_iter
        self.wep_max = wep_max
        self.wep_min = wep_min
        
        self.solution = Solution()
        
    def _normr(self, Mat):
        """Normalize the columns of the matrix"""
        try:
            Mat = Mat.reshape(1, -1)
            if Mat.dtype != "float":
                Mat = np.asarray(Mat, dtype=float)
            
            B = normalize(Mat, norm="l2", axis=1)
            B = np.reshape(B, -1)
            return B
        except Exception as e:
            print(f"Error in normalization: {str(e)}")
            return np.ones(Mat.shape[1]) / Mat.shape[1]
    
    def _roulette_wheel_selection(self, weights):
        """Roulette wheel selection implementation"""
        accumulation = np.cumsum(weights)
        p = random.random() * accumulation[-1]
        chosen_index = -1
        for index in range(0, len(accumulation)):
            if accumulation[index] > p:
                chosen_index = index
                break
        return chosen_index if chosen_index != -1 else 0

    def optimize(self):
        # Initialize universes
        universes = np.zeros((self.num_universes, self.dim))
        for i in range(self.dim):
            universes[:, i] = np.random.uniform(0, 1, self.num_universes) * (self.ub[i] - self.lb[i]) + self.lb[i]
        
        sorted_universes = np.copy(universes)
        convergence = np.zeros(self.max_iter)
        
        best_universe = np.zeros(self.dim)
        best_universe_inflation_rate = float("inf")
        
        timer_start = time.time()
        self.solution.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        
        print(f'MVO is optimizing "{self.obj_func.__name__}"')
        
        # Main loop
        for iteration in range(1, self.max_iter + 1):
            wep = self.wep_min + iteration * ((self.wep_max - self.wep_min) / self.max_iter)
            tdr = 1 - (math.pow(iteration, 1/6) / math.pow(self.max_iter, 1/6))
            
            inflation_rates = []
            for i in range(self.num_universes):
                universes[i] = np.clip(universes[i], self.lb, self.ub)
                inflation_rates.append(self.obj_func(universes[i]))
                
                if inflation_rates[i] < best_universe_inflation_rate:
                    best_universe_inflation_rate = inflation_rates[i]
                    best_universe = np.array(universes[i])
            
            # Sort universes
            sorted_indexes = np.argsort(inflation_rates)
            sorted_inflation_rates = np.sort(inflation_rates)
            
            for i in range(self.num_universes):
                sorted_universes[i] = universes[sorted_indexes[i]]
            
            universes[0] = sorted_universes[0]
            
            normalized_sorted_inflation_rates = self._normr(sorted_inflation_rates)
            
            for i in range(1, self.num_universes):
                for j in range(self.dim):
                    r1 = random.random()
                    
                    # White hole operator
                    if r1 < normalized_sorted_inflation_rates[i]:
                        white_hole_idx = self._roulette_wheel_selection(-np.array(sorted_inflation_rates))
                        if white_hole_idx == -1:
                            white_hole_idx = 0
                        universes[i, j] = sorted_universes[white_hole_idx, j]
                    
                    # Wormhole operator
                    r2 = random.random()
                    if r2 < wep:
                        r3 = random.random()
                        if r3 < 0.5:
                            universes[i, j] = best_universe[j] + tdr * ((self.ub[j] - self.lb[j]) * random.random() + self.lb[j])
                        else:
                            universes[i, j] = best_universe[j] - tdr * ((self.ub[j] - self.lb[j]) * random.random() + self.lb[j])
            
            convergence[iteration - 1] = best_universe_inflation_rate
            if iteration % 10 == 0:
                print(f"At iteration {iteration} the best fitness (lowest loss) is {best_universe_inflation_rate}")
        
        # Save results
        timer_end = time.time()
        self.solution.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.solution.executionTime = timer_end - timer_start
        self.solution.convergence = convergence
        self.solution.optimizer = "MVO"
        self.solution.bestIndividual = best_universe
        self.solution.objfname = self.obj_func.__name__
        
        return self.solution