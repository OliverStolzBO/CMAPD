import numpy as np
from submodules.cmapd.AM.oracle.oracle import oracle
from multiprocessing import Pool, cpu_count
from submodules.cmapd.AM.regression.features_extractor import FeaturesExtractor
from submodules.cmapd.AM.regression.utils import read_grid, format_grid_for_oracle
from submodules.cmapd.AM.regression.grid_solver import get_solver
from xgboost import XGBRegressor
from submodules.cmapd.AM.regression.utils import ravel
from submodules.cmapd.AM.parameters import params


class HelperFunctions:
    def __init__(self):
        grid, self.grid_size = read_grid(params["environment"]["map"])
        self.grid = format_grid_for_oracle(grid)
        self.grid_solver = get_solver(params["environment"]["map"])

    # ---GLOBAL VARIABLES---
    # model = XGBRegressor()
    # model.load_model('model.ubj')
    # ----------------------

    def characteristic_function(self, waypoints):
        """
        Input:
            - waypoints: Pytorch tensor of (x, y) coordinates of points [number_of_agents, 2 * max_collective_size + 1, 2]

        Output:
            - value: Characteristic function value provided by the oracle (float)
        """
        num_agents = len(waypoints)
        # Numpy array with the number of waypoints for each agent (required to parse waypoints inside oracle)
        sep = np.array(list(map(len, waypoints)), dtype=np.int32)
        # Format waypoints to 1d numpy array (x * d_x + y)
        waypoints = np.array(
            [ravel(p, self.grid_size[1]) for a in waypoints for p in a], dtype=np.int32
        )
        # Call the oracle
        return oracle(
            self.grid,
            waypoints,
            sep,
            num_agents,
            len(waypoints),
            self.grid_size[0],
            self.grid_size[1],
        )

    def parallel_pbs(self, waypoints: list[list[list[list[int]]]]):
        """
        waypoints: a list of waypoint, that is, a list of list of point, that is, a list of int of length 2
        """
        with Pool(4) as pool:
            iterator_results = pool.map(self.characteristic_function, waypoints)
        return list(iterator_results)

    def parallel_distance(self, waypoints: list[list[list[list[int]]]]):
        with Pool(4) as pool:
            iterator_results = pool.map(self.distance, waypoints)
        return list(iterator_results)

    def distance(self, waypoint: list[list[list[int]]]):
        cost = 0
        for elem in waypoint:
            cost += self.grid_solver.get_waypoints_distance(elem)
        return cost


# def predict_costs(waypoints: list[list[list[list[int]]]]):
#     with Pool(cpu_count()) as pool:
#         iterator_results = pool.map(__extract_features, waypoints)
#     features_array = np.array(list(iterator_results))
#     return model.predict(features_array).tolist()
#
#
# def __extract_features(w):
#     extractor = FeaturesExtractor(w, grid, grid_size, grid_solver)
#     return extractor.get_features()
