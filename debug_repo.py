import datetime
import numpy as np
import pandas as pd
import xarray as xr
import sys
import os
import math
sys.path.append("/home/voit/Radolan/skripte/wei/paper_repo")
os.chdir("/home/voit/Radolan/skripte/wei/paper_repo")
import xwei_functions as xf

import yaml
with open("/home/voit/Radolan/skripte/config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

sys.path.append(config["path2package"])

import wei_package as wei
import matplotlib.pyplot as plt
import datetime

event = xr.open_dataset("data/braunsbach/event_16058_200.nc")['rainfall_amount']
path2gev_parms = "data/braunsbach/gev_parameters"

durations = ['01', '02', '04', '06', '12', '24', '48', '72']
tolerance_dict={'01': 1, '02': 2, '04': 4, '06': 6, '12': 11, '24': 22, '48': 36, '72': 60}
max_rp = 1000


#ranking = pd.read_csv("/home/voit/Radolan/results/ranking/ranked_all.csv")

wei.Event.config(dict(catrare_path=config['path2catrare'],
                      radolan_file_path=config['path2ncdf']))

# e = wei.Event(16058, 100, durations)
# res = wei.Wei(e).gev(config['path2gev_fits'])



path2roi_parms = "/home/voit/Radolan/skripte/wei/paper_repo/data/braunsbach/roi_parameters/"
max_eta_roi = xf.eta_roi(event, path2roi_parms, durations, tolerance_dict, max_rp)

def weighting_f(d, n, tp):
    """
    Function to calculate the weights based on the distance to the center of the raster for ROI-method. Center has weight one.
    This approach is based on the paper of Burn [1990]
    :param d: distance to be weighted, float
    :param n: parameter 1, have to be checked so there  won't be negative weights. This depends on the choosen distance
    :param tp: parameter 2, have to be checked so there  won't be negative weights. This depends on the choosen distance
    :return: weight
    """
    mu = 1 - (d / tp) ** n
    return mu

def PWM(x):
    """
    Probability weighted moments for a series of data points. According to Burn (1990)
    :param x: series of data points, float
    :return: tuple of the first three moments of the distribution
    """
    # np number of years, x timeseries at point
    # sort x?? Hosking et al. Yes
    x = np.array(sorted(x))

    moments = []
    for i in range(0, 3):
        p = [((j - 0.35) / len(x)) ** i for j in range(1, len(x) + 1)]
        p = np.array(p)
        mr = (1 / len(x)) * sum(p * x)
        moments.append(mr)

    return moments


window_size = 19
# center window array
center = window_size // 2

def dist(p1, p2):
    """
    Euclidian distance between two 2D points
    :param p1: tuple, array index (2D)
    :param p2: tuple, array index (2D)
    :return: distance array
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d = (dy ** 2 + dx ** 2) ** 0.5
    return d


dist_array = np.zeros((window_size, window_size))

for row in range(0, window_size):
    for col in range(0, window_size):
        dist_array[row, col] = dist([center, center], [row, col])

yearmax_path = "/home/voit/Radolan/skripte/wei/paper_repo/data/braunsbach/yearmax_2001_2020/"


def roi(yearmax_path, roi_size):
    center = roi_size // 2
    files = os.listdir(yearmax_path)
    durations = [int(file[:2]) for file in files ]
    parms_dict = dict.fromkeys(sorted(durations))

    #create the distance array
    distance_array = np.zeros((roi_size, roi_size))

    for row in range(roi_size):
        for col in range(roi_size):
            distance_array[row, col] = dist([center, center], [row, col])

    for file in files:

        print(f"{datetime.datetime.now()} Fitting ROI parameters for duration {file[:2]} h.")
        nc = xr.open_dataset(f"{yearmax_path}/{file}")
        nc = nc['rainfall_amount'].values
        nc = nc[:20, :, :]

        # pad with nan so no values will be lost. Otherwise np.slinding_window will return a smaller array
        nc = np.pad(nc, center, 'constant', constant_values=np.nan)
        # remove pads for the 3rd(time) dimension
        nc = nc[center:nc.shape[0] - center, :, :]

        #create dictionary to store all the arrays
        parm_names = ["g", "xi", "alpha", "m0"]
        res = dict.fromkeys(parm_names)


        """
        Using this moving window approach one can save the padding when splitting up the array for multiprocessing.
        Basically for each cell in the array there is the window stored as additionally dimensions. The window/cube for
        each respective cell (x,y) can be accessed by: file[:, x, y, :, :]
        """
        nc = np.lib.stride_tricks.sliding_window_view(nc, window_shape=(roi_size, roi_size), axis=(1, 2))

        # arrays to store the results
        g_store = np.empty(shape=(nc.shape[1], nc.shape[2]))
        g_store[:] = np.nan
        xi_store = np.empty(shape=(nc.shape[1], nc.shape[2]))
        xi_store[:] = np.nan
        alpha_store = np.empty(shape=(nc.shape[1], nc.shape[2]))
        alpha_store[:] = np.nan
        m0_store = np.empty(shape=(nc.shape[1], nc.shape[2]))
        m0_store[:] = np.nan

        for row in range(nc.shape[1]):
            for col in range(nc.shape[2]):
                weight_dist = weighting_f(distance_array, 1, 26)
                r = nc[:, row, col, :, :]

                # if np.isnan(r[:, center, center]).sum() != 0:
                if any(np.isnan(r[:, center, center])):
                    continue

                else:
                    pwm_grid = np.zeros((3, roi_size, roi_size))

                    for m in range(0, pwm_grid.shape[1]):
                        for n in range(0, pwm_grid.shape[2]):
                            if any(np.isnan(r[:, m, n])):

                                pwm_grid[0, m, n] = np.nan
                                pwm_grid[1, m, n] = np.nan
                                pwm_grid[2, m, n] = np.nan
                                weight_dist[m, n] = np.nan

                            else:
                                pw_moments = PWM(r[:, m, n])
                                pwm_grid[0, m, n] = pw_moments[0]
                                pwm_grid[1, m, n] = pw_moments[1]
                                pwm_grid[2, m, n] = pw_moments[2]

                    t1 = pwm_grid[1] / pwm_grid[0]
                    t2 = pwm_grid[2] / pwm_grid[0]

                    T1 = (np.nansum(t1 * r.shape[0] * weight_dist)) / (np.nansum(r.shape[0] * weight_dist))
                    T2 = (np.nansum(t2 * r.shape[0] * weight_dist)) / (np.nansum(r.shape[0] * weight_dist))

                    c = (2 * T1 - 1) / (3 * T2 - 1) - math.log(2) / math.log(3)
                    g = 7.859 * c + 2.955 * c ** 2
                    alpha = (2 * T1 - 1) * g / (math.gamma(1 + g) * (1 - 2 ** -g))
                    xi = 1 + alpha * (math.gamma(1 + g) - 1) / g

                    # needed for calculation of return period is also the mean value of r[:,center,center]

                    g_store[row, col] = g
                    xi_store[row, col] = xi
                    alpha_store[row, col] = alpha
                    m0_store[row, col] = r[:, center, center].mean()

        # dummy = pd.DataFrame(g_store)
        # dummy.to_csv(f"roi_g_{file[:2]}.csv", header=False, index=False)
        #
        # dummy = pd.DataFrame(xi_store)
        # dummy.to_csv(f"roi_xi_{file[:2]}.csv", header=False, index=False)
        #
        # dummy = pd.DataFrame(alpha_store)
        # dummy.to_csv(f"roi_alpha_{file[:2]}.csv", header=False, index=False)
        #
        # dummy = pd.DataFrame(m0_store)
        # dummy.to_csv(f"roi_m0_{file[:2]}.csv", header=False, index=False)

        res["g"] = g_store
        res["xi"] = xi_store
        res["alpha"] = alpha_store
        res["m0"] = m0_store

        current_duration = file[:2]
        parms_dict[current_duration] = res

    return parms_dict



#check subset:
g_compute = np.loadtxt("/home/voit/Radolan/skripte/wei/paper_repo/roi_g_02.csv", delimiter=",")

test = 1 / g_compute
np.max(test)


g_subset = np.loadtxt("/home/voit/Radolan/skripte/wei/paper_repo/data/braunsbach/roi_parameters/roi_alpha_01.csv",
                       delimiter=",")

g_compute = g_compute[20:180, 20:180]
g_subset = g_subset[20:180, 20:180]

np.array_equal(g_compute, g_subset)



for row in range(182):
    if np.array_equal(g_subset[row,:], g_compute[row,:]):
        print(f"row {row}: True")
    else:
        print(f"row {row}: False")
