import numpy as np
import xarray as xr
import datetime
import pandas as pd
from scipy.stats import genextreme as gev_dist
from scipy.interpolate import griddata
import matplotlib as mpl
import matplotlib.pyplot as plt
import math



def eta_gev(array, path2parameter_fits, duration_levels,
            nan_tolerance={'01': 1, '02': 2, '04': 4, '06': 6, '12': 11, '24': 22, '48': 36, '72': 60}, max_rp=1000):
    """
    Parameters
    ----------
    array: xarray.core.dataset.Dataset, array containing the rainfall values
    path2parameter_fits: String.
            Path to folder where the files for all the gev_parameters (shape, loc, scale) for each duration are.
    duration_levels: list of strings
            Durations for which the Eta should be computed. The according parameter files
            have to exist in the data folder. Example: ["01", "02", "04", "06", "12", "24", "48", "72"]
    nan_tolerance: Dictionary
            The higher the duration, the bigger the moving window. By default if the moving window just contains one4
             NaN, the resulting moving window sum is als NaN. With this dictionary the tolerance towards NaN can be
             adjusted. The keys need to be the durations, the integer describes how many minimum non NaN-values need
             to be inside the window to still calculate the sum and thereby ignoring the NaN. This dictionary sets the
             values for xarray.rolling(...min_periods= DICCTIONARY VALUE).sum().
             If set to None, there will be no tolerance to NaN.
    max_rp: Integer or None
            According to Müller & Kaspar (2014), to avoid unrealistically high return periods all return periods above
            this threshold will be set equall to max_rp. If set to None, no threshold is set.
        tolerance_dict: Dictionary

    :return: pd.Dataframe
            Dataframe oontaining all the Eta-values for each duration
    XXXXXX

    """
    wei_calc_start = datetime.datetime.now()

    # to store the results
    results_dict = dict.fromkeys(duration_levels, None)


    for duration in duration_levels:
        locals()[f"shape_{duration}"] =\
            np.loadtxt(f"{path2parameter_fits}/gev_shape_{duration.zfill(2)}.csv", delimiter=",")
        locals()[f"loc_{duration}"] = \
            np.loadtxt(f"{path2parameter_fits}/gev_loc_{duration.zfill(2)}.csv", delimiter=",")
        locals()[f"scale_{duration}"] =\
            np.loadtxt(f"{path2parameter_fits}/gev_scale_{duration.zfill(2)}.csv", delimiter=",")

    for i in range(0, len(duration_levels)):

        time_info = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"{time_info}: Calculating Eta for duration {duration_levels[i]}h.")

        subset = _make_input_array(duration_levels[i], max(duration_levels), array,
                                   nan_tolerance)

        duration = duration_levels[i]

        # calculate return periods
        subset[np.isnan(subset)] = 1
        Pu = gev_dist.cdf(subset, c=locals()[f"shape_{duration}"],
                          loc=locals()[f"loc_{duration}"],
                          scale=locals()[f"scale_{duration}"])

        Pu[Pu == 1] = 0.999  # To fix the runtime error when division is by zero

        rp_array = 1 / (1 - Pu)

        result = _calc_eta(max_rp, rp_array)

        '''
        this dictionary contains all the Eta values for every timestep of the event and every duration.
        '''
        results_dict[duration] = result

    '''
    For each duration we select the one with the highest value
    '''
    max_eta = get_max_eta(results_dict)

    max_vals = pd.DataFrame(max_eta).round(2)

    wei_calc_end = datetime.datetime.now() - wei_calc_start

    print(f"Total execution time: {wei_calc_end}")

    return max_vals

def eta_dgev(array, path2parameter_fits, duration_levels,
             nan_tolerance={'01': 1, '02': 2, '04': 4, '06': 6, '12': 11, '24': 22, '48': 36, '72': 60}, max_rp=1000):
    """
    Calculation of return periods based on the concept of duration dependent GEV-curves. dGEV parameters were
    derived with the R-package IDF. The advantage of this method is, that we just need five parameters that work for
    every duratoin instead of three for each duration (which would be in our case 24 parameter sets)
    References:
    Fauer, Felix S., et al. "Flexible and consistent quantile estimation for intensity–duration–frequency curves
    " Hydrology and Earth System Sciences 25.12 (2021): 6479-6494.

    Koutsoyiannis, D., Kozonis, D., and Manetas, A.: A mathematical framework for studying rainfall intensity-
    duration-frequency relationships, J. Hydrol., 206, 118–135, https://doi.org/10.1016/S0022-1694(98)00097-3,
    1998

    Parameters
    ----------
    array: xarray.core.dataset.Dataset, array containing the rainfall values
    path2parameter_fits: String.
            Path to folder where the files for all the dgev_parameters (shape, mod_loc, scale_0, duration offeset,
            duration exponent).
    duration_levels: list of strings
            Durations for which the Eta should be computed. The according parameter files
            have to exist in the data folder. Example: ["01", "02", "04", "06", "12", "24", "48", "72"]
    nan_tolerance: Dictionary
            The higher the duration, the bigger the moving window. By default if the moving window just contains one4
             NaN, the resulting moving window sum is als NaN. With this dictionary the tolerance towards NaN can be
             adjusted. The keys need to be the durations, the integer describes how many minimum non NaN-values need
             to be inside the window to still calculate the sum and thereby ignoring the NaN. This dictionary sets the
             values for xarray.rolling(...min_periods= DICCTIONARY VALUE).sum().
             If set to None, there will be no tolerance to NaN.
    max_rp: Integer or None
            According to Müller & Kaspar (2014), to avoid unrealistically high return periods all return periods above
            this threshold will be set equall to max_rp. If set to None, no threshold is set.
        tolerance_dict: Dictionary

    :return: pd.Dataframe
            Dataframe oontaining all the Eta-values for each duration
    XXXXXX
    """
    wei_calc_start = datetime.datetime.now()

    # to store the results
    results_dict = dict.fromkeys(duration_levels, None)

    mod_loc = np.loadtxt(f"{path2parameter_fits}/dgev_mod_loc.csv", delimiter=",")
    scale_0 = np.loadtxt(f"{path2parameter_fits}/dgev_scale_0.csv", delimiter=",")
    shape = np.loadtxt(f"{path2parameter_fits}/dgev_shape.csv", delimiter=",")
    duration_offset = np.loadtxt(f"{path2parameter_fits}/dgev_duration_offset.csv",
                                 delimiter=",")
    duration_exp = np.loadtxt(f"{path2parameter_fits}/dgev_duration_exp.csv", delimiter=",")


    for i in range(0, len(duration_levels)):
        time_info = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"{time_info}: Calculating WEI for duration {duration_levels[i]}h.")

        subset = _make_input_array(duration_levels[i], max(duration_levels), array,
                                   nan_tolerance)

        duration = duration_levels[i]

        # calculate return periods
        sigma_d = scale_0 * (int(duration) + duration_offset) ** -duration_exp
        mu_d = mod_loc * sigma_d
        #
        # dirty fix to fix nan
        subset[np.isnan(subset)] = 1
        Pu = gev_dist.cdf(subset / int(duration), c=shape, loc=mu_d, scale=sigma_d)
        Pu[Pu == 1] = 0.999  # To fix the runtime error when division is by zero

        rp_array = 1 / (1 - Pu)
        # rp_array = make_rp_array_dgev(subset_subset, mod_loc_subset, scale_0_subset, shape_subset,
        #                              duration_offset_subset, duration_exp_subset, int(duration))

        result = _calc_eta(max_rp, rp_array)

        results_dict[duration] = result

    max_eta = get_max_eta(results_dict)

    max_vals = pd.DataFrame(max_eta).round(2)

    wei_calc_end = datetime.datetime.now() - wei_calc_start

    print(f"Total execution time: {wei_calc_end}")

    return max_vals


def eta_roi(array, path2parameter_fits, duration_levels,
            nan_tolerance={'01': 1, '02': 2, '04': 4, '06': 6, '12': 11, '24': 22, '48': 36, '72': 60}, max_rp=1000):
    """
        Calculation of the return periods using the Region of Interest method (ROI), described by Burn (1990)
         "Evaluation of regional flood frequency analysis  with a region of influence approach"
        Here the approach is used for a raster surrounding the point of interest.
        The ROI uses four parameter: g, xi, alpha and M0.
        INFORMATION: These parameters are not to be confused with parameters for the GEV-distribution, they are rather
        used for a dimensionless distribution.


    Parameters
    ----------
    array: xarray.core.dataset.Dataset, array containing the rainfall values
    path2parameter_fits: String.
            Path to folder where the files for all the roi_parameters (g, xi, m0, alpha) for each duration are.
    duration_levels: list of strings
            Durations for which the Eta should be computed. The according parameter files
            have to exist in the data folder. Example: ["01", "02", "04", "06", "12", "24", "48", "72"]
    nan_tolerance: Dictionary
            The higher the duration, the bigger the moving window. By default if the moving window just contains one4
             NaN, the resulting moving window sum is als NaN. With this dictionary the tolerance towards NaN can be
             adjusted. The keys need to be the durations, the integer describes how many minimum non NaN-values need
             to be inside the window to still calculate the sum and thereby ignoring the NaN. This dictionary sets the
             values for xarray.rolling(...min_periods= DICCTIONARY VALUE).sum().
             If set to None, there will be no tolerance to NaN.
    max_rp: Integer or None
            According to Müller & Kaspar (2014), to avoid unrealistically high return periods all return periods above
            this threshold will be set equall to max_rp. If set to None, no threshold is set.
        tolerance_dict: Dictionary

    :return: pd.Dataframe
            Dataframe oontaining all the Eta-values for each duration.
    XXXXXX


    """
    wei_calc_start = datetime.datetime.now()
    # to store the results
    results_dict = dict.fromkeys(duration_levels, None)

    for duration in duration_levels:
        locals()[f"g_{duration}"] = np.loadtxt(f"{path2parameter_fits}/roi_g_{duration}.csv", delimiter=",")
        locals()[f"xi_{duration}"] = np.loadtxt(f"{path2parameter_fits}/roi_xi_{duration}.csv", delimiter=",")
        locals()[f"alpha_{duration}"] = np.loadtxt(f"{path2parameter_fits}/roi_alpha_{duration}.csv",
                                                   delimiter=",")
        locals()[f"m0_{duration}"] = np.loadtxt(f"{path2parameter_fits}/roi_m0_{duration}.csv", delimiter=",")


    for i in range(0, len(duration_levels)):
        time_info = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"{time_info}: Calculating WEI for duration {duration_levels[i]}h.")

        subset = _make_input_array(duration_levels[i], max(duration_levels), array, nan_tolerance)

        duration = duration_levels[i]

        # calculate return periods

        subset[np.isnan(subset)] = 1
        a = ((subset / locals()[f"m0_{duration}"]) - locals()[f"xi_{duration}"]) * \
            locals()[f"g_{duration}"] / locals()[f"alpha_{duration}"]

        rp_array = 1 / (-np.exp(-(-a + 1) ** (1 / locals()[f"g_{duration}"])) + 1)

        ###DEBUG####
        if i == 1:
            print("Stop")
        for j in range(a.shape[0]):
            test = 1 / (-np.exp(-(-a[i, :, :] + 1) ** (1 / locals()[f"g_{duration}"])) + 1)

        result = _calc_eta(max_rp, rp_array)

        results_dict[duration] = result

    max_eta = get_max_eta(results_dict)

    max_vals = pd.DataFrame(max_eta).round(2)

    wei_calc_end = datetime.datetime.now() - wei_calc_start

    print(f"Total execution time: {wei_calc_end}")

    return max_vals


def _make_input_array(duration, max_duration, array, nan_tolerance):
    '''
    Creates the input array and optimizes the shape of the inital xarray. x and y are subset time is trimmed so it
    is just enough to compute the rolling window statistics.

    :param duration: string,
        Duration level to create the array for. Duration is needed for rolling window calculation
    :param max_duration: string
        max duration that will be computed. this is needed for trimming the array
    :param array: xr.DataArray
        Data array with maximum time dimensions in 1h resolution.
    :param nan_tolerance: Dictionary
            The higher the duration, the bigger the moving window. By default if the moving window just contains one4
             NaN, the resulting moving window sum is als NaN. With this dictionary the tolerance towards NaN can be
             adjusted. The keys need to be the durations, the integer describes how many minimum non NaN-values need
             to be inside the window to still calculate the sum and thereby ignoring the NaN. This dictionary sets the
             values for xarray.rolling(...min_periods= DICCTIONARY VALUE).sum().
             If set to None, there will be no tolerance to NaN.
    :return: np.array,
        A trimmed rainfall array ready for rolling window aggregation
    '''


    padding_left = int(int(int(max_duration)) / 2)
    padding_right = padding_left - 1

    if int(duration) > 1:
        # pre-trim (time) the subset here for less rolling computing
        padding_left = int(int(int(max_duration)) / 2)
        padding_right = padding_left - 1
        index_left = int(padding_left - int(duration) / 2)
        index_right = int(array.shape[0] - padding_right + int(duration) / 2 - 1)
        subset = array[index_left:index_right, :, :]
        if nan_tolerance is None:
            subset = subset.rolling(time=int(duration), center=True).sum()

        else:
            subset = subset.rolling(time=int(duration), min_periods=nan_tolerance[duration],
                                    center=True).sum()
        padding_left = int(int(duration) / 2)
        padding_right = subset.shape[0] - padding_left + 1
        subset = subset[padding_left: padding_right, :, :]
        subset = subset.values

    else:
        if isinstance(array, np.ndarray):
            subset = array[padding_left:array.shape[0] - padding_right, :, :]
        if isinstance(array, xr.DataArray):
            subset = array.values[padding_left:array.shape[0] - padding_right, :, :]

    if np.isnan(subset).sum() != 0:
        print(f"WARNING: Uncertain results for duration {duration}h due to NaN in array.")

    return subset


def _calc_eta(max_rp, rp_array):
    """
    This function calculates the Eta series according to Müller and Kaspar (2014) for every timestep within one duration
    :param max_rp: Int. maximum possible return period
    :param rp_array: array with all the return periods
    :return: a list with all Eta-series for every timestep in one duration
    """

    if max_rp is not None:
        rp_array[rp_array > max_rp] = max_rp

    result = list()
    for j in range(0, (rp_array.shape[0])):
        data = rp_array[j, :, :].flatten(order="c")

        '''
        set nan to zero. check paper S.149
        does this cause the floating point error/ runtime warning in get_log_Gta?
        wouldn't it be better to set it to 1? does tis have an influence on the WEI?
        Before nan was set to 0 causing runtime issues
        '''
        data[np.isnan(data)] = 1
        '''
        this is a dirty fix:
        sometimes the GEV fit doesn't seem to work properly:
        cells that have less precipitation than the maximum cell value observed
        end up with a return period of infinite because the cdf of the gev distribution
        resulted in zero. Of course it seems highly unlikely, especially in the case,
        where it is neighbouring cells. This happens for example for Event 11 on the duration 06h.
        Because the inf values mess with the following computation, all inf values will be set to the
        maximum.To keep track of this manipulation, a log file should be written.'
        21.12.21: This fix is obsolete now due to the infinte values 
        which get set to the self.max_rp threshold except when self.max_rp == None
        '''
        if max_rp is None:
            data[np.where(np.isinf(data))] = np.nanmax(data[data != np.inf])

        data = np.sort(data)[::-1]

        eta = _get_eta(data)

        result.append(eta.round(3))


    return result


def _get_eta(data: np.array):
    """
    Calculation of Eta according to Müller and Kaspar (2014)
    :param data: flattened array of return periods
    :return: np.array of Eta values
    """
    cumulated = np.cumsum(np.log(data))
    size_cells = np.array([*range(len(data))]) + 1
    log_gta = cumulated / size_cells
    r = np.sqrt(size_cells) / np.sqrt(np.pi)
    eta = log_gta * r
    return eta


def get_max_eta(results_dict: dict):
    """
    Chooses the Eta series with highest maximum value for each duration. If an event is four hours long we make an look
    at the Eta-values for each hour and duration. For each duration we pick the one with highest maxima with this
    function

    Parameters
    ----------
    results_dict : Dictionary
        Eta series returned from function _calc_eta()

    Returns
    -------
    max_vals_dict : pd.Dataframe
        highest Eta series for each duration
    """

    max_vals = dict.fromkeys(results_dict.keys(), None)
    # areas_dict_new = dict.fromkeys(results_dict.keys(), None)

    for key in results_dict.keys():
        for i in range(0, len(results_dict[key])):
            if i == 0:
                max_vals[key] = results_dict[key][i]
            #  areas_dict_new[key] = area_dict[key]
            else:
                if np.nanmax(results_dict[key][i]) > np.nanmax(max_vals[key]):
                    max_vals[key] = results_dict[key][i]
            #     areas_dict_new[key] = area_dict[key][i]

    return pd.DataFrame.from_dict(max_vals).round(2)

def calc_xwei(eta_vals: pd.DataFrame, resolution=1000, precision=1, show_plot=False, rotate=45) -> float:
    """
    Calculates the xWEI based on the given Eta values. Values are interpolated with option "cubic". To see the volume
    which is calculated one can use plot_xwei() which uses this method.

    Parameters
    ----------
    eta_vals : pd.DataFrame
        Dataframe containing the maximum Eta series for each duration. This
        could be f.e. the output of wei.get_max_eta() converted to a pandas data
        frame.
    resolution: int
        scaling of the duration axis. The higher the number, the finer the grid, the longer the computation will take
        1000 seems to be a good compromise.
    precision: int: 1
        aggregates the input eta_val dataframe. This can significantly speed up the calculation of xwei but will
        result in a less accurate calculation. Precision can be from 1 (no aggregation, highest precision) to a higher
        number, depending on the length of the input dataframe.
    show_plot: Boolean: False
        plots the xWEI
    rotate: int
        rotates the plot. Just used if show_plot=True


    Returns
    -------
    xWEI: Float (rounded to 3 decimals)
    """

    int_divisors = [number for number in range(1, len(eta_vals) + 1) if len(eta_vals + 1) % number == 0]
    print(f"You can choose between precision levels 1 (highest precision) to {len(int_divisors)} (lowest precision)")

    aggregated = int_divisors[precision - 1]

    if aggregated > 1:
        eta_vals = eta_vals.groupby(eta_vals.index // aggregated).mean()

    # create data array with coordinates for all the points
    coords = np.ones((len(eta_vals.columns) * len(eta_vals), 3))

    x_coords = list(np.arange(1, len(eta_vals) + 1, 1) * aggregated - (aggregated - 1)) * len(eta_vals.columns)

    durations = sorted([int(eta_vals.columns[i]) for i in range(len(eta_vals.columns))])
    # from https://stackoverflow.com/questions/2449077/duplicate-each-member-in-a-list
    y_coords = [val for val in durations for _ in (range(1, len(eta_vals) + 1))]

    # for sure this can be done better with pd.Dataframe.stack oder .melt(). Double list comprehension...

    stacked_cols = [eta_vals[col] for col in eta_vals.columns]
    z_coords = [item for sublist in stacked_cols for item in sublist]

    z_coords = np.array(eta_vals).flatten("F")  # basically a vertical stack of all the columns

    coords[:, 0] = np.log(x_coords)
    coords[:, 1] = np.log(y_coords)
    coords[:, 2] = z_coords

    x_range = np.linspace(0, np.log(len(eta_vals) * aggregated), len(eta_vals))  # * aggregated
    y_range = np.linspace(0, np.log(durations[-1]), resolution)
    grid_x, grid_y = np.meshgrid(x_range, y_range)

    # interpolate the eta values to get a surface
    grid_z = griddata(coords[:, 0:2], coords[:, 2], (grid_x, grid_y), method='cubic')

    dx = np.log(len(eta_vals) * aggregated) / len(eta_vals)
    dy = np.log(durations[-1]) / resolution

    xwei = np.nansum(dx * dy * grid_z)

    if show_plot:
        # get the yticks. There should be a more elegant way to do it
        x_ticks = [1]

        while x_ticks[-1] < 2 * len(eta_vals) * aggregated:
            x_ticks.append(x_ticks[-1] * 10)

        xticks = np.log(np.array(x_ticks))
        yticks = np.log(np.array(durations))

        fig = plt.figure(figsize=(8, 12))
        ax = plt.axes(projection='3d')

        ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none', alpha=0.9, rstride=10)

        ax.set_title("xWEI")
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(durations)
        ax.set_ylabel('Duration [h]')
        ax.set_xlabel('Area [km²]')
        ax.set_zlabel('$E_{ta}$')
        ax.view_init(30, rotate)

        plt.show()

    return round(xwei, 3)


def plot_eta(max_vals: pd.DataFrame):
    """
    Plots the Eta-curves
    Parameters
    ----------
    max_vals : pd.Dataframe
        Dataframe with maximum Eta values for each duration generated. The output of eta_gev/dgev/roi functions can be
        used.


    Returns
    -------
    Plot

    """

    xmax = 0
    eta_max = 0
    which_duration_max = list(max_vals.keys())[0]
    plt.figure(figsize=(12,8))

    line_colors = plt.colormaps['tab10'](np.linspace(0, 1, len(max_vals.columns)))

    max_etas = max_vals.max().to_list()
    max_etas = np.array(max_etas) * -1  # inverse the values so the sorting will be descending

    indices = list(range(len(max_etas)))  # from stack overflow
    indices.sort(key=lambda x: max_etas[x])
    ranks = [0] * len(indices)
    for i, j in enumerate(indices):
        ranks[j] = i

    line_colors = line_colors[ranks, :]

    for i, key in enumerate(max_vals.keys()):

        if len(max_vals[key]) > xmax:
            xmax = len(max_vals[key])
            which_duration_max = key

        if np.nanmax(max_vals[key]) > eta_max:
            eta_max = np.nanmax(max_vals[key])
            which_duration_max = key

        x = np.linspace(0, xmax, max_vals[key].shape[0])

        plt.semilogx(x, max_vals[key], label=f"{key}h", color=line_colors[i])

    # keep x and y lim
    xlim = plt.xlim()
    # ylim = plt.ylim()

    # this has to be done if there is more than one x value for the maximum
    max_xindex = np.where(max_vals[which_duration_max] == eta_max)
    max_xindex = max_xindex[0].mean().round(0)
    max_xindex = int(max_xindex)

    plt.axvline(x=max_xindex, ymin=0, ymax=1, linestyle=':', color='red')

    plt.xlabel("$Area [km²]$")
    plt.ylabel("$E_{tA}$")
    plt.grid(which="both", ls="-")

    nl = '\n'  # for linebreak in f string
    plt.annotate(
        f"max. WEI = {eta_max.round(2)} ({which_duration_max}h){nl}area max = {np.where(max_vals[which_duration_max] == eta_max)[0][0]} $km²$",
        (1, 0.3 * eta_max), fontsize=10)

    # old version with annotation at maximum
    # (np.where(max_vals[which_duration_max] ==Eta_max)[0][0] - 100, Eta_max - 0.8)

    plt.legend(fontsize=8, loc='upper left')

    plt.show()

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


def roi(ncfile, durations, window_size, distance_array):
    '''
    Calculates the ROI-parameters for each duration
    :param ncfile: 
    :param durations:
    :param window_size:
    :param distance_array:
    :return:
    '''
    center = window_size // 2
    parm_names = ["g", "xi", "alpha", "m0"]

    # create dictionary to store all the arrays
    parms_dict = dict.fromkeys(durations)

    for i in range(len(durations)):
        # open ncfile with yearmax values
        nc = ncfile['rainfall_amount'].values

        nc = nc[:20, :, :]

        # pad with nan so no values will be lost. Otherwise np.slinding_window will return a smaller array
        nc = np.pad(nc, center, 'constant', constant_values=np.nan)
        # remove pads for the 3rd(time) dimension
        nc = nc[center:nc.shape[0] - center, :, :]

        """
        Using this moving window approach one can save the padding when splitting up the array for multiprocessing.
        Basically for each cell in the array there is the window stored as additionally dimensions. The window/cube for each
        respective cell (x,y) can be accessed by: file[:, x, y, :, :]
        by 
        """
        nc = np.lib.stride_tricks.sliding_window_view(nc, window_shape=(window_size, window_size), axis=(1, 2))

        # arrays to store the results
        g_store = np.empty(shape=(nc.shape[1], nc.shape[2]))
        g_store[:] = np.nan
        xi_store = np.empty(shape=(nc.shape[1], nc.shape[2]))
        xi_store[:] = np.nan
        alpha_store = np.empty(shape=(nc.shape[1], nc.shape[2]))
        alpha_store[:] = np.nan
        m0_store = np.empty(shape=(nc.shape[1], nc.shape[2]))
        m0_store[:] = np.nan

        res = dict.fromkeys(parm_names)

        for row in range(nc.shape[1]):
            for col in range(nc.shape[2]):
                weight_dist = weighting_f(distance_array, 1, 26)
                r = nc[:, row, col, :, :]

                # if np.isnan(r[:, center, center]).sum() != 0:
                if any(np.isnan(r[:, center, center])):
                    continue

                else:
                    pwm_grid = np.zeros((3, window_size, window_size))

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

        res["g"] = g_store
        res["xi"] = xi_store
        res["alpha"] = alpha_store
        res["m0"] = m0_store

        parms_dict[durations[i]] = res

        return parms_dict
