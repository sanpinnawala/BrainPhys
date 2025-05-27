from scipy.integrate import solve_ivp
from scipy.ndimage import laplace, convolve
from scipy.stats import truncnorm
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numba import njit

@njit
def mask_aware_laplacian(y, mask, dx):
    lap = np.zeros_like(y)
    kernel = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(1, y.shape[0] - 1):
        for j in range(1, y.shape[1] - 1):
            if mask[i, j]:
                total = 0.0
                count = 0
                for di, dj in kernel:
                    ni, nj = i + di, j + dj
                    if mask[ni, nj]:
                        total += y[ni, nj]
                        count += 1
                lap[i, j] = (total - count * y[i, j]) / dx ** 2
    return lap

def generate_circular_mask(hyperparams):
    spatial_points = hyperparams['spatial_points']
    mask = np.zeros((spatial_points, spatial_points), dtype=np.uint8)
    center = spatial_points // 2
    radius = spatial_points // 2

    # circular domain mask (inside=1, outside=0)
    for i in range(spatial_points):
        for j in range(spatial_points):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < radius - 1:
                mask[i, j] = 1  # active region
    return mask

def generate_single_sample(hyperparams, mask):
    alpha_range = hyperparams['alpha_range']
    total_space = hyperparams['total_space']
    spatial_points = hyperparams['spatial_points']
    total_time = hyperparams['total_time']
    time_points = hyperparams['time_points']
    predefined_t = hyperparams['predefined_t']
    reaction = hyperparams['reaction']
    reaction_coeff_range = hyperparams['reaction_coeff_range']
    noise = hyperparams['noise']
    snr = hyperparams['snr']

    x = np.linspace(0., total_space, spatial_points)  # spatial domain
    y = np.linspace(0., total_space, spatial_points)  # spatial domain
    t = np.linspace(0., total_time, time_points)  # temporal domain
    selected_indices = np.searchsorted(t, predefined_t)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    alpha = np.random.uniform(alpha_range[0], alpha_range[1])  # thermal diffusivity
    if reaction:
        reaction_coeff = np.random.uniform(reaction_coeff_range[0], reaction_coeff_range[1])  # reaction coefficient
    else:
        reaction_coeff = 0.0

    initial_tau = np.zeros((spatial_points, spatial_points))
    radius = total_space / 4
    center_x, center_y = spatial_points // 2, spatial_points // 2

    for i in range(spatial_points):
        for j in range(spatial_points):
            # Euclidean distance from the center
            dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if dist < radius:
                initial_tau[i, j] = np.exp(
                    -dist ** 2 / (2 * radius ** 2))  # gaussian-like initial concentration

    # initial conditions
    init_u = (initial_tau * mask).flatten()

    def laplace_x(y, dx):
        kernel = np.array([[1, -2, 1]])
        return convolve(y, kernel, mode='reflect') / dx ** 2

    def laplace_y(y, dy):
        kernel = np.array([[1], [-2], [1]])
        return convolve(y, kernel, mode='reflect') / dy ** 2

    def heat_ode(t, u):
        # reshape the 2d grid into a vector
        u_reshaped = u.reshape((spatial_points, spatial_points))

        # du_dt_x = alpha * laplace_x(u_reshaped, dx)
        # du_dt_y = alpha * laplace_y(u_reshaped, dy)

        # du_dt = du_dt_x + du_dt_y
        du_dt = alpha * (mask_aware_laplacian(u_reshaped, mask, dx))

        # reaction term
        if reaction:
            du_dt += reaction_coeff * u_reshaped * (1 - u_reshaped)

        return du_dt.flatten()

    # solve ODE
    sol = solve_ivp(heat_ode, [0., total_time], init_u, t_eval=t, method='RK45')

    # reshape back into 2d spatial grids for each time step
    heat_map = np.array([sol.y[:, i].reshape((spatial_points, spatial_points)) for i in selected_indices])

    # mask for each timepoint
    heat_map_mask = np.repeat(mask[None, :, :], len(predefined_t), axis=0)

    if noise:
        # add noise (snr = 5)
        noise_std = np.mean(heat_map) / snr
        # no negatives
        noise = truncnorm.rvs(0, np.inf, loc=0, scale=noise_std, size=heat_map.shape)
        noisy_heat_map = heat_map + (noise * heat_map_mask)
        # normalise
        normalised_heat_map = (noisy_heat_map - noisy_heat_map.min()) / (noisy_heat_map.max() - noisy_heat_map.min())
    else:
        normalised_heat_map = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min())

    params = (alpha, reaction_coeff)
    return normalised_heat_map, heat_map_mask, predefined_t, params

def generate_multiple_samples(hyperparams, num):
    heat_data, mask_data, time_data, param_data = [], [], [], []
    mask = generate_circular_mask(hyperparams)

    for _ in range(num):
        heat_map, heat_map_mask, predefined_t, params = generate_single_sample(hyperparams, mask)
        heat_data.append(heat_map)
        mask_data.append(heat_map_mask)
        time_data.append(predefined_t)
        param_data.append(params)

    return np.array(heat_data), np.array(mask_data), np.array(time_data), np.array(param_data)

def generate_synthetic_data(hyperparams):
    num_samples = hyperparams['num_samples']
    data_dir = hyperparams['data_dir']

    train_num = int(0.8 * num_samples)
    val_num = num_samples - train_num

    train_data, train_mask, train_time, train_param = generate_multiple_samples(hyperparams, train_num)
    val_data, val_mask, val_time, val_param = generate_multiple_samples(hyperparams, val_num)
    test_data, test_mask, test_time, test_param = generate_multiple_samples(hyperparams, num_samples)

    np.save(f"{data_dir}/train_data.npy", train_data)
    np.save(f"{data_dir}/train_mask.npy", train_mask)
    np.save(f"{data_dir}/train_time.npy", train_time)
    np.save(f"{data_dir}/train_param.npy", train_param)

    np.save(f"{data_dir}/val_data.npy", val_data)
    np.save(f"{data_dir}/val_mask.npy", val_mask)
    np.save(f"{data_dir}/val_time.npy", val_time)
    np.save(f"{data_dir}/val_param.npy", val_param)

    np.save(f"{data_dir}/test_data.npy", test_data)
    np.save(f"{data_dir}/test_mask.npy", test_mask)
    np.save(f"{data_dir}/test_time.npy", test_time)
    np.save(f"{data_dir}/test_param.npy", test_param)