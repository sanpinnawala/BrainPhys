from scipy.integrate import solve_ivp
from scipy.ndimage import laplace, convolve
from scipy.stats import truncnorm
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def generate_synthetic_data(hyperparams):
    num_samples = hyperparams['num_samples']
    data_dir = hyperparams['data_dir']

    train_num = int(0.8 * num_samples)
    val_num = num_samples - train_num

    train_data, train_time, train_param = generate_multiple_samples(hyperparams, train_num)
    val_data, val_time, val_param = generate_multiple_samples(hyperparams, val_num)
    test_data, test_time, test_param = generate_multiple_samples(hyperparams, num_samples)

    np.save(f"{data_dir}/train_data.npy", train_data)
    np.save(f"{data_dir}/train_time.npy", train_time)
    np.save(f"{data_dir}/train_param.npy", train_param)

    np.save(f"{data_dir}/val_data.npy", val_data)
    np.save(f"{data_dir}/val_time.npy", val_time)
    np.save(f"{data_dir}/val_param.npy", val_param)

    np.save(f"{data_dir}/test_data.npy", test_data)
    np.save(f"{data_dir}/test_time.npy", test_time)
    np.save(f"{data_dir}/test_param.npy", test_param)

def generate_multiple_samples(hyperparams, num):
    heat_data, time_data, param_data = [], [], []

    for _ in range(num):
        heat_map, predefined_t, params = generate_single_sample(hyperparams)
        heat_data.append(heat_map)
        time_data.append(predefined_t)
        param_data.append(params)

    return np.array(heat_data), np.array(time_data), np.array(param_data)

def generate_single_sample(hyperparams):
    alpha_range_x = hyperparams['alpha_range_x']
    alpha_range_y = hyperparams['alpha_range_y']
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

    alpha_x = np.random.uniform(alpha_range_x[0], alpha_range_x[1])  # x-direction thermal diffusivity
    alpha_y = np.random.uniform(alpha_range_y[0], alpha_range_y[1])  # y-direction thermal diffusivity
    reaction_coeff = np.random.uniform(reaction_coeff_range[0], reaction_coeff_range[1])  # reaction coefficient

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
    init_u = initial_tau.flatten()

    def laplace_x(y, dx):
        kernel = np.array([[1, -2, 1]])
        return convolve(y, kernel, mode='reflect') / dx ** 2

    def laplace_y(y, dy):
        kernel = np.array([[1], [-2], [1]])
        return convolve(y, kernel, mode='reflect') / dy ** 2

    def heat_ode(t, u):
        # reshape the 2d grid into a vector
        u_reshaped = u.reshape((spatial_points, spatial_points))

        # anisotropic diffusion
        du_dt_x = alpha_x * laplace_x(u_reshaped, dx)
        du_dt_y = alpha_y * laplace_y(u_reshaped, dy)

        du_dt = du_dt_x + du_dt_y

        # reaction term
        if reaction:
            du_dt += reaction_coeff * u_reshaped * (1 - u_reshaped)
        return du_dt.flatten()

    # solve ODE
    sol = solve_ivp(heat_ode, [0., total_time], init_u, t_eval=t, method='RK45')

    # reshape back into 2d spatial grids for each time step
    heat_map = np.array([sol.y[:, i].reshape((spatial_points, spatial_points)) for i in selected_indices])

    if noise:
        # add noise (snr = 10)
        noise_std = np.mean(heat_map) / snr
        # no negatives
        noise = truncnorm.rvs(0, np.inf, loc=0, scale=noise_std, size=heat_map.shape)
        noisy_heat_map = heat_map + noise
        # normalise
        normalised_heat_map = (noisy_heat_map - noisy_heat_map.min()) / (noisy_heat_map.max() - noisy_heat_map.min())
    else:
        normalised_heat_map = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min())

    params = (alpha_x, alpha_y, reaction_coeff)
    return normalised_heat_map, predefined_t, params

