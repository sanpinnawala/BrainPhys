from scipy.integrate import solve_ivp
from scipy.ndimage import laplace, convolve
from scipy.stats import truncnorm
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_sample(spatial_points, total_time, selected_t, alpha_x, alpha_y, reaction_coeff):
    total_space = spatial_points - 1
    time_points = total_time + 1

    x = np.linspace(0., total_space, spatial_points)  # spatial domain
    y = np.linspace(0., total_space, spatial_points)  # spatial domain
    t = np.linspace(0., total_time, time_points)  # temporal domain
    selected_indices = np.searchsorted(t, selected_t)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

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
        du_dt += reaction_coeff * u_reshaped * (1 - u_reshaped)
        return du_dt.flatten()

    # solve ODE
    sol = solve_ivp(heat_ode, [0., total_time], init_u, t_eval=t, method='RK45')

    # reshape back into 2d spatial grids for each time step
    heat_map = np.array([sol.y[:, i].reshape((spatial_points, spatial_points)) for i in selected_indices])
    normalised_heat_map = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min())

    return normalised_heat_map