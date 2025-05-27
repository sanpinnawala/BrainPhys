import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
from utils.analysis import get_sample

def plot_recon(x, x_t, x_recon, path):
    n_cols = len(x_t)  # number of timepoints to plot
    n_rows = 3

    subplot_titles = []

    subplot_titles += [f"True t{x_t[i]}" for i in range(n_cols)]
    subplot_titles.append(f"True t{x_t[0]}")
    subplot_titles += [f"Predicted t{x_t[i]}" for i in range(1, n_cols)]
    subplot_titles += [f"Residual t{x_t[i]}" for i in range(n_cols)]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True, shared_yaxes=True,
        x_title='X', y_title='Y',
        horizontal_spacing=0.025, vertical_spacing=0.05
    )
    colorbars = [
        dict(title="True", y=0.92, len=0.35),
        dict(title="Predicted", y=0.55, len=0.35),
        dict(title="Residual", y=0.18, len=0.35)
    ]
    all_values = np.concatenate([x[i].flatten() for i in range(n_cols)] + [x_recon[i].flatten() for i in range(n_cols)])
    global_min = np.min(all_values)
    global_max = np.max(all_values)

    all_diffs = [np.round((x_recon[i] - x[i]), decimals=4) for i in range(n_cols)]
    global_diff_max = np.max(np.abs(all_diffs))

    for i in range(n_cols):
        # ground truth
        fig.add_trace(go.Heatmap(z=x[i], colorscale="Inferno", showscale=(i == n_cols - 1), zmin=global_min, zmax=global_max, colorbar=colorbars[0] if i == n_cols - 1 else None), row=1, col=i + 1)
        # prediction
        fig.add_trace(go.Heatmap(z=x_recon[i], colorscale="Inferno", showscale=(i == n_cols - 1), zmin=global_min, zmax=global_max, colorbar=colorbars[1] if i == n_cols - 1 else None), row=2, col=i + 1)
        # difference
        diff_map = np.round((x_recon[i]- x[i]), decimals=4)
        fig.add_trace(go.Heatmap(z=diff_map, colorscale="RdBu", showscale=(i == n_cols - 1), zmin=-global_diff_max, zmax=global_diff_max, colorbar=colorbars[2] if i == n_cols - 1 else None), row=3, col=i + 1)

    fig.update_layout(height=n_rows * 275, width=n_cols * 300, showlegend=False, xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))

    fig.write_image(path)


def plot_uncertainty(x, recons_stacked, cross_section_idx, path):
    last_timepoint = recons_stacked[:, 0, -1, :, :]  # [N, 25, 25]
    true_image = x[-1, :, :]

    true_x = true_image[cross_section_idx, :]
    true_y = true_image[:, cross_section_idx]

    x_cross_section = last_timepoint[:, cross_section_idx, :]  # [N, 25]
    mean_x = np.mean(x_cross_section, axis=0)
    std_x = np.std(x_cross_section, axis=0)
    min_x = np.min(x_cross_section, axis=0)
    max_x = np.max(x_cross_section, axis=0)

    y_cross_section = last_timepoint[:, :, cross_section_idx]
    mean_y = np.mean(y_cross_section, axis=0)
    std_y = np.std(y_cross_section, axis=0)
    min_y = np.min(y_cross_section, axis=0)
    max_y = np.max(y_cross_section, axis=0)

    x_positions = np.arange(last_timepoint.shape[0])

    fig = make_subplots(
        rows=1, cols=2,
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.15, vertical_spacing=0.05,
        x_title='Spatial Position', y_title='Intensity',
        subplot_titles=["X cross-section", "Y cross-section"]
    )

    fig.add_trace(go.Scatter(x=x_positions, y=true_x, mode='lines', line=dict(color='royalblue'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_positions, y=mean_x, mode='lines', line=dict(color='rgba(106,81,163,1)'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_positions, y=max_x, mode='lines', line=dict(color='rgba(106,81,163,0)'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_positions, y=min_x, mode='lines', fill='tonexty', fillcolor='rgba(106,81,163,0.2)', line=dict(color='rgba(106,81,163,0)'), showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=x_positions, y=true_y, mode='lines', line=dict(color='royalblue'), name='True'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_positions, y=mean_y, mode='lines', line=dict(color='rgba(106,81,163,1)'), name='Mean of Predictions'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_positions, y=max_y, mode='lines', line=dict(color='rgba(106,81,163,0)'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_positions, y=min_y, mode='lines', fill='tonexty', fillcolor='rgba(106,81,163,0.2)', line=dict(color='rgba(106,81,163,0)'), name='Range of Predictions'), row=1, col=2)

    fig.update_layout(height=400, width= 750, template="plotly", showlegend=True, legend=dict(x=0.5, y=-0.25, orientation="h", xanchor="center"))

    fig.write_image(path)


def plot_hist(zX, zR, param_path, path):
    learned_values = np.hstack([zX.detach().cpu().numpy(), zR.detach().cpu().numpy()])
    true_values = np.load(param_path)
    true_values_stacked = np.column_stack([true_values[:, 0], true_values[:, 1]])

    errors = learned_values - true_values_stacked

    df = pd.DataFrame(errors, columns=['zX', 'zR'])
    df_melted = df.melt(var_name='Parameter', value_name='Error')
    custom_colors = [px.colors.sequential.Plasma[0], 'rgba(229,123,2,1)']

    fig = px.histogram(df_melted, x="Error", color="Parameter", marginal="box", nbins=50, opacity=0.5,
                       color_discrete_sequence=custom_colors, barmode="overlay")
    fig.update_layout(yaxis_title="Count", font=dict(size=18))

    fig.write_image(path)


def plot_extrap(full_t, x_extrap, param_path, path):
    space = x_extrap[0][0].shape[0]
    # get true data
    test_param = np.load(param_path)
    true_extrap = get_sample(int(space), int(full_t[-1]), full_t, test_param[0][0], test_param[0][1])

    n_cols = len(full_t)  # number of timepoints to plot
    n_rows = 3

    subplot_titles = []

    subplot_titles += [f"True t{full_t[i]}" for i in range(n_cols)]
    subplot_titles.append(f"True t{full_t[0]}")
    subplot_titles += [f"Predicted t{full_t[i]}" for i in range(1, n_cols)]
    subplot_titles += [f"Residual t{full_t[i]}" for i in range(n_cols)]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True, shared_yaxes=True,
        x_title='X', y_title='Y',
        horizontal_spacing=0.025, vertical_spacing=0.05
    )

    colorbars = [
        dict(title="True", y=0.92, len=0.35),
        dict(title="Predicted", y=0.55, len=0.35),
        dict(title="Residual", y=0.18, len=0.35)
    ]
    all_values = np.concatenate([true_extrap[i].flatten() for i in range(n_cols)] + [x_extrap[i].flatten() for i in range(n_cols)])
    global_min = np.min(all_values)
    global_max = np.max(all_values)

    all_diffs = [np.round((x_extrap[i] - true_extrap[i]), decimals=4) for i in range(n_cols)]
    global_diff_max = np.max(np.abs(all_diffs))

    for i in range(n_cols):
        # true
        fig.add_trace(go.Heatmap(z=true_extrap[i], colorscale="Inferno", showscale=(i == n_cols - 1), zmin=global_min, zmax=global_max, colorbar=colorbars[0] if i == n_cols - 1 else None), row=1, col=i + 1)
        # prediction
        fig.add_trace(go.Heatmap(z=x_extrap[i], colorscale="Inferno", showscale=(i == n_cols - 1), zmin=global_min, zmax=global_max, colorbar=colorbars[1] if i == n_cols - 1 else None), row=2, col=i + 1)
        # difference
        diff_map = np.round((x_extrap[i] - true_extrap[i]), decimals=4)
        fig.add_trace(go.Heatmap(z=diff_map, colorscale="RdBu", showscale=(i == n_cols - 1), zmin=-global_diff_max, zmax=global_diff_max, colorbar=colorbars[2] if i == n_cols - 1 else None), row=3, col=i + 1)

    fig.update_layout(height=n_rows * 275, width=n_cols * 300, showlegend=False, xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))

    fig.write_image(path)

def plot_extrap_adni(full_t, x_extrap, path):
    n_cols = len(full_t)  # number of timepoints to plot
    n_rows = 1

    subplot_titles = [f"t{full_t[i]}" for i in range(n_cols)]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True, shared_yaxes=True,
        x_title='X', y_title='Y',
        horizontal_spacing=0.025, vertical_spacing=0.05
    )

    all_values = [x_extrap[i].flatten() for i in range(n_cols)]
    global_min = np.min(all_values)
    global_max = np.max(all_values)

    for i in range(n_cols):
        fig.add_trace(go.Heatmap(z=x_extrap[i], colorscale="Inferno", showscale=(i == n_cols - 1), zmin=global_min, zmax=global_max, colorbar=dict(y=0.5, len=1) if i == n_cols - 1 else None), row=1, col=i + 1)

    fig.update_layout(height=n_rows * 385, width=n_cols * 300, showlegend=False, xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))

    fig.write_image(path)

def plot_hist_adni(zX, path):
    #raise Exception(zY)
    true_values = np.hstack([zX.detach().cpu().numpy()])

    df = pd.DataFrame(true_values, columns=['zX', 'zY', 'zR'])
    df_melted = df.melt(var_name='Parameter', value_name='Value')
    custom_colors = [px.colors.sequential.Inferno[2], px.colors.sequential.Plasma[0], 'rgba(229,123,2,1)']

    fig = px.histogram(df_melted, x="Value", color="Parameter", marginal="box", nbins=13, opacity=0.5, color_discrete_sequence=custom_colors, barmode="overlay")
    fig.update_layout(yaxis_title="Count", font=dict(size=18))

    fig.write_image(path)

def get_mean_residual(x, recon_samples):
    x = x.detach().cpu().numpy()
    recons_stacked = np.stack(recon_samples, axis=0)
    mean_recons = np.mean(recons_stacked, axis=0)

    mean_residuals = np.mean(np.abs(x - mean_recons), axis=(0, 2, 3))

def test_plot_recon(x, x_t, recon_samples, dir):
    get_mean_residual(x, recon_samples)

    x = x.detach().cpu().numpy()[0]  # first sample
    x_t = x_t.detach().cpu().numpy()[0]
    idx = int(x.shape[2]/2)

    recons_stacked = np.stack(recon_samples, axis=0)
    mean_recon = np.mean(recons_stacked, axis=0)[0]

    plot_recon(x, x_t, mean_recon, path=f"{dir}/test_recon.png")
    plot_uncertainty(x, recons_stacked, cross_section_idx= idx, path=f"{dir}/test_uncert.png")

def test_plot_param(zX, zR, dir, data_dir):
    plot_hist(zX, zR, param_path=f"{data_dir}/test_param.npy", path=f"{dir}/test_param_hist.png")
    #plot_hist_adni(zX, path=f"{dir}/test_param_hist.png")

def test_plot_extrap(full_t, x_extrap, dir, data_dir):
    x_extrap = x_extrap.detach().cpu().numpy()[0]
    plot_extrap(full_t, x_extrap, param_path=f"{data_dir}/test_param.npy", path=f"{dir}/test_extrap.png")
    #plot_extrap_adni(full_t, x_extrap, path=f"{dir}/test_extrap.png")

def val_plot_recon(x, x_t, x_recon, zX, zR, dir):
    x = x.detach().cpu().numpy()[0]  # first sample
    x_t = x_t.detach().cpu().numpy()[0]
    x_recon = x_recon.detach().cpu().numpy()[0]

    plot_recon(x, x_t, x_recon, path=f"{dir}/val_recon.png")

