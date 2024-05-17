import matplotlib.pyplot as plt
import numpy as np


def bar_plot(ax, x, y, title, x_label, y_label):
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def indep_var_scatter_plot(indep_vars, dep_var, dummy, title, text):
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 10))

    for ax, col in zip(axes.flatten(), indep_vars.columns):
        ax.scatter(indep_vars[col][dummy == 0], dep_var[dummy == 0], c='blue', label='0')
        ax.scatter(indep_vars[col][dummy == 1], dep_var[dummy == 1], c='red', label='1')
        ax.set_title(col)
        ax.legend()

    fig.suptitle(title, fontsize=16)
    fig.text(0.5, 0.92, text, ha='center', fontsize=12)
    plt.show()


def line_plot(ax, x, y, title, x_label, y_label, y_lim=None):
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_lim is not None:
        ax.set_ylim(y_lim)


def lasso_weights_plot(weights, lambdas, column_names, title, x_label, y_label):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

    for i in range(1, len(weights[0]) - 1):
        axes.plot(lambdas, [w[i] for w in weights], label=column_names[i])

    axes.set_title(title, fontsize=18)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.legend()
    plt.show()

def plot_mse_comparisons(mse_ols_original, mse_ols_missing, mse_ols_imputed,
                         mse_lasso_original, mse_lasso_missing, mse_lasso_imputed,
                         mse_ridge_original, mse_ridge_missing, mse_ridge_imputed,
                         lambdas_lasso, lambdas_ridge, ylim, missing_percentage):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))

    # Plot settings
    labels = ['Original Matrix', 'Missing values replaced with 0s', 'Missing values imputed']

    # Unconstrained Least Squares MSE
    axes[0].plot(mse_ols_original * np.ones_like(lambdas_lasso), 'r--', label=labels[0])
    axes[0].plot(mse_ols_missing * np.ones_like(lambdas_lasso), 'g', label=labels[1])
    axes[0].plot(mse_ols_imputed * np.ones_like(lambdas_lasso), 'b', label=labels[2])
    axes[0].set_title('Unconstrained Least Squares MSE')
    axes[0].set_ylabel('MSE')
    axes[0].set_ylim(ylim)

    # Lasso MSE
    axes[1].plot(lambdas_lasso, mse_lasso_original, 'r--', label=labels[0])
    axes[1].plot(lambdas_lasso, mse_lasso_missing, 'g', label=labels[1])
    axes[1].plot(lambdas_lasso, mse_lasso_imputed, 'b', label=labels[2])
    axes[1].set_title('Lasso Regression MSE for each λ')
    axes[1].set_xlabel('λ')
    axes[1].set_ylabel('MSE')
    axes[1].set_ylim(ylim)

    # Ridge MSE
    axes[2].plot(lambdas_ridge, mse_ridge_original, 'r--', label=labels[0])
    axes[2].plot(lambdas_ridge, mse_ridge_missing, 'g', label=labels[1])
    axes[2].plot(lambdas_ridge, mse_ridge_imputed, 'b', label=labels[2])
    axes[2].set_title('Ridge Regression, MSE for each λ')
    axes[2].set_xlabel('λ')
    axes[2].set_ylabel('MSE')
    axes[2].set_ylim(ylim)

    # Add a figure title
    fig.suptitle(f'MSE Comparison for {missing_percentage}% Missing Values', fontsize=19)

    # Set a single legend for the whole figure, below the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)

    # Adjust the layout to make room for the legend
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
