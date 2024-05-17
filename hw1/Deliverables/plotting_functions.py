
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def plot_energy_singvals(s, energy, index):

    # create a 2 plot figure
    fig = plt.figure(figsize=(15, 5))

    #plot the cumulative energy of the singular values
    ax1 = fig.add_subplot(121)
    ax1.plot(range(1, len(energy) + 1), energy, marker='o', markersize=3, linestyle='--', color='b')
    ax1.set_xlabel('Number of Singular Values')
    ax1.set_ylabel('Cumulative Energy')
    ax1.set_title('Scree Plot')

    # plot the elbow point
    ax1.plot(index, energy[index - 1], marker='o', markersize=7, color='r')
    #draw a line from elbow point to the x-axis
    ax1.plot([index, index], [0, energy[index - 1]], linestyle='--', color='r')

    # plot the singular values, on the right of the figure
    ax2 = fig.add_subplot(122)
    ax2.plot(s, marker='o', linestyle='-', color='b', markersize=4)
    ax2.set_title('Singular Values')
    ax2.set_xlabel('Number of Singular Values')
    ax2.set_ylabel('Singular Values')

    #plot the elbow point
    ax2.plot(index, s[index - 1], marker='o', markersize=7, color='r')
    #draw a line from elbow point to the x-axis
    ax2.plot([index, index], [0, s[index - 1]], linestyle='--', color='r')

    plt.tight_layout()
    plt.show()

def plot_U_V(U, V, cmap='seismic'):

    # create a 2 plot figure, to plot U, Vt
    fig = plt.figure(figsize=(15, 5))

    # plot U
    ax1 = fig.add_subplot(121)
    ax1.imshow(U, cmap=cmap, interpolation='none', aspect='auto')
    ax1.set_xlabel('Latent Features')
    ax1.set_ylabel('Users')
    ax1.set_title('User-Latent Feature Matrix')



    # plot Vt
    ax2 = fig.add_subplot(122)
    ax2.imshow(V, cmap=cmap, interpolation='none', aspect='auto')
    ax2.set_xlabel('Latent Features')
    ax2.set_ylabel('Movies')
    ax2.set_title('Movie-Latent Feature Matrix')

    # add a single colorbar for both
    plt.colorbar(ax1.imshow(U, cmap=cmap, interpolation='none', aspect='auto'), ax=[ax1, ax2])

    plt.show()


# function takes as input 3 dictionaries
# each dictionary has 4 things: array, title, axis1_name, axis2_name
# plots them with the respective cmap and colorbar
def plot_heatmaps(info1, info2, info3, cmap):

    fig = plt.figure(figsize=(21, 7))

    # plot first
    ax1 = fig.add_subplot(131)
    ax1.imshow(info1['array'], cmap=cmap, interpolation='nearest', aspect='auto')
    ax1.set_xlabel(info1['axis1_name'])
    ax1.set_ylabel(info1['axis2_name'])
    ax1.set_title(info1['title'])
    plt.colorbar(ax1.imshow(info1['array'], cmap=cmap, interpolation='nearest', aspect='auto'))

    # plot second
    ax2 = fig.add_subplot(132)
    ax2.imshow(info2['array'], cmap=cmap, interpolation='nearest', aspect='auto')
    ax2.set_xlabel(info2['axis1_name'])
    ax2.set_ylabel(info2['axis2_name'])
    ax2.set_title(info2['title'])
    plt.colorbar(ax2.imshow(info2['array'], cmap=cmap, interpolation='nearest', aspect='auto'))

    # plot third
    ax3 = fig.add_subplot(133)
    ax3.imshow(info3['array'], cmap=cmap, interpolation='nearest', aspect='auto')
    ax3.set_xlabel(info3['axis1_name'])
    ax3.set_ylabel(info3['axis2_name'])
    ax3.set_title(info3['title'])
    plt.colorbar(ax3.imshow(info3['array'], cmap=cmap, interpolation='nearest', aspect='auto'))

    plt.show()


def plot_scatters(matrix1, matrix2, title1, title2):

    # create a 4 plot figure
    fig = plt.figure(figsize=(15, 10))

    # plot 1 column of matrix1
    ax1 = fig.add_subplot(221)
    ax1.scatter(matrix1[:, 0], np.zeros(matrix1.shape[0]), marker='o', color='b', alpha=0.8, s=5)
    ax1.set_title(title1)


    # plot 2 column of matrix1
    ax2 = fig.add_subplot(222)
    ax2.scatter(matrix1[:, 0], matrix1[:, 1], marker='o', color='b', alpha=0.8, s=5)
    ax2.set_title(title1)

    # plot 1 column of matrix2
    ax3 = fig.add_subplot(223)
    ax3.scatter(matrix2[:, 0], np.zeros(matrix2.shape[0]), marker='o', color='r', alpha=0.8, s=5)
    ax3.set_title(title2)

    # plot 2 column of matrix2
    ax4 = fig.add_subplot(224)
    ax4.scatter(matrix2[:, 0], matrix2[:, 1], marker='o', color='r', alpha=0.8, s=5)
    ax4.set_title(title2)

    plt.tight_layout()
    plt.show()


def plotSimilarMovies(ratings, index_1, index_2):
    """
    Calculate Spearman correlation for the ratings of two users and plot the difference in ratings for common movies.

    Parameters:
    - ratings (numpy.ndarray): Ratings matrix.
    - index_1 (int): Index of the first user.
    - index_2 (int): Index of the second user.

    Returns:
    - None

    Example:
    plotSimilarMovies(1, 2)
    """
    # Calculate Spearman correlation for users
    user1 = ratings[index_1]
    user2 = ratings[index_2]

    # Calculate and plot the difference of the ratings of movies the users both rated
    intersection = np.intersect1d(np.nonzero(user1), np.nonzero(user2))

    user1_inter = user1[intersection]
    user2_inter = user2[intersection]

    difference = user1_inter - user2_inter

    # Find indexes where difference <= 0.5
    indexes = np.where(np.abs(difference) <= 0.5)[0]
    print(f"Ratio of movies with difference smaller than 0.5: {100 * indexes.shape[0] / difference.shape[0]}%")

    # Plot the difference
    fig = plt.figure(figsize=(15, 5))

    # Draw a plot of the 2 users' ratings
    ax1 = fig.add_subplot(121)

    # Draw a line plot
    ax1.plot(user1_inter, marker='o', linestyle='-', color='b', markersize=4, label=f"User {index_1}", alpha=0.8)
    ax1.plot(user2_inter, marker='o', linestyle='-', color='r', markersize=4, label=f"User {index_2}", alpha=0.8)
    ax1.set_xlabel('Movies')
    ax1.set_ylabel('Ratings')
    ax1.set_title(f"Ratings of User {index_1} and User {index_2}")
    # Set y-axis to 0-5
    ax1.set_ylim([0, 6])
    ax1.legend()

    # Plot the difference
    ax2 = fig.add_subplot(122)

    # Draw a bar plot of the difference
    ax2.bar(np.arange(len(difference)), difference, color='b')
    ax2.set_xlabel('Movies')
    ax2.set_ylabel('Difference in Ratings')
    ax2.set_title(f"Difference in Ratings between User {index_1} and User {index_2}")
    ax2.set_ylim([-5, 5])

    plt.show()


def plot_graphs(graphs, corr_threshold_arr, val_index, train_color='blue', val_color='red', node_sz=[10, 10, 15], edge_width=[0.8, 0.3, 0.2]):


    # Create a 2x3 grid of subplots for the first 6 graphs
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))

    # first 6 graphs in 2 rows
    for i, G in enumerate(graphs[:6]):
        # Calculate subplot position based on index
        row = i // 3
        col = i % 3

        # Create a list of colors for each node based on the type (validation or training)
        node_colors = [val_color if node in val_index else train_color for node in G.nodes]

        # Plot the graph on the current subplot
        nx.draw(G, ax=axes1[row, col], pos=nx.spring_layout(G, seed=1969), node_size=node_sz[0], node_color=node_colors, edge_color='gray', width=edge_width[0])
        axes1[row, col].set_title(f'Corr Threshold = {corr_threshold_arr[i]:.2f}')

    plt.tight_layout()
    plt.show()

    # Create a 1x2 grid of subplots for the next 2 graphs
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8))

    # next 2 graphs in one row
    for i, G in enumerate(graphs[6:8]):
        # Calculate subplot position based on index
        col = i

        # Create a list of colors for each node based on the type (validation or training)
        node_colors = [val_color if node in val_index else train_color for node in G.nodes]

        # Plot the graph on the current subplot
        nx.draw(G, ax=axes2[col], pos=nx.spring_layout(G, seed=1970), node_size=node_sz[1], node_color=node_colors, edge_color='gray', width=edge_width[1])
        axes2[col].set_title(f'Corr Threshold = {corr_threshold_arr[i+6]:.2f}')

    plt.show()

    # Create a standalone subplot for the last graph
    fig3, ax3 = plt.subplots(figsize=(20, 10))

    # last graph
    G = graphs[8]

    # Create a list of colors for each node based on the type (validation or training)
    node_colors = [val_color if node in val_index else train_color for node in G.nodes]

    # Plot the graph on the current subplot
    nx.draw(G, ax=ax3, pos=nx.spring_layout(G, seed=1256), node_size=node_sz[2], node_color=node_colors, edge_color='gray', width=edge_width[2])

    ax3.set_title(f'Corr Threshold = {corr_threshold_arr[8]:.2f}')

    plt.show()


def plot_graphs_info(results, threshold_arr, train_color='blue', val_color='red'):

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 5))

    # Set the width of the bars
    bar_width = 0.25

    # Calculate the x-axis positions for the bars
    x = np.arange(len(results))

    # set colors
    # index 0 is blue, 1 is red, rest is whatever
    colors = ['b', 'r']
    colors.extend(['g'] * (len(results) - 2))

    # Plot the bars for the number of nodes
    ax.bar(x - bar_width, results[:, 0] - results[:, 1], width=bar_width, label='Number of Training Nodes', color=train_color)
    ax.bar(x - bar_width, results[:, 1], width=bar_width, label='Number of Validation Nodes', bottom=results[:, 0] - results[:, 1], color= val_color)

    # Plot the bars for the number of articulation points
    ax.bar(x, results[:, 3] - results[:, 4], width=bar_width, label='Number of Train Articulation Points', color='cornflowerblue')
    ax.bar(x, results[:, 4], width=bar_width, label='Number of Validation Articulation Points', bottom=results[:, 3] - results[:, 4], color='lightsalmon')

    # Plot the bars for the number of components
    ax.bar(x + bar_width, results[:, 2], width=bar_width, label='Number of Components', color='greenyellow')


    # Set labels, title, and legend
    ax.set_xlabel('Correlation Threshold')
    ax.set_ylabel('Count')
    ax.set_title('Number of Nodes, Components, and Articulation Points in Each Graph')
    ax.set_xticks(x)
    ax.set_xticklabels(threshold_arr)
    ax.legend()

    # Show the plot
    plt.show()


# plot the graph

def plotGraph(G, val_index, node_sz= 50, val_color='tomato', train_color='deepskyblue'):

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)

    # Create a list of colors for each node based on the type (validation or training)
    node_colors = [val_color if node in val_index else train_color for node in G.nodes]

    nx.draw(G, ax=ax, pos=nx.spring_layout(G, seed=1969), node_size=node_sz, node_color=node_colors, edge_color='gray', with_labels=True, font_size=7, font_color='k', width=0.4)

    # add a legend
    ax.plot([], [], 'o', color='tomato', label='Validation User')
    ax.plot([], [], 'o', color='deepskyblue', label='Train User')
    ax.legend()

    plt.show()