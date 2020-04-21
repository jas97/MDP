import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


def visualize_val_function(val_func):
    '''
    Visualizes the passed value function
    :param val_func: array of float values
    '''
    val_func_mat = convert_array_to_matrix(val_func)

    fig, ax = plt.subplots()
    ax.set_title("Grid world value function")
    ax.matshow(val_func_mat, cmap='seismic')
    add_values_to_plot(val_func_mat, ax)
    plt.show()

def add_values_to_plot(mat, ax) :
    '''
    Adds values to each cell on the plot
    :param mat: matrix of values to print on the plot
    :param ax: plot
    :return:
    '''
    for (i, j), val in np.ndenumerate(mat):
        label = "NA" if np.isnan(val) else val
        ax.text(j, i, '{}'.format(label), ha='center', va='center',fontweight='bold', color='gray')

def convert_array_to_matrix(arr):
    '''
    Converts an array into matrix format
    :param arr: array
    :return: matrix obtained from the array
    '''
    dim = int(np.sqrt(len(arr)))
    mat = np.ndarray((dim, dim))
    for i in range(dim):
        for j in range(dim):
            mat[i, j] = arr[i*dim + j]
    return mat

def visualize_reward_shaping(val_func):
    '''
    Visualizes the pseudo-rewards obtained by reward-shaping method
    :param val_func: value function
    :return:
    '''
    dim = int(np.sqrt(len(val_func)))
    # calculate reward shaping function F for each action and add it to the standard reward(-1)
    left_reward = [get_left_reward(val_func, i) for i in range(len(val_func))]
    right_reward = [get_right_reward(val_func, i) for i in range(len(val_func))]
    up_reward = [get_up_reward(val_func, i) for i in range(len(val_func))]
    down_reward = [get_down_reward(val_func, i) for i in range(len(val_func))]

    rewards = [left_reward, right_reward, up_reward, down_reward]
    reward_matrices = [convert_array_to_matrix(arr) for arr in rewards]

    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    titles = ['Reward shaping -- left',
              'Reward shaping -- right',
              'Reward shaping -- up',
              'Reward shaping -- down']

    for i in range(4):
        ax = plt.Subplot(fig, outer[i])
        ax.matshow(reward_matrices[i], cmap='seismic')
        ax.set_title(titles[i])
        add_values_to_plot(reward_matrices[i], ax)
        fig.add_subplot(ax)

    plt.show()


def get_left_reward(arr, ind):
    '''
    Calculates the pseudo-reward for state with index ind, and action 'left'
    :param arr: value function
    :param ind: index of state
    :return: pseudo-reward
    '''
    dim = int(np.sqrt(len(arr)))
    left = ind - 1

    if left < 0:
        return None
    if ind % dim == 0:
        return None  # cannot move left

    return -1 + arr[left] - arr[ind]

def get_right_reward(arr, ind):
    '''
    Calculates the pseudo-reward for state with index ind, and action 'right'
    :param arr: value function
    :param ind: index of state
    :return: pseudo-reward
    '''
    dim = int(np.sqrt(len(arr)))
    right = ind + 1

    if right < 0:
        return None
    if right % dim == 0:
        return None

    return -1 + arr[right] - arr[ind]

def get_up_reward(arr, ind) :
    '''
    Calculates the pseudo-reward for state with index ind, and action 'up'
    :param arr: value function
    :param ind: index of state
    :return: pseudo-reward
    '''
    dim = int(np.sqrt(len(arr)))
    up = ind - dim

    if up < 0:
        return None
    return -1 + arr[up] - arr[ind]

def get_down_reward(arr, ind):
    '''
    Calculates the pseudo-reward for state with index ind, and action 'down'
    :param arr: value function
    :param ind: index of state
    :return: pseudo-reward
    '''
    dim = int(np.sqrt(len(arr)))
    down = ind + dim

    if down >= dim * dim:
        return None

    return -1 + arr[down] - arr[ind]


def visualize_optimal_policy(val_function):
    '''
    Visualizes the optimal policy with given value function
    :param val_function: value function
    :return:
    '''
    dim = int(np.sqrt(len(val_function)))
    val_func_matrix = convert_array_to_matrix(val_function)

    fig, ax = plt.subplots()
    ax.set_title("Grid world optimal policy")
    ax.matshow(val_func_matrix,  cmap='seismic')
    field_centers = [(i, j) for j in range(dim) for i in range(dim)]
    print(len(field_centers))
    for i, fc in enumerate(field_centers):
        arrow_size = 0.4
        directions = get_optimal_directions(i, val_function)
        end_points = []
        if 'up' in directions:
            end_points.append((fc[0], fc[1] - arrow_size))
        if 'down' in directions:
            end_points.append((fc[0], fc[1] + arrow_size))
        if 'left' in directions:
            end_points.append((fc[0] - arrow_size, fc[1]))
        if 'right' in directions:
            end_points.append((fc[0] + arrow_size, fc[1]))
        for j, d in enumerate(directions):
            ax.annotate("", xy=end_points[j], xytext=fc,
                            arrowprops = dict(arrowstyle="->", color='black'))

    plt.tight_layout()
    plt.show()

def get_optimal_directions(ind, val_function):
    '''
    Calculates the optimal action at state ind
    :param ind: index of state
    :param val_function: value function
    :return:
    '''
    left = get_left_reward(val_function, ind)
    right = get_right_reward(val_function, ind)
    up = get_up_reward(val_function, ind)
    down = get_down_reward(val_function, ind)
    directions = []
    if left is not None and left >= 0:
        directions.append("left")
    if right is not None and right >= 0:
        directions.append('right')
    if up is not None and up >= 0:
        directions.append('up')
    if down is not None and down >= 0:
        directions.append('down')

    return directions