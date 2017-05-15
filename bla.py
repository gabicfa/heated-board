import numpy as np
import matplotlib.pyplot as plt


def calc_item(list_prev, i, j, mult_lambda):
    return mult_lambda*(list_prev[i + 1][j]
                        + list_prev[i - 1][j]
                        + list_prev[i][j + 1]
                        + list_prev[i][j - 1]) + ((1 - 4 * mult_lambda) * list_prev[i][j])


def create_bar(temp_top,
               temp_bottom,
               temp_left,
               temp_right,
               temp_init,
               col,
               row):
    list_main = [[temp_init]*(col + 1)]*(row + 1)
    list_main[0] = [temp_top] * (col + 1)
    list_main[-1] = [temp_bottom] * (col + 1)
    for i in range(row + 1):
        list_main[i][0] = temp_left
        list_main[i][-1] = temp_right
    return list_main


def get_temps_in_time(list_prev, time, mult_lambda):
    list_current = [x[:] for x in list_prev]
    for p in range(time):
        for i in range(1, len(list_prev) - 1):
            for j in range(1, len(list_prev[i]) - 1):
                list_current[i][j] = calc_item(list_prev, i, j, mult_lambda)
        list_prev = [x[:] for x in list_current]
    return list_current


def calc_mult_lambda(alpha, d_x, d_t):
    return alpha * (d_t / ((d_x) ** 2))


def plot_color_gradients(gradient):
    print(type(gradient))
    print(type(gradient[0]))
    gradient = np.array(gradient)
    # print(gradient)
    print(type(gradient))
    print(type(gradient[0]))
    print(type(gradient[0][0]))
    gradient = np.delete(gradient, [0, len(gradient)], 0)
    gradient = np.delete(gradient, [0, len(gradient)], 1)
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.9, bottom=0, left=0, right=0.99)
    ax.set_title('Titulo', fontsize=14)
    ax.imshow(gradient, aspect='equal', cmap=plt.get_cmap('magma'))
    ax.set_axis_off()


def main():
    list_prev = create_bar(100, 0, 0, 0, 0, 10, 10)
    mult_lambda = calc_mult_lambda(1, 0.1, 0.001)
    print(mult_lambda)
    list_current = get_temps_in_time(list_prev, 1000, mult_lambda)
    [print(_list) for _list in list_current]
    print(len( list_current ))
    print(len( list_current[0] ))
    plot_color_gradients(list_current)
    plt.show()


if __name__ == "__main__":
    main()
