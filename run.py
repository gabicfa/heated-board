import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pylab


ERROR = 0.0002
LEFT_TEMP = 0.0
RIGHT_TEMP = 0.0
BOT_TEMP = 0.0
TOP_TEMP = 100.0
BAR_TEMP = 0
ALPHA = 1
D_X = 0.1
D_Y = 0.1
D_T = 10 ** (-3)

class TempCalculator2D:

    def __init__(self, len_x, len_y, x_step, y_step, t_step, alpha):
        self.x_step, self.y_step = x_step, y_step
        self.n_steps = int(len_x / x_step)
        self.len_x, self.len_y = len_x, len_y
        self.t_step = t_step
        self.f_0 = alpha * t_step / (x_step ** 2)
        self.create_matrix()
        self.temp_board = [self.board]
        self.error = [0]

    def create_matrix(self):
        self.board = []
        arr = []
        arr.append([TOP_TEMP for x in range(self.n_steps + 1)])
        self.board.append([TOP_TEMP] * (self.n_steps + 1))
        for i in range(self.n_steps - 1):
            # miolo
            arr = [LEFT_TEMP]  # border
            for j in range(self.n_steps - 1):
                arr.append(BAR_TEMP)
            arr.append(RIGHT_TEMP)  # border
            self.board.append(arr)
        self.board.append([BOT_TEMP for x in range(self.n_steps + 1)])

    def calculate_temp_2d(self, time_target, error_target):
        curr_id = len(self.temp_board) - 1
        if curr_id < time_target:
            for j in range(curr_id, time_target):
                if self.calc(j, error_target):
                    time_target = j
                    break
        return self.temp_board[time_target], self.error[time_target]

    def calc(self, j, error_target):
        _board = self.temp_board[j]
        board = [[TOP_TEMP] * (self.n_steps + 1)]
        for i in range(1, self.n_steps):
            arr = [LEFT_TEMP]
            max_error = -1
            for z in range(1, self.n_steps - 1):
                t_ij = self.f_0 * (_board[i + 1][z] + _board[i - 1][z] +
                                   _board[i][z + 1] + _board[i][z - 1]) +\
                                   (1 - 4 * self.f_0) * _board[i][z]
                # t_ij > 0 to avoid division by zero
                if t_ij > 0:
                    curr_error = abs((t_ij - _board[i][z]) / t_ij)
                    if curr_error > max_error:
                        max_error = curr_error
                arr.append(t_ij)
            arr.append(RIGHT_TEMP)
            board.append(arr)
            self.error.append(max_error if max_error > 0 else 0)
        board.append([BOT_TEMP] * (self.n_steps + 1))
        self.temp_board.append(board)
        return error_target <= max_error

class TempCalculator:

    def __init__(self, bar_len, bar_step, time_step, alpha):
        self.bar_len = bar_len
        self.bar_step = bar_step  # h
        self.n_steps = bar_len // bar_step
        self.time_step = time_step
        self.alpha = alpha
        self.lamb = alpha * self.time_step / (self.bar_step ** 2)
        self.gen_array()
        self.temp_array = [self.bar_array]  # copy by value


    def gen_array(self):
        self.bar_array = []
        self.bar_array.append(BORDER_TEMP)
        for i in range(self.n_steps - 2):
            # steps - 1 because gabi said soo
            self.bar_array.append(BAR_TEMP)
        self.bar_array.append(BORDER_TEMP)


    def calc_temp_1d(self, time_target):
        curr_id = len(self.temp_array) - 1
        if curr_id < time_target:
            for j in range(curr_id, time_target):
                _list = [BORDER_TEMP]
                for i in range(1, self.n_steps - 1):
                    u_i = self.temp_array[j][i] + self.lamb * \
                        (self.temp_array[j][i + 1] - 2 * self.temp_array[j][i] +
                         self.temp_array[j][i - 1])
                    _list.append(u_i)
                _list.append(BORDER_TEMP)
                self.temp_array.append(_list)
        return self.temp_array[time_target]



# fig, ax = plt.subplots()
# x = range(10)
# line, = ax.plot(x, temps[0])
temps = []

# def animate(i):
#     line.set_ydata(temps[i])
#     return line,

# def init():
#     line.set_ydata(temps[0])
#     return line,

def plot_color_gradients(gradient):
    """ Straight up stolen from Dias' group """
    # gradient = np.array(gradient)
    # gradient = [np.array(line) for line in gradient]
    print(type(gradient))
    print(type(gradient[0]))
    gradient = np.array(gradient)
    # print(gradient)
    print(type(gradient))
    print(type(gradient[0]))
    gradient = np.delete(gradient, len(gradient) - 1, 0)
    gradient = np.delete(gradient, len(gradient[0]) - 1, 1)
    fig, ax = plt.subplots()
    # fig.subplots_adjust(top=0.9, bottom=0, left=0, right=0.99)
    # ax.set_title('Titulo', fontsize=14)
    ax.imshow(gradient, aspect='equal', cmap=plt.get_cmap('magma'))
    ax.set_axis_off()

def plot(data):
    # data = np.array([np.array(d, dtype=float) for d in data])
    print(len(data), len(data[0]))
    ndata = np.zeros((len(data), len(data[0])))
    print(len(ndata), len(ndata[0]))
    for i in range(len(ndata)):
        for j in range(len(ndata[0])):
            ndata[i, j] = data[i][j]
    print(type(data))
    print(type(data[0]))
    pylab.pcolor(ndata)
    pylab.colorbar()
    pylab.show()
    # fig, axes = plt.subplots(nrows=len(data[0]))
    # fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)

    # for ax in axes:
    #     ax.imshow(data, aspect='auto', cmap=plt.get_cmap('inferno'))
    #     ax.set_axis_off()

calculator = TempCalculator2D(1, 1, D_X, D_Y, D_T, ALPHA)
for i in range(10, 501, 10):
    m = calculator.calculate_temp_2d(i, ERROR)
    temps.append(m[0])  # m[1] represents it's error

# ani = animation.FuncAnimation(fig, animate, np.arange(1, 10), init_func=init,
#                               interval=800, blit=True)
# plot_color_gradients(m[0])
plot_color_gradients(temps[len(temps) - 1])
plt.show()
