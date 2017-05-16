from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


K = 1
ERROR = 0.0002
LEFT_TEMP = 75.0
RIGHT_TEMP = 50.0
BOT_TEMP = 0.0
TOP_TEMP = 100.0
BAR_TEMP = 0
ALPHA = 1
D_X = .1
D_Y = .1
D_T = 0.001
FLUX_TOP = None
FLUX_LEFT = None
FLUX_RIGHT = None
FLUX_BOT = None

class TempCalculator2D:

    def __init__(self, len_x, len_y, x_step, y_step, t_step, alpha, k,
                 flux_top, flux_left, flux_right, flux_bot):
        self.x_step, self.y_step = x_step, y_step
        self.n_steps = int(len_x / x_step)
        self.len_x, self.len_y = len_x, len_y
        self.t_step = t_step
        self.f_0 = alpha * t_step / (x_step ** 2)
        self.create_matrix()
        self.temp_board = [self.board]
        self.error = [0]
        self.k = k
        self.flux_top = flux_top
        self.flux_left = flux_left
        self.flux_right = flux_right
        self.flux_bot = flux_bot

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


    def calculate_flux_2d(self, time_target, error_target):
        curr_id = len(self.temp_board) - 1
        if curr_id < time_target:
            _board, _ = self.calculate_temp_2d(time_target, error_target)
        else:
            _board = self.temp_board[time_target]

        flux = []
        for i in range(self.n_steps):
            line = []
            for j in range(self.n_steps):
                flux_x = -self.k * (_board[i + 1][j] - _board[i - 1][j]) /\
                         (2 * self.x_step)
                flux_y = -self.k * (_board[i + 1][j] - _board[i - 1][j]) /\
                         (2 * self.x_step)
                line.append(sqrt(flux_x ** 2 + flux_y ** 2))
            flux.append(line)
        return flux, error_target


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
        flux, _ = self.calculate_flux_2d(j - 1, error_target)
        top = [TOP_TEMP]
        for i in range(1, self.n_steps):
            if self.flux_top != None:
                tmp = self.f_0 * (2 * _board[1][i] - 2 * self.y_step * self.flux_top +
                                _board[0][i + 1] + _board[0][i - 1]) + \
                                (1 - 4 * self.f_0) * _board[0][i]
            else:
                tmp = TOP_TEMP
            top.append(tmp)
        top.append(TOP_TEMP)
        board = [top]
        for i in range(1, self.n_steps):
            if self.flux_left != None:
                left = self.f_0 * (2 * _board[i][1] - 2 * self.x_step * self.flux_left +
                                   _board[i + 1][0] + _board[i - 1][0]) + \
                                   (1 - 4 * self.f_0) * _board[i][0]
            else:
                left = LEFT_TEMP
            arr = [left]
            max_error = -1
            for z in range(1, self.n_steps):
                t_ij = self.f_0 * (_board[i + 1][z] + _board[i - 1][z] +
                                   _board[i][z + 1] + _board[i][z - 1]) +\
                                   (1 - 4 * self.f_0) * _board[i][z]
                # t_ij > 0 to avoid division by zero
                if t_ij > 0:
                    curr_error = abs((t_ij - _board[i][z]) / t_ij)
                    if curr_error > max_error:
                        max_error = curr_error
                arr.append(t_ij)

            if self.flux_right != None:
                z = self.n_steps
                right = self.f_0 * (2 * _board[i][z] - 2 * self.x_step *
                                    self.flux_right + _board[i + 1][z] +
                                    _board[i - 1][z]) + \
                                    (1 + 4 * self.f_0) * _board[i][z]
            else:
                right = RIGHT_TEMP
            arr.append(right)
            board.append(arr)
            self.error.append(max_error if max_error > 0 else 0)

        bot = [BOT_TEMP]
        z = self.n_steps
        for i in range(1, self.n_steps):
            if self.flux_bot != None:
                tmp = self.f_0 * (2 * _board[z - 1][i] - 2 * self.y_step *
                                  self.flux_bot + _board[z][i + 1] +
                                  _board[z][i - 1]) + \
                                  (1 + 4 * self.f_0) * _board[z][i]
            else:
                tmp = BOT_TEMP
            bot.append(tmp)
        bot.append(BOT_TEMP)
        board.append(bot)
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


def plot_color_gradients(gradient):
    final = np.zeros((len(gradient), len(gradient[0])))
    for x in range(len(gradient)):
        for y in range(len(gradient[0])):
            final[x, y] = gradient[x][y]
    ax.imshow(final, aspect='equal', cmap=plt.get_cmap('hot'))

calculator = TempCalculator2D(.4, .4, D_X, D_Y, D_T, ALPHA, K,
                              FLUX_TOP, FLUX_LEFT, FLUX_RIGHT, FLUX_BOT)
temps = []
for i in range(1, 201, 10):
    m = calculator.calculate_temp_2d(i, ERROR)
    temps.append(m[0])  # m[1] represents it's error


fig, ax = plt.subplots()
fig.subplots_adjust(top=0.9, bottom=0, left=0, right=0.99)
ax.set_title('Titulo', fontsize=14)
ax.set_axis_off()

def animate(i):
    plot_color_gradients(temps[i])

def init():
    plot_color_gradients(temps[0])


init()
ani = animation.FuncAnimation(fig, animate, np.arange(1, len(temps)),
                              interval=200, repeat=True)

print(temps[-1])
# print(calculator.calculate_flux_2d(200, ERROR)[0])
plt.show()
