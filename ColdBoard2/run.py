from math import sqrt
import math

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LEFT_TEMP = 75.0
RIGHT_TEMP = 75.0
BOT_TEMP = 0.0
TOP_TEMP = 100.0

FLUX_TOP = 0
FLUX_LEFT = 0
FLUX_RIGHT = None
FLUX_BOT = 0

T_TOTAL = 10
T_STEP = 0.001

LEN_X = 0.5
LEN_Y = 0.5
LEN_STEP = 0.1

MIN_MULT = .9
MAX_MULT = 1.1

K = 1
C = 1
D = 1
ALPHA = K/(C*D)
ERROR = 0.0


class BorderTemp:
    def __init__(self, temp, temp_var=None, starting_angle=0, angle_inc=.3):
        self.temp = temp
        self.angle = starting_angle  # radians
        self.angle_inc = angle_inc
        self.temp_var = temp_var if temp_var is not None else temp * .2

    def get_new_temp(self):
        self.angle += self.angle_inc
        return self.temp + math.sin(self.angle) * self.temp_var


class TempCalculator2D:

    def __init__(self, len_x, len_y, len_step, t_total, t_step, alpha,
                 k, flux_top, flux_left, flux_right, flux_bot):
        self.len_step = len_step
        self.len_x, self.len_y = len_x, len_y
        self.matrix_x = int(len_x/len_step)
        self.matrix_y = int(len_y/len_step)
        self.t_step = t_step
        self.t_total = t_total
        self.f_0 = alpha * t_step / (len_step ** 2)
        print(self.f_0)
        if self.f_0 > 0.25:
            print("f0 muito grande")
        self.k = k
        self.flux_top = flux_top
        self.flux_left = flux_left
        self.flux_right = flux_right
        self.flux_bot = flux_bot
        self.border_calc = BorderTemp(RIGHT_TEMP)
        self.create_matrix_inicial()
        self.matrix_temps()
        self.calculate_flux()

    def create_matrix_inicial(self):
        self.board = [[0 for i in range(self.matrix_x)]
                      for j in range(self.matrix_y)]

        if(self.flux_top is not None):
            for i in range(1, self.matrix_x-1):
                self.board[0][i] = TOP_TEMP

        else:
            for i in range(0, self.matrix_x):
                self.board[0][i] = TOP_TEMP

        if(self.flux_bot is not None):
            for i in range(1, self.matrix_x-1):
                self.board[self.matrix_y - 1][i] = BOT_TEMP

        else:
            for i in range(0, self.matrix_x):
                self.board[self.matrix_y - 1][i] = BOT_TEMP

        if(self.flux_left is not None):
            for i in range(1, self.matrix_y-1):
                self.board[i][0] = LEFT_TEMP

        else:
            for i in range(0, self.matrix_y):
                self.board[i][0] = LEFT_TEMP

        if(self.flux_right is not None):
            for i in range(1, self.matrix_y-1):
                self.board[i][self.matrix_x - 1] = RIGHT_TEMP

        else:
            for i in range(0, self.matrix_y):
                self.board[i][self.matrix_x - 1] = RIGHT_TEMP

        for i in self.board:
            print(i)
            print('\n')

    def matrix_temps(self):
        self.arr_temps = []

        curr_err = 10
        if self.t_total < 1:
            self.t_total *= 10000
            self.t_step *= 10000

        n_temps = int(self.t_total/self.t_step)
        self.arr_temps.append(self.board)

        for t in range(1, n_temps):
            matrix_t1 = [[0 for i in range(self.matrix_x)]
                         for j in range(self.matrix_y)]
            temp_right = self.border_calc.get_new_temp()
            for i in range(0, self.matrix_x):
                for j in range(0, self.matrix_y):
                    if i == 0:
                        if FLUX_TOP is None:
                            matrix_t1[i][j] = self.arr_temps[t-1][i][j] * \
                                              rd.uniform(MIN_MULT, MAX_MULT)
                        else:
                            if j != 0 and j != self.matrix_x-1:
                                matrix_t1[i][j] = self.f_0*(
                                    2*self.arr_temps[t-1][i+1][j] -
                                    2*self.len_step*self.flux_top +
                                    self.arr_temps[t-1][i][j+1] +
                                    self.arr_temps[t-1][i][j-1]) + \
                                    (1-4*self.f_0)*self.arr_temps[t-1][i][j]

                    if j == 0:
                        if FLUX_LEFT is None:
                            matrix_t1[i][j] = self.arr_temps[t-1][i][j] * \
                                              rd.uniform(MIN_MULT, MAX_MULT)
                        else:
                            if i != 0 and i != self.matrix_x-1:
                                matrix_t1[i][j] = self.f_0*(
                                    2*self.arr_temps[t-1][i][j+1] -
                                    2*self.len_step*self.flux_left +
                                    self.arr_temps[t-1][i-1][j] +
                                    self.arr_temps[t-1][i+1][j]) + \
                                    (1-4*self.f_0)*self.arr_temps[t-1][i][j]

                    if i == self.matrix_y-1:
                        if FLUX_BOT is None:
                            matrix_t1[i][j] = self.arr_temps[t-1][i][j] * \
                                              rd.uniform(MIN_MULT, MAX_MULT)
                        else:
                            if i != j and j != 0:
                                matrix_t1[i][j] = self.f_0*(
                                    2*self.arr_temps[t-1][i-1][j] -
                                    2*self.len_step*self.flux_bot +
                                    self.arr_temps[t-1][i][j-1] +
                                    self.arr_temps[t-1][i][j+1]) + \
                                    (1-4*self.f_0)*self.arr_temps[t-1][i][j]

                    if j == self.matrix_x-1:
                        if FLUX_RIGHT is None:
                            matrix_t1[i][j] = temp_right
                        else:
                            if i != 0 and i != self.matrix_x-1:
                                matrix_t1[i][j] = self.f_0*(
                                    2*self.arr_temps[t-1][i][j-1] -
                                    2*self.len_step*self.flux_right +
                                    self.arr_temps[t-1][i+1][j] +
                                    self.arr_temps[t-1][i-1][j]) + \
                                    (1-4*self.f_0)*self.arr_temps[t-1][i][j]

                    if i != 0 and i != self.matrix_x-1:
                        if j != 0 and j != self.matrix_y-1:
                            matrix_t1[i][j] = self.f_0*(
                                self.arr_temps[t-1][i][j+1] +
                                self.arr_temps[t-1][i][j-1] +
                                self.arr_temps[t-1][i-1][j] +
                                self.arr_temps[t-1][i+1][j]) + \
                                (1-4*self.f_0)*self.arr_temps[t-1][i][j]

                    dif = abs(self.arr_temps[t-1][i][j] - matrix_t1[i][j])
                    if dif < curr_err and dif != 0:
                        curr_err = dif

            self.arr_temps.append(matrix_t1)

            if curr_err <= ERROR:
                break

    def calculate_flux(self):
        self.arr_flux = []
        for t in range(len(self.arr_temps)):
            matrix_f = [[0 for i in range(self.matrix_x)]
                        for j in range(self.matrix_y)]
            for i in range(0, self.matrix_x):
                for j in range(0, self.matrix_y):
                    flux_x = -1
                    flux_y = -1
                    if i == 0:
                        if FLUX_TOP is not None:
                            if j != 0 and j != self.matrix_x-1:
                                flux_x = -self.k*(
                                    self.arr_temps[t-1][i][j+1] -
                                    self.arr_temps[t-1][i][j-1]) / \
                                    (2*self.len_step)
                                flux_y = self.flux_top

                    if j == 0 and i != 0 and i != self.matrix_x-1:
                        if FLUX_LEFT is not None:
                            flux_y = -self.k*(
                                self.arr_temps[t-1][i+1][j] -
                                self.arr_temps[t-1][i-1][j]) / \
                                (2*self.len_step)
                            flux_x = self.flux_left

                    if i == self.matrix_y-1 and i != j and j != 0:
                        if FLUX_BOT is not None:
                            flux_x = -self.k*(
                                self.arr_temps[t-1][i][j+1] -
                                self.arr_temps[t-1][i][j-1]) / \
                                (2*self.len_step)
                            flux_y = self.flux_bot

                    if j == self.matrix_x-1 and i != 0 and \
                       i != self.matrix_y-1:
                        if FLUX_RIGHT is not None:
                            flux_y = -self.k*(
                                self.arr_temps[t-1][i+1][j] -
                                self.arr_temps[t-1][i-1][j]) / \
                                (2*self.len_step)
                            flux_x = self.flux_right

                    if i != 0 and i != self.matrix_x-1:
                        if j != 0 and j != self.matrix_y-1:
                            flux_x = -self.k*(
                                self.arr_temps[t-1][i][j+1] -
                                self.arr_temps[t-1][i][j-1]) / \
                                (2*self.len_step)
                            flux_y = -self.k*(
                                self.arr_temps[t-1][i+1][j] -
                                self.arr_temps[t-1][i-1][j]) / \
                                (2*self.len_step)

                    if(flux_x != -1 or flux_y != -1):
                        flux_p = (sqrt(flux_x**2 + flux_y**2))
                        matrix_f[i][j] = flux_p

            self.arr_flux.append(matrix_f)


MATRIX_TESTE = TempCalculator2D(LEN_X, LEN_Y, LEN_STEP, T_TOTAL, T_STEP, ALPHA,
                                K, FLUX_TOP, FLUX_LEFT, FLUX_RIGHT, FLUX_BOT)

for i in MATRIX_TESTE.arr_temps[-1]:
    print(i)
print('\n')


def plot_color_gradients(gradient):
    final = np.zeros((len(gradient), len(gradient[0])))
    for x in range(len(gradient)):
        for y in range(len(gradient[0])):
            final[x, y] = gradient[x][y]
    return ax.imshow(final, aspect='equal', cmap=plt.get_cmap('hot'),
                     vmin=0, vmax=100)


def animate(i):
    plot_color_gradients(MATRIX_TESTE.arr_temps[i])


def init():
    im = plot_color_gradients(MATRIX_TESTE.arr_temps[0])
    plt.colorbar(im)


fig, ax = plt.subplots()
fig.subplots_adjust(top=0.9, bottom=0, left=0, right=0.99)
ax.set_title('Temperatura na Placa ao Longo do Tempo', fontsize=14)
ax.set_axis_off()

init()

ani = animation.FuncAnimation(fig, animate,
                              np.arange(0,
                                        int(3*(len(MATRIX_TESTE.arr_temps))
                                            / 4), 1),
                              interval=200, repeat=True)
plt.show()
