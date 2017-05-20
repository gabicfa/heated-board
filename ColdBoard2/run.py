from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LEFT_TEMP = 75.0
RIGHT_TEMP = 50.0
BOT_TEMP = 0.0
TOP_TEMP = 100.0

FLUX_TOP = None
FLUX_LEFT = None
FLUX_RIGHT = None
FLUX_BOT = 0

T_TOTAL = 200
T_STEP = 0.001

LEN_X = 5
LEN_Y = 5
LEN_STEP = 1

ALPHA = 1
ERROR = 0.000002

class TempCalculator2D:

    def __init__(self, len_x, len_y, len_step, t_total, t_step, alpha,
                 flux_top, flux_left, flux_right, flux_bot):
        self.len_step = len_step
        self.len_x, self.len_y = len_x, len_y
        self.matrix_x = int (len_x/len_step)
        self.matrix_y = int (len_y/len_step)
        self.t_step = t_step
        self.t_total = t_total
        self.f_0 = alpha * t_step / (len_step ** 2)
        if self.f_0 > 0.25:
            print("f0 muito grande")
        # self.create_matrix()
        # self.temp_board = [self.board]
        # self.error = [0]
        # self.k = k
        self.flux_top = flux_top
        self.flux_left = flux_left
        self.flux_right = flux_right
        self.flux_bot = flux_bot
        self.create_matrix_inicial()
        self.matrix_temps()

        # self.arr_temps = []

    def create_matrix_inicial(self):
        self.board = [[0 for i in range (self.matrix_x)]for j in range (self.matrix_y)]
        for i in range(1,self.matrix_x-1):
            self.board[0][i] = TOP_TEMP
            self.board[self.matrix_y - 1][i] = BOT_TEMP


        for i in range (1, self.matrix_y):
            self.board[i][0] = LEFT_TEMP
            self.board[i][self.matrix_x - 1] = RIGHT_TEMP



    def matrix_temps(self):
        self.arr_temps=[]
        curr_err = 10
        if self.t_total < 1:
            self.t_total *=10000
            self.t_step *= 10000


        n_temps = int (self.t_total/self.t_step)
        self.arr_temps.append(self.board)


        for t in range(1, n_temps) :
            matrix_t1 = [[0 for i in range (self.matrix_x)]for j in range (self.matrix_y)]

            for i in range (0, self.matrix_x):
                for j in range(0, self.matrix_y):
                    if i == 0  and j!=0 and j!= self.matrix_x-1:
                        if FLUX_TOP == None :
                            matrix_t1[i][j] = self.arr_temps[t-1][i][j]
                        else:
                            matrix_t1[i][j] = self.f_0*(2*self.arr_temps[t-1][i+1][j]-
                                            2*self.len_step*self.flux_top+self.arr_temps[t-1][i][j+1]+
                                            self.arr_temps[t-1][i][j-1])+(1-4*self.f_0)*self.arr_temps[t-1][i][j]

                    if j == 0 and i!=0 :
                        if FLUX_LEFT == None :
                            matrix_t1[i][j] = self.arr_temps[t-1][i][j]
                        else:
                            matrix_t1[i][j] = self.f_0*(2*self.arr_temps[t-1][i][j+1]-
                                            2*self.len_step*self.flux_left+self.arr_temps[t-1][i-1][j]+
                                            self.arr_temps[t-1][i+1][j])+(1-4*self.f_0)*self.arr_temps[t-1][i][j]

                    if i == self.matrix_y-1 and i!=j and j!=0:
                        if FLUX_BOT == None :
                            matrix_t1[i][j] = self.arr_temps[t-1][i][j]
                        else:
                            matrix_t1[i][j] = self.f_0*(2*self.arr_temps[t-1][i-1][j]-
                                            2*self.len_step*self.flux_bot+self.arr_temps[t-1][i][j-1]+
                                            self.arr_temps[t-1][i][j+1])+(1-4*self.f_0)*self.arr_temps[t-1][i][j]

                    if j == self.matrix_x-1 and i!=0 :
                        if FLUX_RIGHT == None :
                            matrix_t1[i][j] = self.arr_temps[t-1][i][j]
                        else:
                            matrix_t1[i][j] = self.f_0*(2*self.arr_temps[t-1][i][j-1]-
                                            2*self.len_step*self.flux_right+self.arr_temps[t-1][i+1][j]+
                                            self.arr_temps[t-1][i-1][j])+(1-4*self.f_0)*self.arr_temps[t-1][i][j]

                    if  i!=0 and i!=self.matrix_x-1:
                        if j!=0 and j!=self.matrix_y-1:
                            matrix_t1[i][j] = self.f_0*(self.arr_temps[t-1][i][j+1]+
                            self.arr_temps[t-1][i][j-1]+self.arr_temps[t-1][i-1][j]+
                            self.arr_temps[t-1][i+1][j])+(1-4*self.f_0)*self.arr_temps[t-1][i][j]
                            dif = abs(self.arr_temps[t-1][i][j] - matrix_t1[i][j])
                            print("dif")
                            print(dif)
                            if dif < curr_err and dif!=0:
                                curr_err = dif
                            print("curr_err")
                            print(curr_err)

            self.arr_temps.append(matrix_t1)
            if curr_err <= ERROR:
                print(t)
                break



MATRIX_TESTE =  TempCalculator2D(LEN_X, LEN_Y, LEN_STEP, T_TOTAL, T_STEP, ALPHA,
                                FLUX_TOP, FLUX_LEFT,FLUX_RIGHT,FLUX_BOT)

print('\n')
for i in MATRIX_TESTE.arr_temps[-1]:
    print(i)


def plot_color_gradients(gradient):
    final = np.zeros((len(gradient), len(gradient[0])))
    for x in range(len(gradient)):
        for y in range(len(gradient[0])):
            final[x, y] = gradient[x][y]
    ax.imshow(final, aspect='equal', cmap=plt.get_cmap('hot'))

# fig, ax = plt.subplots()
# fig.subplots_adjust(top=0.9, bottom=0, left=0, right=0.99)
# ax.set_title('Titulo', fontsize=14)
# ax.set_axis_off()

def animate(i):
    plot_color_gradients(MATRIX_TESTE.arr_temps[i])

def init():
    plot_color_gradients(MATRIX_TESTE.arr_temps[0])


# init()
# ani = animation.FuncAnimation(fig, animate, np.arange(1, len(MATRIX_TESTE.arr_temps)),
#                               interval=200, repeat=True)


# plt.show()
