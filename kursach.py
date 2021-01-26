import numpy as np
import matplotlib.pyplot as plt
 
def f1(x, y, a, b):
    return a*x - b*y*x

def f2(x, y, c, d):
    return -c*y + d*x*y


class Solution(object):
    
    def __init__(self, a, b, c, d, x0, y0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        self.x0 = x0
        self.y0 = y0
        
        self.T0 = 0
        self.T = 75
        
    @staticmethod
    def find_e(y1, y2, x1, x2, p):
        r1 = np.array(abs(y1 - y2)/(2**p -1))
        e1 = max(r1)
        r2 = np.array(abs(x1 - x2)/(2**p -1))
        e2 = max(r2)
        return max(e1, e2)

    def adam_solver(self, h):
        t = np.arange(self.T0, self.T, h)
        y = np.zeros(len(t))
        x = np.zeros(len(t))
        y[0] = self.y0
        x[0] = self.x0
        for i in range(len(y) - 1):
            x_next, y_next = self.predicator(h, x[i], y[i])
            x[i + 1], y[i + 1] = self.corrector(h, x[i], y[i], x_next, y_next)
        return t, x, y, 2

    def euler_solver(self, h):
        t = np.arange(self.T0, self.T, h)
        y = np.zeros(len(t))
        x = np.zeros(len(t))
        y[0] = self.y0
        x[0] = self.x0
        for i in range(len(y) - 1):
            x[i + 1] = x[i] + h*f1(x_pred, y_pred, self.a, self.b)
            y[i + 1] = y[i] + h*f2(x_pred, y_pred, self.c, self.d)
        return t, x, y, 1


    def acc(self, solver, eps):
        k = 0
        h = 0.2
        t1, x_1, y_1, p = solver(h/2)
        y1 = y_1[::2]
        x1 = x_1[::2]
        t2, x2, y2 = solver(h)[0], solver(h)[1], solver(h)[2]
        
        e = self.find_e(y1, y2, x1, x2, p)
        h = h/2
        
        while(e >= eps):
            k += 1
            h /= 2
            y2 = y_1
            x2 = x_1
            t1, x_1, y_1 = solver(h)[0], solver(h)[1], solver(h)[2]
            y1 = y_1[::2]
            x1 = x_1[::2]
            e = self.find_e(y1, y2, x1, x2, p)
        print("k", k, "h = ", h) 
        return t1, x_1, y_1
                
    def predicator(self, h, x_pred, y_pred):
        x = x_pred + h*f1(x_pred, y_pred, self.a, self.b)
        y = y_pred + h*f2(x_pred, y_pred, self.c, self.d)
        return x, y
    
    def corrector(self, h, x_pred, y_pred, x_next, y_next):
        x = x_pred + h/2*(f1(x_pred, y_pred, self.a, self.b) + f1(x_next, y_next, self.a, self.b))
        y = y_pred + h/2*(f2(x_pred, y_pred, self.c, self.d) + f2(x_next, y_next, self.c, self.d))
        return x, y

    

def test_solution(t):
    return 4*np.exp(-t)-np.exp(2*t), np.exp(-t)-np.exp(2*t)

def get_plot(x, y, label, color = 'darkblue'):
    plt.plot(x, y, ls="-", label = label, color = color)
    plt.legend()
    plt.savefig("test1.png", dpi = 500)
    plt.show()

def get_plot_2(x, y, z, label1, label2, color1 = 'darkblue', color2 = 'deeppink'):
    plt.plot(x, y, ls="-", label = label1, color = color1)
    plt.plot(x, z, ls="-", label = label2, color = color2)
    plt.legend()
    plt.savefig("test1.png", dpi = 500)
    plt.show()  
    
if __name__ == '__main__':
    a = 0.4
    b = 1.4
    c = 0.4
    d = 1.4
    x0 = 1
    y0 = 1

    eps = 0.01
    solution = Solution(a, b, c, d, x0, y0)
    
    
    t, x, y = solution.acc(solution.adam_solver, eps)
    get_plot_2(t, x, y, "x(t)", "y(t)")
    get_plot(x, y, "y(x)")




































