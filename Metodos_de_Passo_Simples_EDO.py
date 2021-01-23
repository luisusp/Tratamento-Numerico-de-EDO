"""
IME-USP
MAP5725 - TNEDO - 2021

@author: luis Eduardo

e-mail: luislopes@ime.usp.br
"""

import math
import matplotlib.pyplot as plt
import numpy as np

def plot_grafico(t, y1, y2):

    plt.plot(t, y1, 'k:', color='black', linestyle=(0,(1,1,3,1)),label = 'x(t)  (in x units)')
    plt.xlabel('time t   (in units)')
    plt.ylabel('x(t)  state variables')
    plt.title('Numerical Approximation of State Variables for n = {} '.format(n))
    plt.legend()
    plt.show()

    plt.plot(t, y2, 'k--', label = 'y(t)  (in y units)')
    plt.xlabel('time t   (in units)')
    plt.ylabel('y(t)  state variables')
    plt.title('Numerical Approximation of State Variables for n = {} '.format(n))
    plt.legend()
    plt.show()

def matriz_inversa(t, h):
    '''
        Matriz inversa (I-hA)^(-1)
    '''
    A = np.array([
        [0,            1  ], 
        [-4 * t**2,    1/t]
    ])

    I = np.eye(2)

    return np.linalg.inv(I - h * A)

def solucao_exata(t):
    '''
        Solução exata da EDO
    '''

    s0 = np.sin(t**2)
    s1 = 2 * t * (np.cos(t**2))

    return np.array([s0,s1])

def f(t, y):
    '''
        Função do sistema de EDO: f(x, y) = (y, y/t - 4xt^2)
    '''
    f0 = y[1]
    f1 = y[1]/t - 4 * (y[0]) * (t**2)

    return np.array([f0,f1])

def explicit_euler_method(y0, n, t0, T):
    '''
        Método de Euler Explícito
    '''
    t_n = [t0]
    y_n = [y0]

    h = (T - t0)/n

    for i in range(n):

        yn = y_n[-1] + h * f(t_n[-1], y_n[-1])
        tn = t_n[-1] + h

        y_n.append(yn)
        t_n.append(tn)

        h = min(h, T-t_n[-1])

    return np.array(y_n), t_n

def implicit_euler_method(y0, n, t0, T):
    '''
        Método de Euler Implícito 
    '''
    h = (T - t0)/n
    t_n = [t0]
    y_n = [y0]

    for i in range(n):
        
        M   = matriz_inversa(t_n[-1], h)
        tn  = t_n[-1] + h
        yn  = y_n[-1] + h*f(tn, M.dot(y_n[-1]) )
        
        y_n.append(yn)
        t_n.append(tn)

    return (y_n, t_n)

def g(t,y,h,f):
    '''
        Função auxiliar para o Método de Euler Aprimorado
    '''
    k1 = f(t, y)
    k2 = f(t+h, y + h*k1)

    
    return (1/2) * (k1 + k2) 

def improved_euler_method(y0, n, t0, T):
    '''
        Método de Euler Aprimorado
    '''
    t_n = [t0]
    y_n = [y0]

    h = (T - t0)/n

    for i in range(n):

        yn = y_n[-1] + h * g(t_n[-1], y_n[-1], h, f)
        tn = t_n[-1] + h

        y_n.append(yn)
        t_n.append(tn)

        h = min(h, T-t_n[-1])

    return np.array(y_n), t_n

# Intervalo de tempo t em [t_0, T]
tn = np.sqrt(np.pi)                 # Tempo inicial (dado)
T  = 2 * (np.pi)                    # Tempo final (escolhido)
# T  = 4 * (np.pi)                    # Tempo final (escolhido - caso extra)
# T  = (np.pi)                        # Tempo final (escolhido - caso extra)

# Condição inicial: f(x_0, y_0) = (0, -2sqrt(pi))
y0 = np.array([0, -2 * np.sqrt(np.pi)])

#Subdivisões do intervalo [t_0, T]
n = int(input("n = "))

#Solução Exata
print('Solução exata: ', solucao_exata(T))

#Solução do Método de Euler Explícito
sol, t = explicit_euler_method(y0, n, tn, T)
print('Euler Explicito - Solução aproximada: ', sol[-1])

#Gráfico do Método de Euler Explícito 
plot_grafico(t, sol[:, 0], sol[:, 1])

#Erro Global - Euler Explícito
e_k = solucao_exata(T) - sol[-1]
print('Erro de Euler Explicito = ', e_k)

#Solução do Método de Euler Implícito
sol = implicit_euler_method(y0, n, tn, T)[0]
t = implicit_euler_method(y0, n, tn, T)[1]
sol = np.array(sol) 
print('Euler Implicito - Solução aproximada: ', sol[n]) 

#Gráfico do Método de Euler Implícito 
plot_grafico(t, np.array(sol[:, 0]), np.array(sol[:, 1]))

#Erro Global - Euler Implícito
e_k = solucao_exata(T) - sol[n]
print('Erro de Euler Implicito = ', e_k)

#Solução do Método de Euler Aprimorado
sol, t = improved_euler_method(y0, n, tn, T)
print('Euler Aprimorado - Solução aproximada: ', sol[-1])

#Gráfico do Método de Euler Aprimorado 
plot_grafico(t, sol[:, 0], sol[:, 1])

#Erro Global - Euler #Gráfico do Método de Euler Aprimorado 
e_k = solucao_exata(T) - sol[-1]
print('Erro de Euler Aprimorado = ', e_k)