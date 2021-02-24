"""
IME-USP
MAP5725 - TNEDO - 2021
@author: Luis Eduardo
e-mail: luislopes@ime.usp.br
www.ime.usp.br/~luislopes
"""

import numpy as np
import matplotlib.pyplot as plt
    
""" ================================================================
    Case 1: IVP of ODE (x, y)' = (y, y/t - 4xt^2)
                with initial value x(pi) = 0 e y(pi) = -2 sqrt(pi)
                Exact solution is (x, y) = (sin(t^2), 2t cos(t^2))
                a = sqrt(pi), b = 2pi, N = 1024 , ya = [0, -2 sqrt(pi)]
"""

def f(t, y):
    '''
        IVP of ODE: f(x, y) = (y, y/t - 4xt^2)
    '''
    f0 = y[1]
    f1 = y[1]/t - 4 * (y[0]) * (t**2)

    return np.array([f0,f1])

def solucao_exata(t):
    '''
        Exact solution
    '''
    
    s0 = np.sin(t**2)
    s1 = 2 * t * (np.cos(t**2))

    return np.array([s0,s1])

def AB_Explicit_Method(def_fn, a, b, ya, N):
    '''
        Adams-Bashford three-Step Explict Method
    '''
        
    # intialization of inputs
    f = def_fn                           # intakes function to method to approximate 
    
    h = (b-a)/N                          # creates step size based on input values of a, b, N 
    
    t = np.arange(a, b+h, h)             # time t
    
    y = [0 for _ in range (0, N+1)]      # state variables
    
    y[0] = ya                            # intial condition    
    
    #using RK3 to obtain the first 3 points
    for i in range(0,N):        # establishes iteration for N mesh points
        if i in range(0,2):     # for the first 2 iterations in N, runge kutta three order holds
            
            k1 = h * f(t[i],y[i])
            k2 = h * f(t[i] + (h/2), y[i] +(k1*h)/2)
            k3 = h * f(t[i] + h, y[i] - h*k1 + (2)*(k2)*h)
        
            y[i + 1] = y[i] + (k1 + 4*k2 + k3)/6
        else:                   # else, N>2, explicit AB3 method
        
            y[i + 1] = y[i] + h*(23 * f(t[i],y[i]) - 16 * f(t[i-1],y[i-1]) + 5 * f(t[i-2],y[i-2]))/12
    
    return(np.array(t), np.array(y))

def Method_2(def_fn, a, b, ya, N):
    '''
        Three-step method 2
    '''    
    # intialization of inputs
    f = def_fn                           # intakes function to method to approximate 
    
    h = (b-a)/N                          # creates step size based on input values of a, b, N 
    
    t = np.arange(a, b+h, h)             # time t
    
    y = [0 for _ in range (0, N+1)]      # state variables
    
    y[0] = ya                            # intial condition     
    
    # using RK3 to obtain the first 3 points
    for i in range(0,N):        # establishes iteration for N mesh points
        if i in range(0,2):     # for the first 2 iterations in N, runge kutta three order holds
            
            k1 = h * f(t[i],y[i])
            k2 = h * f(t[i] + (h/2), y[i] +(k1*h)/2)
            k3 = h * f(t[i] + h, y[i] - h*k1 + 2*(k2)*h)
        
            y[i + 1] = y[i] + (k1 + 4*k2 + k3)/6
        else:                   # else, N>2, explicit AB3 method
        
            y[i + 1] = (11*y[i])/10 - y[i-1]/10 + h*(75*f(t[i],y[i]) - 56 * f(t[i-1],y[i-1]) +  17*f(t[i-2],y[i-2]))/40
    
    return(np.array(t), np.array(y))

def Method_3(def_fn, a, b, ya, N):
    '''
        Three-step method 3
    '''    
    # intialization of inputs
    f = def_fn                           # intakes function to method to approximate 
    
    h = (b-a)/N                          # creates step size based on input values of a, b, N 
    
    t = np.arange(a, b+h, h)             # time t
    
    y = [0 for _ in range (0, N+1)]      # state variables
    
    y[0] = ya                            # intial condition     
    
    # using RK3 to obtain the first 3 points
    for i in range(0,N):        # establishes iteration for N mesh points
        if i in range(0,2):     # for the first 2 iterations in N, runge kutta three order holds
            
            k1 = h * f(t[i],y[i])
            k2 = h * f(t[i] + (h/2), y[i] +(k1*h)/2)
            k3 = h * f(t[i] + h, y[i] - h*k1 + (2)*(k2)*h)
        
            y[i + 1] = y[i] + (k1 + 4*k2 + k3)/6
        else:                   # else, N>2, explicit AB3 method
        
            y[i + 1] = 2*y[i] - 3*(y[i-1])/2 + y[i-2]/2 + h*(35*f(t[i],y[i]) - 40 * f(t[i-1],y[i-1]) + 17 * f(t[i-2],y[i-2]))/24
    
    return(np.array(t), np.array(y))

def plot_grafico(y1, y2, N):

    plt.plot(y1, y2, 'k:', color='black', linestyle=(0,(1,1,3,1)),  label = '(x(t), y(t))')
    plt.xlabel('x(t)  state variables')
    plt.ylabel('y(t)  state variables')
    plt.title('Numerical Approximation of State Variables for n = {} '.format(N))
    plt.legend()
    plt.show()

# RK3 for case #1

N_ab1 = 1024  # choose the value of n - number of integration steps
a_ab1 = np.sqrt(np.pi) # left end point of interval [a,b]
b_ab1 = 2 * (np.pi) # right end point of interval [a,b]
ya_ab1 = np.array([0, -2 * np.sqrt(np.pi)]) # initial value y(a)


# defining function and true solution of function #1 
def_fn_ab1 = f #case1_def_fn
sol_ab1 = solucao_exata

##############################################################################
##############################################################################
"""
    Adams-Bashforth Explicit Approximation Tests/Plotting
"""
##############################################################################
##############################################################################

# eulers approx for case 1 function
(t_ab1,y_ab1) = AB_Explicit_Method(def_fn_ab1, a_ab1, b_ab1, ya_ab1, N_ab1)

# compute exact for example #1 solution for comparison
z_ab1 = sol_ab1(t_ab1[-1]) 

# global discretization error
e_k = abs(z_ab1 - y_ab1[-1])

print("Exact Solution: ", max(abs(z_ab1)))
print("--------------------------------------------------------------------\n");
print("  n\t\t \tAB3\t\t \tGlobal Disc. Error\n");
print("--------------------------------------------------------------------\n");
print(N_ab1, max(abs(y_ab1[-1])), max(e_k))

# AB3 graph
plot_grafico(y_ab1[:, 0], y_ab1[:, 1], N_ab1)
              
##############################################################################
##############################################################################
"""
    Method 2 Approximation Tests/Plotting
"""
##############################################################################
##############################################################################

# eulers approx for example 1 function
(t_ab1,y_ab1) = Method_2(def_fn_ab1, a_ab1, b_ab1, ya_ab1, N_ab1)

# compute exact for example #1 solution for comparison
z_ab1 = sol_ab1(t_ab1[-1]) 

# global discretization error
e_k = abs(z_ab1 - y_ab1[-1])

print("--------------------------------------------------------------------\n");
print("  n\t\t Method 2 \t\tGlobal Disc. Error\n");
print("--------------------------------------------------------------------\n");
print(N_ab1, max(abs(y_ab1[-1])), max(e_k))

# Method 2 graph
plot_grafico(y_ab1[:, 0], y_ab1[:, 1], N_ab1)
            
##############################################################################
##############################################################################
"""
    Method 3 Approximation Tests/Plotting
"""
##############################################################################
##############################################################################

# eulers approx for example 1 function
(t_ab1,y_ab1) = Method_3(def_fn_ab1, a_ab1, b_ab1, ya_ab1, N_ab1)

# compute exact for example #1 solution for comparison
z_ab1 = sol_ab1(t_ab1[-1]) 

# global discretization error
e_k = abs(z_ab1 - y_ab1[-1])

print("--------------------------------------------------------------------\n");
print("  n\t\t Method 3 \t\tGlobal Disc. Error\n");
print("--------------------------------------------------------------------\n");
print(N_ab1, max(abs(y_ab1[-1])), max(e_k))

# Method 3 graph
plot_grafico(y_ab1[:, 0], y_ab1[:, 1], N_ab1)
               
##############################################################################
##############################################################################