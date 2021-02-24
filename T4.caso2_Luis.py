"""
IME-USP
MAP5725 - TNEDO - 2021
@author: Luis Eduardo
e-mail: luislopes@ime.usp.br
www.ime.usp.br/~luislopes
"""

from math import*
import matplotlib.pyplot as plt
import numpy as np 

""" ================================================================
    Case 2: IVP of ODE y' = {-2t + [5 cos(5t)]/[2 + sin(5t)]} y
                with initial value y(-pi) = 2 exp(-pi^2)
                Exact solution is y(t) = [2 + sin(5t)] exp(-t^2)
                a = -pi, b = 3pi, N = 1024 , y0 = 2 exp(-pi^2)
"""

def f(t,y):   
    '''
        IVP of ODE
    '''     

    return -2*t*y + ( (5*y*cos(5*t)) / (2+sin(5*t)) )

def exact_solution(t):
    '''
        Exact solution
    '''
    
    return exp(-t**2)*(2+sin(5*t))

def AB_Explicit_Meth(f,a,y0,b,n):
    '''
        Adams-Bashford three-Step Explict Method
    '''    
        
    h = (b-a)/n

    yk = [np.array([a,y0])]
    yk.append([a + h, exp(-(a + h)**2)*(2+sin(5*(a + h)))])
    yk.append([a + 2*h, exp(-(a + 2*h)**2)*(2+sin(5*(a + 2*h)))])
    
    while yk[-1][0] < b:
        yk.append( [ yk[-1][0] + h, yk[-1][1] + h*( (5)*f(yk[-3][0],yk[-3][1]) - (16)*f(yk[-2][0],yk[-2][1]) + (23)*f(yk[-1][0],yk[-1][1]) )/12 ] )        
    
    return np.array(yk)

def Method_2(f,a,y0,b,n):
    '''
        Three-step method 2
    '''
     
    h = (b-a)/n

    yk = [np.array([a,y0])]
    yk.append([a + h, exp(-(a + h)**2)*(2+sin(5*(a + h)))])
    yk.append([a + 2*h, exp(-(a + 2*h)**2)*(2+sin(5*(a + 2*h)))])
    
    while yk[-1][0] < b:
        yk.append( [ yk[-1][0] + h, - (1/10)*yk[-2][1] + (11/10)*yk[-1][1] + h*( (17)*f(yk[-3][0],yk[-3][1]) - (56)*f(yk[-2][0],yk[-2][1]) + (75)*f(yk[-1][0],yk[-1][1]) )/40 ] )        
    
    return np.array(yk)

def Method_3(f,a,y0,b,n):
    '''
        Three-step method 3
    '''
     
    h = (b-a)/n

    yk = [np.array([a,y0])]
    yk.append([a + h, exp(-(a + h)**2)*(2+sin(5*(a + h)))])
    yk.append([a + 2*h, exp(-(a + 2*h)**2)*(2+sin(5*(a + 2*h)))])
    
    while yk[-1][0] < b:
        yk.append( [ yk[-1][0] + h, (1/2)*yk[-3][1] - (3/2)*yk[-2][1] + 2*yk[-1][1] + h*( (17)*f(yk[-3][0],yk[-3][1]) - (40)*f(yk[-2][0],yk[-2][1]) + (35)*f(yk[-1][0],yk[-1][1]) )/24 ] )        
    
    return np.array(yk)

def Error_AB(exact_solution,AB_Explicit_Meth,a,y0,b,n): 
    '''
        Global discretization error - AB3
    ''' 

    F = []
    
    while (n <= 100000):
        F.append( np.array([n, (b - a)/n, abs(exact_solution(b) - AB_Explicit_Meth(f,a,y0,b,n)[:,1][n])]) )
        n = 2*n
    
    return np.array(F)


def Order_AB(Error_AB,a,y0,b,n):
    '''
        Estimation of the order of convergence - AB3
    '''        
    t = len(Error_AB(exact_solution,AB_Explicit_Meth,a,y0,b,n)[:,0])
    error = Error_AB(exact_solution,AB_Explicit_Meth,a,y0,b,n)[:,2]
    order = []
    
    for i in range(1,t):
        R = (error[i-1]) / (error[i])
        order.append(np.array([R,log(R,2)]))
                 
    return np.array(order)

def Error_Method_2(exact_solution,Method_2,a,y0,b,n):
    '''
        Global discretization error - Method 2
    ''' 
    
    F = []
    
    while (n <= 100000):
        F.append( np.array([n, (a - b)/n, abs(exact_solution(b) - Method_2(f,a,y0,b,n)[:,1][n])]) )
        n = 2*n
    
    return np.array(F)

def Order_Method_2(Error_Method_2,a,y0,b,n):
    '''
        Estimation of the order of convergence - Method 2
    '''     
    t = len(Error_Method_2(exact_solution,Method_2,a,y0,b,n)[:,0])
    error = Error_Method_2(exact_solution,Method_2,a,y0,b,n)[:,2]
    order = []
    
    for i in range(1,t):
        R = (error[i-1]) / (error[i])
        order.append(np.array([R,log(R,2)]))
                  
    return np.array(order)

def Error_Method_3(exact_solution,Method_3,a,y0,b,n):
    '''
        Global discretization error - Method 3
    ''' 
       
    F = []
    
    while (n <= 100000):
        F.append( np.array([n, (b - a)/n, abs(exact_solution(b) - Method_3(f,a,y0,b,n)[:,1][n])]) )
        n = 2*n
    
    return np.array(F)

def Order_Method_3(EGD_M33,a,y0,b,n):
    '''
        Estimation of the order of convergence - Method 3
    '''     
    t = len(Error_Method_3(exact_solution,Method_3,a,y0,b,n)[:,0])
    error = Error_Method_3(exact_solution,Method_3,a,y0,b,n)[:,2]
    order = []
    
    for i in range(1,t):
        R = (error[i-1]) / (error[i])
        order.append(np.array([R,log(R,2)]))
                 
    return np.array(order)

""" ================================================================
                             RESULTS
    ================================================================"""

print('1. Adams-Bashforth methods with 3 steps.')
Error1 = Error_AB(exact_solution,AB_Explicit_Meth,-pi,2*exp(-pi**2),3*pi,16)
print('Verification of the order of convergence of the AB3 Method at time t = 3pi.')
print('/       n       /      h       /  |e(t,h)|  /')
print(Error1)
Order1 = Order_AB(Error_AB,-pi,2*exp(-pi**2),3*pi,16)
print('Finally, we conclude that: ')
print('/ q = |e(t,2h)|/|e(t,h)| /   Order (log_2 (q))   /')
print(Order1)

print("--------------------------------------------------------------------\n");
print('2. Multistep Methods #2 with 3 steps.')
Error2 = Error_Method_2(exact_solution,Method_2,-pi,2*exp(-pi**2),3*pi,16)
print('Verification of the order of convergence of Method II at time t = 3pi.')
print('/       n       /      h       /  |e(t,h)|  /')
print(Error2)
Order2 = Order_Method_2(Error_Method_2,-pi,2*exp(-pi**2),3*pi,16)
print('Finally, we conclude that: ')
print('/ q = |e(t,2h)|/|e(t,h)| /   Order  (log_2 (q)) /')
print(Order2)

print("--------------------------------------------------------------------\n");
print('3. Multistep Methods #3 with 3 steps.')
Error3 = Error_Method_3(exact_solution,Method_3,-pi,2*exp(-pi**2),3*pi,16)
print('Verification of the order of convergence of Method II at time t = 3pi.')
print('/       n       /      h       /  |e(t,h)|  /')
print(Error3)
Order3 = Order_Method_3(Error_Method_3,-pi,2*exp(-pi**2),3*pi,16)
print('Finally, we conclude that: ')
print('/ q = |e(t,2h)|/|e(t,h)| /   Order (log_2 (q))  /')
print(Order3)

""" ================================================================
                             GRAPHICS
    ================================================================"""
  
#Exact solution.
t = np.linspace(-pi,3*pi,1024)
sol = np.exp(-t**2)*(2+np.sin(5*t))

plt.plot(t,sol,'black',label='(t, y(t)) exact')
plt.title('Exact Solution of ODE')
plt.ylabel('y(t)  state variable')
plt.xlabel('time t   (in units)')
plt.legend()
plt.show()    

####################################################################################
  
n = 1024 # choose the value of n - number of integration steps

#AB3 Method
M1 = AB_Explicit_Meth(f,-pi,2*exp(-pi**2),3*pi,n)

#Graph of approximate solution - AB3
plt.plot(M1[:,0],M1[:,1],'k--',label='(t, y(t)) aprox.')
plt.title('Numerical Approximation to the 3-step AB Method for n = {} '.format(n))
plt.ylabel('y(t)  state variable')
plt.xlabel('time t   (in units)')
plt.legend()
plt.show()

####################################################################################

# Method 2
M2 = Method_2(f,-pi,2*exp(-pi**2),3*pi,n)

#Graph of approximate solution - Method 2
plt.plot(M2[:,0],M2[:,1],'k--',label='(t, y(t)) aprox.')
plt.title('Numerical Approximation of the 3-step Method II for n = {} '.format(n))
plt.ylabel('y(t)  state variable')
plt.xlabel('time t   (in units)')
plt.legend()
plt.show()

####################################################################################

# Method 3
M3 = Method_3(f,-pi,2*exp(-pi**2),3*pi,n)

#Graph of approximate solution - Method 3
plt.plot(M3[:,0],M3[:,1],'k--',label='(t, y(t)) aprox.')
plt.title('Numerical Approximation of the 3-step Method III for n = {} '.format(n))
plt.ylabel('y(t)  state variable')
plt.xlabel('time t   (in units)')
plt.legend()
plt.show()