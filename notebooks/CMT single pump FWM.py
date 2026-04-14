import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

# -------- parameters --------
s = 5.0

# -------- nonlinear system --------
def equations(v, Delta):

    x1,y1,x2,y2,x3,y3 = v

    f1 = -x1 + x1**2*y1 + 2*x3**2*y1 + y1**3 \
        + 2*x2*x3*y2 + 2*y1*y2**2 \
        + x2**2*(2*y1 - y3) \
        + y2**2*y3 + 2*y1*y3**2 \
        + Delta*y1

    f2 = -x1**3 - x2**2*x3 - y1 + x3*y2**2 \
        - 2*x2*y2*y3 \
        - x1*(2*x2**2 + 2*x3**2 + y1**2 + 2*y2**2 + 2*y3**2 + Delta)

    f3 = s + 2*x1**2*y2 + x2**2*y2 \
        - 2*x1*x3*y2 + 2*x3**2*y2 \
        + 2*y1**2*y2 + y2**3 \
        + 2*y1*y2*y3 + 2*y2*y3**2 \
        + x2*(-1 + 2*x3*y1 + 2*x1*y3) \
        + Delta*y2

    f4 = -2*x1**2*x2 - x2**3 \
        - (1 + 2*x3*y1)*y2 \
        - 2*x1*(x2*x3 + y2*y3) \
        - x2*(2*x3**2 + 2*y1**2 + y2**2 - 2*y1*y3 + 2*y3**2 + Delta)

    f5 = -x3 + 2*x1*x2*y2 + y1*y2**2 \
        - x2**2*(y1 - 2*y3) \
        + 2*x1**2*y3 + x3**2*y3 \
        + 2*y1**2*y3 + 2*y2**2*y3 \
        + y3**3 + Delta*y3

    f6 = -2*x1**2*x3 - 2*x2**2*x3 - x3**3 \
        - 2*x3*y1**2 - 2*x2*y1*y2 \
        - 2*x3*y2**2 \
        + x1*(-x2**2 + y2**2) \
        - y3 - x3*y3**2 - Delta*x3

    return [f1,f2,f3,f4,f5,f6]


# -------- detuning sweep --------
detuning = np.linspace(-7,5,200)

I1=[]
I2=[]
I3=[]

# initial guess
guess = np.array([0,0,0.5,0,0,0])

for d in detuning:

    sol = root(equations,guess,args=(d,),method='hybr')

    if sol.success:

        guess = sol.x

        x1,y1,x2,y2,x3,y3 = sol.x

        I1.append(x1**2+y1**2)
        I2.append(x2**2+y2**2)
        I3.append(x3**2+y3**2)

    else:

        I1.append(np.nan)
        I2.append(np.nan)
        I3.append(np.nan)


# -------- plot --------
plt.figure(figsize=(7,5))

plt.plot(detuning,I1,label="|a1|^2")
plt.plot(detuning,I2,label="|a2|^2")
plt.plot(detuning,I3,label="|a3|^2")

plt.xlabel("Detuning Δ")
plt.ylabel("Intensity")
plt.legend()
plt.grid()

plt.show()