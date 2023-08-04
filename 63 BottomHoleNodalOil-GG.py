"""
Petroleum Production Engineering
Computer-Assisted Approach
- Boyun Guo, Ph.D.
- William C. Lyons, Ph.D.
- Ali Ghalambor, Ph.D.

Chapter 6: Well Deliverability
Example Problem 6.3
Description: This program calculates flowing bottom-hole pressure based on tubing
head pressure and tubing flow performance using the Guo–Ghalambor method.
"""
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.optimize import newton
from scipy.optimize import minimize_scalar

import matplotlib.pyplot as plt
figsize = (10, 6)

from Module_File import cur_date

# Input data
pr = 3000  # Reservoir pressure:	3000	 psia
MD = 7000  # Total measured depth:	7000	 ft
AIA = 20  # Average inclination angle:	20	 deg
TiD = 1.995  # Tubing I.D.:	1.995	 in.
gsg = 0.7  # Gas specific gravity:	0.7	 air=1
gso = 0.85  # Oil specific gravity:	0.85	 H2O=1
WC = 0.3  # Water cut:	30	 %
gsw = 1.05  # Water specific gravity:	1.05	 H2O=1
qs = 1  # Solid production rate:	1	 ft3/d
gss = 2.65  # Solid specific gravity:	2.65	 H2O=1
Twh = 100  # Tubing head temperature:	100	 oF
Tbh = 160  # Bottom hole temperature:	160	 oF
pwh = 300  # Tubing head pressure:	300	 psia
AOF = 2000  # Absolute open flo (AOF):	2000	 bbl/d

# My addition to the given data
pb = pr  # 800
GLR = 800  # Gas-Liquid Ratio (GLR) scf/stb


# Solution:
A = np.pi * TiD ** 2 / 4  # in^2
D = TiD / 12  # ft
Tavr = (Twh + Tbh) / 2 + 460.0  # oR
cost = np.cos(AIA * np.pi / 180)  # = COS(deg/57.3)


def TPR_GG(px, q0):
    global Twh, MD, cost, gsg, gso, gsw, gss, GLR, AOF, qs
    ql = q0
    qo = ql * (1 - WC)
    qg = ql * GLR
    qw = ql * WC

    rliq = 350.17 * (gso + qw * gsw / qo) + 0.0765 * qg * gsg / qo  # lbm
    Drv = 1.4737 / 100000 * rliq * qo / (TiD / 12)
    fM = 4 * 4 * 10 ** (1.444 - 2.5 * np.log10(Drv))

    a = cost * (0.0765 * gsg * qg + 350 * gso * qo + 350 * gsw * qw + 62.4 * gss * qs) / (4.07 * qg * Tavr)
    b = (5.615 * qo + 5.615 * qw + qs) / (4.07 * qg * Tavr)
    c = 6.78 / 1000 * Tavr * qg / A
    d = (0.00166 / A) * (5.615 * qo + 5.615 * qw + qs)
    e = fM / (2 * 32.17 * D) / cost

    M = c * d * e / (cost + e * d ** 2)
    # M = c * d * e / (1 + e * d ** 2) # in the book cost=1
    N = (c ** 2 * e * cost) / (cost + e * d ** 2) ** 2
    # N = (c ** 2 * e) / (1 + e * d ** 2) ** 2 # in the book cost=1

    pxx = px * 144  # lbf/ft^2
    pwh1 = pwh *144  # lbf/ft^2

    part1 = b * (pxx - pwh1)
    part2 = (1 - 2 * b * M) / 2 * np.log(abs((((pxx + M) ** 2 + N) / ((pwh1 + M) ** 2 + N))))
    part3 = (M + b * N / c - b * M ** 2) / (N ** 0.5)
    part3 = part3 * (np.arctan((pxx + M) / (N ** 0.5)) - np.arctan((pwh1 + M) / (N ** 0.5)))

    RHS = part1 + part2 - part3
    LHS = a * (cost + e * d ** 2) * MD / cost
    # LHS = a * (1 + e * d ** 2) * MD / 1  # in the book cost=1

    err = RHS - LHS
    return err


def IPRf(q0):   # Bottom hole pressure, pwf
    global pr, AOF

    IPR0 = 0.125 * pr * ((81 - 80 * q0 / AOF) ** 0.5 - 1)
    return IPR0

def nodal(q1):
    pwf1 = IPRf(q1)
    pbh1 = fsolve(TPR_GG, pwf1, args=q1)[0]
    return pwf1 - pbh1

qop = fsolve(nodal, pr/2)[0]
pop = IPRf(qop)
print(f'Operating Liquid Rate = {qop:.0f} stb/d')
print(f'Operating Botom Hole Pressure = {pop:.0f} psia')

pointsnum = 101
qbegin = AOF / pointsnum
qend = AOF
rates63 = np.linspace(qbegin, qend, pointsnum)

IPR = []
TPR = []
for q0 in rates63:

    IPRq = IPRf(q0)
    IPR.append(IPRq)

    TPRq = fsolve(TPR_GG, 1000, args=q0)[0]
    TPR.append(TPRq)


df63 = pd.DataFrame({
    'rate': rates63,
    'IPR': IPR,
    'TPR': TPR,

})

print('Writing results to Excel file.')
curdate = cur_date()
path_res = f'Nodal_out'
file_out_xls = f'{path_res}/63 BottomHoleNodalOil-GG-{curdate}.xlsx'
file_out_fig = f'{path_res}/63 BottomHoleNodalOil-GG-{curdate}.png'

with pd.ExcelWriter(file_out_xls) as writer:
    df63.to_excel(writer, sheet_name='Nodal Oil-GG')

fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(df63['rate'], df63['IPR'], color='k', linewidth=2, linestyle='-', label='IPR Vogel')
ax.plot(df63['rate'], df63['TPR'], color='b', linewidth=1, linestyle='--', label='TPR Guo–Ghalambor')
ax.plot((qop, qop), (0, pop), color='r', linewidth=1, linestyle=':', label=f'Operating Liquid Rate = {qop:.0f} stb/d')
ax.plot((0, qop), (pop, pop), color='r', linewidth=1, linestyle=':', label=f'Operating Pressure = {pop:.0f} psia')
ax.set_title('Oil Bottom Hole Nodal by Guo–Ghalambor Method')
ax.set_xlim([0, AOF])
ax.set_ylim([0, pr])
ax.grid(True)
ax.set_xlabel('Liquid Production Rate (stb/d)')
ax.set_ylabel('Bottom Hole Pressure (psia)')
plt.legend()
plt.savefig(file_out_fig)
plt.show()

