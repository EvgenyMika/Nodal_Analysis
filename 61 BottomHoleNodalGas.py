"""
Petroleum Production Engineering Computer-Assisted Approach
- Boyun Guo, Ph.D.
- William C. Lyons, Ph.D.
- Ali Ghalambor, Ph.D.

Chapter 6: Well Deliverability
Example Problem 6.1
Description: This program calculates gas well deliverability with bottom-hole node.
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from Module_File import cur_date

# Input Data:
gsg = 0.65  # Gas specific gravity (gg): 0.65
TDi = 2.259  # Tubing inside diameter (D): 2.259 in.
eD = 0.0006  # Tubing relative roughness (e/D): 0.0006
L = 10000  # Measured depth at tubing shoe (L): 10000 ft
theta = 0  # Inclination angle (theta): 0 Deg
phf = 800  # Wellhead pressure (phf): 800 psia
Thf = 150  # Wellhead temperature (Thf ): 150 F
Twf = 200  # Bottom hole temperature (Twf): 200 F
pr = 2000  # Reservoir pressure (p): 2000 psia
C = 0.01  # C-constant in backpressure IPR model: 0.01 Mscf/d-psi2n
n = 0.8  # n-exponent in backpressure IPR model: 0.8

# Solution
figsize = (10, 6)
tpc = 168 + 325 * gsg - 12.5 * gsg ** 2  # oR
ppc = 677 + 15 * gsg - 37.5 * gsg ** 2  # psia
print(f'Gas pseudo criticals: tpc = {tpc:.4f} oR, ppc = {ppc:.4f} psia')

Tav = (Thf + Twf) / 2 + 460
pav = (phf + pr) / 2
thetarad = theta * np.pi / 180
tavpr = Tav / tpc
pavrpr = pav / ppc


def Zf(tpr, ppr):  # Z
    Az = 1.39 * (tpr - 0.92) ** 0.5 - 0.36 * tpr - 0.101
    Bz = (0.62 - 0.23 * tpr) * ppr
    Bz = Bz + (0.066 / (tpr - 0.86) - 0.037) * ppr ** 2
    Bz = Bz + 0.32 / 10 ** (9 * (tpr - 1)) * ppr ** 6
    Cz = 0.132 - 0.32 * np.log10(tpr)
    Dz = 10 ** (0.3106 - 0.49 * tpr + 0.1824 * tpr ** 2)

    Z = Az + (1 - Az) / np.exp(Bz) + Cz * ppr ** Dz
    return Z


zav = Zf(tavpr, pavrpr)
s = 0.0375 * gsg * L * np.cos(thetarad) / zav / Tav
es = np.exp(s)
fm = (1 / (1.74 - 2 * np.log10(2 * eD))) ** 2
AOF = C * (pr ** 2) ** n
print(f'AOF = {AOF:.0f} Mscf/d')

def TPRgas(q0):
    global es, phf, fm, zav, Tav, TDi
    p0 = es * phf ** 2
    p0 = (p0 + (0.000667 * (es - 1) * fm * (q0 * zav * Tav) ** 2) / (TDi ** 5 * np.cos(thetarad))) ** 0.5
    return p0


def IPRgas(q0):
    global pr, C, n
    if (pr ** 2 - (q0 / C) ** (1 / n)) >= 0:
        IPR0 = (pr ** 2 - (q0 / C) ** (1 / n)) ** 0.5
    else:
        IPR0 = 0
    return IPR0


def NodalGas(q0):
    OF61 = IPRgas(q0) - TPRgas(q0)
    return OF61


pointsnum = 100
rates = np.linspace(0, AOF, pointsnum)
IPR = []
TPR = []
for q1 in rates:
    IPR.append(IPRgas(q1))
    TPR.append(TPRgas(q1))

df61 = pd.DataFrame({
    'rate': rates,
    'IPR': IPR,
    'TPR': TPR,
})

qop61 = fsolve(NodalGas, AOF/2)[0]
pop61 = IPRgas(qop61)

print(f'Operating flowrate = {qop61:.0f}  Mscf/d')
print(f'Operating pressure = {pop61:.0f} psia')

print('Writing results to Excel file.')
curdate = cur_date()
path_res = f'Nodal_out'
file_out_xls = f'{path_res}/61 BottomHoleNodalGas-{curdate}.xlsx'
file_out_fig = f'{path_res}/61 BottomHoleNodalGas-{curdate}.png'

with pd.ExcelWriter(file_out_xls) as writer:
    df61.to_excel(writer, sheet_name='Nodal Gas')

fig, ax = plt.subplots(figsize=figsize)  # Create a figure containing a single axes.
ax.plot(df61['rate'], df61['IPR'], color='k', linewidth=2, linestyle='-', label='IPR')
ax.plot(df61['rate'], df61['TPR'], color='b', linewidth=1, linestyle='--', label='TPR')
ax.plot([qop61, qop61], [0, pop61],
        color='r', linewidth=1, linestyle=':', label=f'Operating flow rate = {qop61:.0f}  Mscf/d')
ax.plot([0, qop61], [pop61, pop61],
        color='r', linewidth=1, linestyle=':', label=f'Operating bottomhole pressure = {pop61:.0f} psia')
ax.set_title('Gas Bottom Hole Nodal Analysis')
ax.set_xlim([0.0, 2000])
ax.set_ylim([0, 2500])
ax.grid(True)
ax.set_xlabel('Gas Production Rate (Mscf/d)')
ax.set_ylabel('Bottomhole Pressure (psia)')
ax.grid(True)
plt.legend()
plt.savefig(file_out_fig)
plt.show()
