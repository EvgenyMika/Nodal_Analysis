"""
Petroleum Production Engineering
Computer-Assisted Approach
- Boyun Guo, Ph.D.
- William C. Lyons, Ph.D.
- Ali Ghalambor, Ph.D.

Chapter 6: Well Deliverability
Example Problem 6.5
Description: This program calculates gas well deliverability with wellhead node.
"""
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from Module_File import cur_date

# Input Data:
gsg = 0.75  # Gas specific gravity (gg): 0.71
Di = 2.259  # Tubing inside diameter (D): 2.259 in.
eD = 0.0006  # Tubing relative roughness (e/D): 0.0006
L = 8000  # Measured depth at tubing shoe (L): 10000 ft
theta = 0  # Inclination angle (q): 0 Deg
Dck = 24  # Wellhead choke size (Dck): 16	 1/64 in.
Dfl = 2  # Flowline diameter (Dfl):	2 in.
k = 1.3  #Gas specific heat ratio (k):	1.3
mugwh = 0.01  # Gas viscosity at wellhead (m):	0.01	 cp
Thf = 120  # Wellhead temperature (Thf ):	120	 oF
Twf = 180  # Bottom hole temperature (Twf):	180	 oF
pr = 2000  # Reservoir pressure (pr):	2000	 psia
C = 0.01  # C-constant in backpressure IPR model:	0.01	 Mscf/d-psi2n
n = 0.8  # n-exponent in backpressure IPR model:	0.8
psc = 14.696  # Standard Pressure (SCP), psi
figsize = (10, 6)

# Solution
Ack = np.pi/4 * (Dck/64)**2
tpc = 168 + 325 * gsg - 12.5 * gsg ** 2  # 392.44875 oR
ppc = 677 + 15 * gsg - 37.5 * gsg ** 2  # 668.74625 psia
print(f'Gas pseudo criticals: tpc = {tpc:.4f} oR, ppc = {ppc:.4f} psia')
fm = (1 / (1.74 - 2 * np.log10(2 * eD))) ** 2
AOF = C * (pr**2) ** n
print(f'AOF = {AOF}  Mscf/d')
qAOF = round(AOF, 0)
DckDfl = Dck / Dfl / 64

def Zfunc(tpr, ppr):
    Az = 1.39 * (tpr - 0.92) ** 0.5 - 0.36 * tpr - 0.101
    Bz = (0.62 - 0.23 * tpr) * ppr + (0.066 / (tpr - 0.86) - 0.037) * ppr ** 2 + 0.32 / 10 ** (9 * (tpr - 1)) * ppr ** 6
    Cz = 0.132 - 0.32 * np.log10(tpr)
    Dz = 10 ** (0.3106 - 0.49 * tpr + 0.1824 * tpr ** 2)
    Z = Az + (1 - Az) / np.exp(Bz) + Cz * ppr ** Dz
    return Z


def ObjFun65(q0):
    global es, fm, zav, pwh, pr, theta, DI, qsc, phf, Tav, C, n
    Re = 20 * q0 * gsg / mugwh / (Dck/64)
    Cck = DckDfl + 0.3167 / DckDfl**0.6 + 0.025 * (np.log10(Re) - 4)
    pop = q0/879/Cck/Ack/(k/gsg/(Thf+460)*(2/(k+1))**((k+1)/(k-1)))**0.5
    q1 = C*(pr**2-(es*pop**2+(6.67e-4*(es-1)*fm*(q0*zav*Tav)**2)/(Di**5 * np.cos(thetarad))))**n
    OF65 = q0 - q1
    return OF65


def WPRf(phf0, q0):
    global C, n, es, fm, zav, pr, Tav, thetarad, Di
    p02 = (6.67e-4 * (es-1)*fm*(q0*zav*Tav)**2)/(Di**5 * np.cos(thetarad))
    q1 = C*(pr**2 - es*phf0**2 - p02)**n
    return q0 - q1


def CPRf(q1):
    global gsg, mugwh, Dck, DckDfl, k, Ack
    Re1 = 20 * q1 * gsg / mugwh / (Dck/64)
    Cck1 = DckDfl + 0.3167 / DckDfl**0.6 + 0.025 * (np.log10(Re1) - 4)
    CPR1 = q1 / 879 / Cck1 / Ack / (k/gsg/(Thf+460)*(2/(k+1))**((k+1)/(k-1)))**0.5
    return CPR1


pointsnum = 20
rates = np.linspace(0, AOF, pointsnum)

# Calculation of pwf with q=0.
Tav = (Thf + Twf) / 2 + 460
pav = (psc + pr) / 2
thetarad = theta * np.pi / 180
tavpr = Tav / tpc
pavrpr = pav / ppc
zav = Zfunc(tavpr, pavrpr)
s = 0.0375 * gsg * L * np.cos(thetarad) / zav / Tav
es = np.exp(s)
WPR0 = pr / es**0.5 * 0.9999
print(f'pwf(q=0) = {WPR0}')

WPR = []
CPR = []

for qi in rates:
    if qi == 0:
        WPRi = WPR0
    else:
        WPRi = fsolve(WPRf, WPR0, args=qi)[0]

    WPR.append(WPRi)
    WPR0 = WPRi
    if qi == 0:
        CPRi = 0
    else:
        CPRi = CPRf(qi)

    CPR.append(CPRi)


df65 = pd.DataFrame({
    'rates': rates,
    'WPR': WPR,
    'CPR': CPR,
})

qop0 = AOF * 0.5
qop = fsolve(ObjFun65, qop0)[0]
ObjFun65res = ObjFun65(qop)
print(f'Operating flowrate = {qop:.0f}  Mscf/d')

pop = CPRf(qop)
print(f'Operating pressure = {pop:.0f} psia')
print(f'Residual of objective function = {ObjFun65res}')

print('Writing results to Excel file.')
curdate = cur_date()
path_res = f'Nodal_out'
file_out_xls = f'{path_res}/65 WellHeadNodalGas-{curdate}.xlsx'
file_out_fig = f'{path_res}/65 WellHeadNodalGas-{curdate}.png'

with pd.ExcelWriter(file_out_xls) as writer:
    df65.to_excel(writer, sheet_name='Nodal Gas')

fig, ax = plt.subplots(figsize=figsize)  # Create a figure containing a single axes.
ax.plot(df65['rates'], df65['WPR'], color='k', linewidth=2, linestyle='-', label='WPR')
ax.plot(df65['rates'], df65['CPR'], color='b', linewidth=1, linestyle='--', label='CPR')
ax.plot([qop, qop], [0, pop], color='r', linewidth=1, linestyle=':', label=f'Operating flowrate = {qop:.0f} Mscf/d')
ax.plot([0, qop], [pop, pop], color='r', linewidth=1, linestyle=':', label=f'Operating pressure = {pop:.0f} psia')
ax.set_title('Gas Wellhead Nodal (Sonic Flow)')
ax.set_xlim([0, AOF])
ax.set_ylim([0, pr])
ax.grid(True)
ax.set_xlabel('Gas Production Rate (Mscf/d)')
ax.set_ylabel('Wellhead Pressure (psia)')
ax.grid(True)
plt.legend()
plt.savefig(file_out_fig)
plt.show()
