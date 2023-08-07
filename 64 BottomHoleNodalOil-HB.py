"""
Petroleum Production Engineering
Computer-Assisted Approach
- Boyun Guo, Ph.D.
- William C. Lyons, Ph.D.
- Ali Ghalambor, Ph.D.

Chapter 6: Well Deliverability
Example Problem 6.4
Description: This spreadsheet calculates operating point using Hagedorn-Brown Correlation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from Module_File import cur_date
import matplotlib.pyplot as plt


# Input data
MD = 9850  # Depth (D):	9 850	 ft
TiD = 1.995  # Tubing inner diameter (dti):	1.995	 in.
API = 45  # Oil gravity (API):	45	oAPI
muo = 2  # Oil viscosity (cp):	2	 cp
GLR = 500  # Production GLR (GLR):	500	 scf/bbl
gsg = 0.7  # Gas specific gravity (gg):	0.7	 air =1
pwh = 450  # Flowing tubing head pressure (phf):	450	 psia
Twh = 80  # Flowing tubing head temperature (thf):	80	oF
Tbh = 180  # Flowing temperature at tubing shoe (twf):	180	oF
WCT = 0.1  # Water cut (WC):	10	 %
pr = 5000  # Reservoir pressure (pe):	5000	 psia
pb = 4000  # Bubble point pressure (pb):	4000	 psia
J = 1.5  # Productivity above bubble point (J*):	1.5	 stb/d-psi

# My addition to the given data
s = 30  # Interfacial tension (s): 30dynes/cm
gsw = 1.05  # Specific gravity of water (gw): 1.05 H2O=1
e = 0.0006
Yh0 = 0.0124657507677914
muw = 0.5
figsize = (10, 6)

# Solution:
print('Description: This program calculates operating point using Hagedorn-Brown Correlation.')
Area = np.pi / 4 * (TiD / 12) ** 2  # ft2
gso = 141.5 / (131.5 + API)
tpc = 169 + 314 * gsg  # oR
ppc = 708.75 - 57.7 * gsg  # psia
print(f'Gas pseudo criticals: tpc = {tpc:.2f} oR, ppc = {ppc:.2f} psia')

qmax = J * (pr - pb + pb / 1.8)
qgraph = (qmax // 1000 + 1) * 1000
qb = J * (pr - pb)
print(f'qb = {qb:.2f} snb/d, qmax = {qmax:.2f} stb/d')


def lnmuf(tpr0, ppr0):
    a0 = -2.462
    a1 = 2.97
    a2 = -0.2862
    a3 = 0.008054
    a4 = 2.808
    a5 = -3.498
    a6 = 0.3603
    a7 = -0.01044
    a8 = -0.7933
    a9 = 1.396
    a10 = -0.1491
    a11 = 0.00441
    a12 = 0.08393
    a13 = -0.1864
    a14 = 0.02033
    a15 = -0.0006095
    lnmu0 = a0 + a1 * ppr0 + a2 * ppr0 ** 2 + a3 * ppr0 ** 3
    lnmu1 = tpr0 * (a4 + a5 * ppr0 + a6 * ppr0 ** 2 + a7 * ppr0 ** 3)
    lnmu2 = tpr0 ** 2 * (a8 + a9 * ppr0 + a10 * ppr0 ** 2 + a11 * ppr0 ** 3)
    lnmu3 = tpr0 ** 3 * (a12 + a13 * ppr0 + a14 * ppr0 ** 2 + a15 * ppr0 ** 3)

    return lnmu0 + lnmu1 + lnmu2 + lnmu3


# for CNl calculation
a11 = -2.698510
a21 = 1.584095 / 10
a31 = -5.509976 / 10
a41 = 5.478492 / 10
a51 = -1.219458 / 10

# for ylf calculation
a12 = -1.030658 / 10
a22 = 6.177740 / 10
a32 = -6.329460 / 10
a42 = 2.959800 / 10
a52 = -4.010000 / 100

# for phi calc
a13 = 9.116257 * 10 ** -1
a23 = -4.821756
a33 = 1.232250 * 10 ** 3
a43 = -2.225358 * 10 ** 4
a53 = 1.161743e+5
pprh, tprh = 1, 1

def Ffunc(Y0, pprh0, tprh0):
    th = 1 / tprh0
    Ah = 0.06125 * th * np.exp(-1.2 * (1 - th) ** 2)
    Bh = th * (14.76 - 9.76 * th + 4.58 * th * th)
    Ch = th * (90.7 - 242.2 * th + 42.4 * th * th)
    Dh = 2.18 + 2.82 * th
    F0 = - Ah * pprh0 + (Y0 + Y0 ** 2 + Y0 ** 3 - Y0 ** 4) / (1 - Y0) ** 3 - Bh * Y0 ** 2 + Ch * Y0 ** Dh
    return F0

pointsnum = 31
pointsnumd = 31
depth = np.linspace(0, MD, pointsnumd)
dh = MD / (pointsnumd - 1)

def TPRf(q0):
    ql = q0
    qg = ql * GLR  # scf/day
    qw = ql / 100 * WCT
    usl = ql * 5.615 / 86400 / Area
    gsl = ((ql - qw) * gso + qw * gsw) / ql
    qm = gsl * 62.4 * ql * 5.615 + 0.0765 * gsg * qg
    mul = (muo * (ql - qw) + muw * qw) / ql
    Nd = 120.872 * TiD / 12 * (62.4 * gsl / s) ** 0.5
    Nl = 0.15726 * mul * (1 / (62.4 * gsl * s ** 3)) ** 0.25

    CNl = 10 ** (a11 + a21 * (np.log10(Nl) + 3) +
                 a31 * (np.log10(Nl) + 3) ** 2 +
                 a41 * (np.log10(Nl) + 3) ** 3 +
                 a51 * (np.log10(Nl) + 3) ** 4)

    dpdzh = 0
    presh = pwh
    for h in depth:
        temph = Twh + (Tbh - Twh) / MD * h
        tprh = (temph + 460) / tpc
        presh = presh + dpdzh * dh
        pprh = presh / ppc

        lnmuh = lnmuf(tprh, pprh)

        viscbaseh = (1.709 / 100000 - 2.062 / 1000000 * gsg) * temph + \
                    8.188 / 1000 - 6.15 / 1000 * np.log10(gsg)
        mugh = viscbaseh / tprh * np.exp(lnmuh)

        th = 1 / tprh
        Ah = 0.06125 * th * np.exp(-1.2 * (1 - th) ** 2)
        Bh = th * (14.76 - 9.76 * th + 4.58 * th * th)
        Ch = th * (90.7 - 242.2 * th + 42.4 * th * th)
        Dh = 2.18 + 2.82 * th

        Yh = fsolve(Ffunc, Yh0, args=(pprh, tprh))[0]

        zh = Ah * pprh / Yh
        usgh = 1 / Area * qg * zh * (460 + temph) / (460 + 60) * (14.7 / presh) / 86400

        #umh = usgh + usl
        Nvlh = 1.938 * usl * (62.4 * gsl / s) ** 0.25
        Nvgh = 1.938 * usgh * (62.4 * gsl / s) ** 0.25
        CNh = Nvlh / Nvgh ** 0.575 * (presh / 14.7) ** 0.1 * CNl / Nd
        ulfh = a12 + a22 * (np.log10(CNh) + 6) + a32 * (np.log10(CNh) + 6) ** 2 + \
               a42 * (np.log10(CNh) + 6) ** 3 + a52 * (np.log10(CNh) + 6) ** 4
        NNNh = Nvgh * Nl ** 0.38 / Nd ** 2.14
        indexh = (NNNh - 0.012) / abs(NNNh - 0.012)
        modifh = (1 - indexh) / 2 * 0.012 + (1 + indexh) / 2 * NNNh
        phih = a13 + a23 * modifh + a33 * modifh ** 2 + a43 * modifh ** 3 + a53 * modifh ** 4
        ylh = ulfh * phih
        NReh = 0.022 * qm / (TiD * mul ** ylh * mugh ** (1 - ylh))
        fh = 1/(-4 * np.log10(e/3.7065 - 5.0452 / NReh * np.log10(e**1.1098/2.8257 + (7.149 / NReh)**0.8981)))**2
        rhogh = 28.97 * gsg * presh / zh / 10.73 / (460 + th)
        rhoavrh = ylh * gsl * 62.4 + (1 - ylh) * rhogh
        dpdzh = 1 / 144 * (rhoavrh + fh * qm ** 2 / 7.413 / 10000000000 / (TiD / 12) ** 5 / rhoavrh)

    return presh


def IPRf(q0):
    global pr, J, pb, qb, qmax

    if q0 < qb:
        IPR0 = pr - q0 / J
    else:
        IPR0 = 0.125 * pb * ((81 - 80 * (q0 - qb) / (qmax - qb))**0.5 - 1)
    return IPR0


pointsnumq = pointsnum
rates = np.linspace(0, qmax, pointsnumq)
IPR = []
TPR = []

for ql in rates:
    if ql == 0:
        IPRq = IPRf(ql)
        TPRq = np.nan
    else:
        IPRq = IPRf(ql)
        TPRq = TPRf(ql)
    IPR.append(IPRq)
    TPR.append(TPRq)

df64 = pd.DataFrame({
    'rates': rates,
    'IPR': IPR,
    'TPR': TPR,
})

def nodalf64(q0):
    nodalf1 = TPRf(q0) - IPRf(q0)
    return nodalf1


qop64 = fsolve(nodalf64, 1000)[0]
pop64 = TPRf(qop64)
print(f'Operating Liquid Rate = {qop64:.2f} stb/d, Operating Bottom-hole Pressure {pop64:.2f} psi')

print('Writing results to Excel file.')
curdate = cur_date()
path_res = f'Nodal_out'
file_out_xls = f'{path_res}/64 BottomHoleNodalOil-HB-{curdate}.xlsx'
file_out_fig = f'{path_res}/64 BottomHoleNodalOil-HB-{curdate}.png'

with pd.ExcelWriter(file_out_xls) as writer:
    df64.to_excel(writer, sheet_name='Nodal Oil-HB')

fig, ax = plt.subplots(figsize=figsize)  # Create a figure containing a single axes.
ax.plot(df64['rates'], df64['IPR'], color='k', linewidth=2, linestyle='-', label='IPR (Vogel)')
ax.plot(df64['rates'], df64['TPR'], color='b', linewidth=1, linestyle='-', label='TPR (Hagedorn-Brown)')
ax.plot([qop64, qop64], [0, pop64],
        color='r', linewidth=2, linestyle=':', label=f'Operating Liquid Rate={qop64:.0f} stb/d')
ax.plot([0, qop64], [pop64, pop64],
        color='r', linewidth=2, linestyle=':', label=f'Operating Bottom-hole Pressure {pop64:.0f} psia')
ax.set_title('Oil Bottom Hole Nodal Hagedorn-Brown Correlation')
ax.set_xlim([0.0, qgraph])
ax.set_ylim([0, pr])
ax.set_xlabel('Liquid Production Rate (stb/d)')
ax.set_ylabel('Bottom Hole Pressure (psia)')
ax.grid(True)
plt.legend()
plt.savefig(file_out_fig)
plt.show()
