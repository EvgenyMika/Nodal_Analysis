"""
Petroleum Production Engineering
Computer-Assisted Approach
- Boyun Guo, Ph.D.
- William C. Lyons, Ph.D.
- Ali Ghalambor, Ph.D.

Chapter 6: Well Deliverability
Example Problem 6.2
Description: This program calculates the operating point using the Poettmann–Carpenter method with wellhead node.
"""
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from Module_File import cur_date

# Input data
pr = 6000  # Reservoir pressure: 6000 psia
TiD = 3.5  # Tubing ID: 1.66 in.
J = 1  # Productivity index above bubble point: 1 stb/d-psi
GLR = 1000  # Producing gas–liquid ratio (GLR): 1,000 scf/stb
WC = 0.25  # Water cut: 25 %
API = 30  # Oil gravity: 30 8API
gsw = 1.05  # Water-specific gravity: 1.05, 1 for water
gsg = 0.65  # Gas-specific gravity: 0.65, 1 for air
N2 = 0  # N2 content in gas: 0 mole fraction
CO2 = 0  # CO2 content in gas: 0 mole fraction
H2S = 0  # H2S content in gas: 0 mole fraction
Twh = 100  # Wellhead temperature: 100 8F
Tbh = 150  # Bottom-hole temperature: 150 8F
TD = 12000  # Tubing shoe depth: 12000 ft
Bw = 1.0  # rb/stb
Choke = 64  # Choke size:	64	 1/64 in
Cc = 10  # Choke constant:	10
Cglrexp = 0.546  # Choke GLR exponent:	0.546
Csexp = 1.89  # Choke size exponent:	1.89

# My addition to the given data
pb = pr
#pwh = 1000
figsize = (10, 6)

# Solution for Example Problem 6.6
print(f'WellHead Oil Nodal analysis with Poettmann–Carpenter method.')
Tidft = TiD / 12

# Oil specific gravity
gso = 141.5 / (131.5 + API)

# Water-oil Ratio
WOR = WC / (1 - WC)

# Gas-oil Ratio
GOR = GLR * (1 + WOR)
qb = J * (pr - pb)
qmax = J * pr
print(f'qmax = {qmax:.1f} stb/d')


def Pseudo(gsg0, N20, CO20, H2S0):  # Pseudo critical P and T

    Ppc0 = 678 - 50 * (gsg0 - 0.5) - 206.7 * N20 + 440 * CO20 + 606.7 * H2S0  # psia
    Tpc0 = 326 + 315.7 * (gsg0 - 0.5) - 240 * N20 - 83.3 * CO2 + 133.3 * H2S0  # oR
    return [Ppc0, Tpc0]


Ppc = Pseudo(gsg, N2, CO2, H2S)[0]
Tpc = Pseudo(gsg, N2, CO2, H2S)[1]
print(f'Ppc = {Ppc:.0f} psia, Tpc = {Tpc:.0f} oR')

# Mass associated with 1 stb of oil
#Mstbo = 350.17 * (gso + WOR * gsw) + GOR * 0.0765 * gsg
#print(f'Mass associated with 1 stb of oil = {Mstbo}')


def Mstbof(q0):  # Mass associated with 1 stb of oil
    global gso, gsw, gsg, GLR, WC
    ql = q0
    qw0 = ql * WC  # Water rate bbl/d
    qo0 = ql - qw0  # Oil rate bbl/d
    qg0 = ql * GLR  # Gas rate scf/d
    WOR0 = qw0 / qo0
    GOR0 = qg0 / qo0  # scf/stb
    Mstbo = 350.17 * (gso + WOR0 * gsw) + GOR0 * 0.0765 * gsg
    return Mstbo


def Rsf(p0, T0):  # Solution gas ratio
    global gsg, API
    Rs = gsg * (p0 / 18 * 10 ** (0.0125 * API) / 10 ** (0.00091 * T0)) ** 1.2048
    return Rs


def Bof(Rs0, T0):  # Oil formation volume factor
    global gsg, gso
    Bo0 = 0.971 + 0.000147 * (Rs0 * (gsg / gso) ** 0.5 + 1.25 * T0) ** 1.175
    return Bo0


def zf(Ppc0, Tpc0, p0, T0):  # Z-factor
    """
    Z-factor function
    :param Ppc0:  Ppc - psi,
    :param Tpc0:  Tpc - oR,
    :param p0:    p - psia,
    :param T0:    T - F
    :return:      z
    """
    Ppr = p0 / Ppc0
    Tpr = (T0 + 460) / Tpc0
    A = 1.39 * (Tpr - 0.92) ** 0.5 - 0.36 * Tpr - 0.101
    B = (0.62 - 0.23 * Tpr) * Ppr + (0.066 / (Tpr - 0.86) - 0.037) * Ppr ** 2 + 0.32 / 10 ** (
            9 * (Tpr - 1)) * Ppr ** 6
    C = (0.132 - 0.32 * np.log10(Tpr))
    D = 10 ** (0.3106 - 0.49 * Tpr + 0.1824 * Tpr ** 2)
    z = A + (1 - A) / np.exp(B) + C * Ppr ** D
    return z


def Vmf(q0, Bo0, Rs0, p0, T0):  # Volume associated with 1 stb of oil
    global Bw, Rs, Ppc, Tpc
    ql0 = q0
    if ql0 == 0:
        Vm0 = 0
    else:
        qw0 = ql0 * WC  # Water rate bbl/d
        qo0 = ql0 - qw0  # Oil rate bbl/d
        qg0 = ql0 * GLR  # Gas rate scf/d
        WOR0 = qw0 / qo0
        GOR0 = qg0 / qo0  # scf/stb
        Vm0 = 5.615 * (Bo0 + WOR0 * Bw) + (GOR0 - Rs0) * (14.7 / p0) * (T0 + 460) / 520 * (zf(Ppc, Tpc, p0, T0) / 1.0)
    return Vm0


def fldensf(q0, Bo0, Rs0, p0, T0):  # Fluid density
    dens = Mstbof(q0) / Vmf(q0, Bo0, Rs0, p0, T0)
    return dens


def ftf(q1):  # Friction term
    global TiD, WC
    ql1 = q1
    # Inertial force (Drv)
    qo1 = ql1 * (1 - WC)
    Drv0 = 1.4737 * 10 ** (-5) * Mstbof(qo1) * qo1 / (TiD / 12)

    # Friction factor
    fm0 = 4 * 10 ** (1.444 - 2.5 * np.log10(Drv0))

    # Friction term
    ftq0 = fm0 * qo1 ** 2 * Mstbof(qo1) ** 2 / 7.4137 / 10 ** 10 / (TiD / 12) ** 5

    return ftq0


def CPRf(q0):
    global GLR, Cc, Choke, Csexp
    cpr = Cc * q0 * (GLR ** Cglrexp) / (Choke ** Csexp)
    return cpr


def WPRf(px, q0):
    global Twh, Tbh, gsg, API, gso, WOR, Bw, GOR, Ppc, Tpc, SGR, TD, Mstbo

    pwf0 = pr - q0 / J
    ft = ftf(q0)

    # well head
    RsWH = Rsf(px, Twh)  # Solution gas-oil ratio at wellhead
    BoWH = Bof(RsWH, Twh)  # Oil formation volume factor at wellhead
    VmWH = Vmf(q0, BoWH, RsWH, px, Twh)  # Volume associated with 1 stb of oil at wellhead
    MstboWH = Mstbof(q0)
    rhoWH = MstboWH / VmWH  # Fluid density at wellhead

    # bottom hole
    RsBH = Rsf(pwf0, Tbh)  # Solution gas-oil ratio at bottom hole
    BoBH = Bof(RsBH, Tbh)  # Oil formation volume factor at bottom hole
    VmBH = Vmf(q0, BoBH, RsBH, pwf0, Tbh)  # Volume associated with 1 stb of oil at bottom hole
    MstboBH = Mstbof(q0)
    rhoBH = MstboBH / VmBH  # Fluid density at bottom hole

    rhoavr = (rhoWH + rhoBH) / 2

    D = 144 * (pwf0 - px) / (rhoavr + ft / rhoavr) - TD
    return D


def nodalf66(q0):
    global pr, J
    pnod = CPRf(q0)
    WPR = fsolve(WPRf, pnod, args=q0)[0]
    nodalf66 = WPR - CPRf(q0)
    return nodalf66


pointsnum = 11
rates66 = np.linspace(0, qmax, pointsnum)

pwf = []
CPR = []
WPR = []
WPR0 = 0

for q0 in rates66:
    pwfi = pr - q0 / J
    pwf.append(pwfi)

    CPRi = CPRf(q0)
    CPR.append(CPRi)

    if q0 == 0:
        WPR.append(np.nan)

    else:
        WPRi = fsolve(WPRf, WPR0, args=q0)[0]
        if WPRi == WPR0:
            WPR.append(np.nan)
        else:
            WPR.append(WPRi)
        WPR0 = WPRi


df66 = pd.DataFrame({
    'rate': rates66,
    'pwf': pwf,
    'WPR': WPR,
    'CPR': CPR,
})


qop = fsolve(nodalf66, qmax/2)[0]
pop = CPRf(qop)
print(f'Operating Liquid Rate = {qop:.0f} stb/d, Operating Wellhead Pressure = {pop:.0f} psia')

print('Writing results to Excel file.')
curdate = cur_date()
path_res = f'Nodal_out'
file_out_xls = f'{path_res}/66 BottomHoleNodalOil-PC-{curdate}.xlsx'
file_out_fig = f'{path_res}/66 BottomHoleNodalOil-PC-{curdate}.png'

with pd.ExcelWriter(file_out_xls) as writer:
    df66.to_excel(writer, sheet_name='WellHeadNodalOil-PC')

fig, ax = plt.subplots(figsize=figsize)  # Create a figure containing a single axes.
ax.plot(df66['rate'], df66['WPR'], color='k', linewidth=2, linestyle='-', label='WPR')
ax.plot(df66['rate'], df66['CPR'], color='b', linewidth=1, linestyle='--', label='CPR')
ax.plot((qop, qop), (0, pop),
        color='r', linewidth=1, linestyle=':',
        label=f'Operating Liquid Rate = {qop:.0f} stb/d')
ax.plot((0.0, qop), (pop, pop),
        color='r', linewidth=1, linestyle=':',
        label=f'Operating Wellhead Pressure = {pop:.0f} psia')
ax.set_title('Oil Wellhead Nodal by Poettmann-Carpenter Method')
ax.set_xlim(0.0, qmax)
ax.set_ylim(0, pr/2)
ax.grid(True)
ax.set_xlabel('Liquid Production Rate (stb/d)')
ax.set_ylabel('Wellhead Pressure (psia)')
plt.legend()
plt.savefig(file_out_fig)
plt.show()
