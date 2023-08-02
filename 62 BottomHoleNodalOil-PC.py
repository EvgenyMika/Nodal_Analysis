"""
Petroleum Production Engineering
Computer-Assisted Approach
- Boyun Guo, Ph.D.
- William C. Lyons, Ph.D.
- Ali Ghalambor, Ph.D.

Chapter 6: Well Deliverability
Example Problem 6.2
Description: This program calculates the operating point using the Poettmann–Carpenter method with bottom-hole node.
"""
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from Module_File import cur_date

# Input data
pr = 3000  # Reservoir pressure: 3000 psia
TiD = 1.66  # Tubing ID: 1.66 in.
pwh = 500  # Wellhead pressure: 500 psia
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
TD = 5000  # Tubing shoe depth: 5,000 ft
Tbh = 150  # Bottom-hole temperature: 150 8F

# My addition to the given data
Bw = 1.05
pb = 800

# Solution for Example Problem 6.2
print(f'Solution for Example Problem 6.2')
figsize = (8, 6)

# Oil specific gravity
gso = 141.5 / (131.5 + API)

# Water-oil Ratio
WOR = WC / (1 - WC)

# Gas-oil Ratio
GOR = GLR * (1 + WOR)

qb = J * (pr - pb)
qmax = J * (pr - pb + pb / 1.8)
print(f'qb = {qb:.1f}, qmax = {qmax:.1f} stb/d')


def Pseudof(gsg0, N20, CO20, H2S0):
    """
    Pseudo critical P and T
    :param gsg0: Gas-specific gravity: 1 for air
    :param N20: N2 content in gas, mole fraction
    :param CO20: CO2 content in gas, mole fraction
    :param H2S0: H2S content in gas, mole fraction
    :return: [Ppc, Tpc]
    """

    Ppc0 = 678 - 50 * (gsg0 - 0.5) - 206.7 * N20 + 440 * CO20 + 606.7 * H2S0  # psia
    Tpc0 = 326 + 315.7 * (gsg0 - 0.5) - 240 * N20 - 83.3 * CO2 + 133.3 * H2S0  # oR
    return [Ppc0, Tpc0]


Ppc = Pseudof(gsg, N2, CO2, H2S)[0]
Tpc = Pseudof(gsg, N2, CO2, H2S)[1]
print(f'Ppc = {Ppc} psia, Tpc = {Tpc} oR')


def Mstbof(q0):
    global gso, gsw, gsg, GLR, WC
    if q0 == 0:
        Mstbo0 = 0
    else:
        qg = q0 * GLR  # scf/d
        qw = q0 * WC  # bbl/d
        qo = q0 - qw
        GOR = qg / qo  # scf/stb
        WOR = qw / qo
        Mstbo0 = 350.17 * (gso + WOR * gsw) + GOR * 0.0765 * gsg
    return Mstbo0


def Rsf(p0, T0):
    global gsg, API
    Rs = gsg * (p0 / 18 * 10 ** (0.0125 * API) / 10 ** (0.00091 * T0)) ** 1.2048
    return Rs


def Bof(Rs0, T0):
    """
    Oil formation volume factor, rb /stb
    :param Rs:
    :param T:
    :return: Bo, rb /stb
    """
    global gsg, gso
    Bo = 0.971 + 0.000147 * (Rs0 * (gsg / gso) ** 0.5 + 1.25 * T0) ** 1.175
    return Bo


def zf(Ppc0, Tpc0, p0, T0):
    """
    # Z-factor function
    Ppc - psi,
    Tpc - oR,
    p - psia,
    T - F
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


def Vmf(Bo0, Rs0, p0, T0):
    """
    # Volume associated with 1 stb of oil
    """
    global Bw, Rs, Ppc, Tpc, GOR, WOR
    Vm0 = 5.615 * (Bo0 + WOR * Bw) + (GOR - Rs0) * (14.7 / p0) * (T0 + 460) / 520 * (zf(Ppc, Tpc, p0, T0) / 1.0)
    return Vm0


def ftf(q1):  # Friction term
    global TiD, WC
    ql1 = q1
    qo1 = q1 * (1 - WC)
    Drv0 = 1.4737 * 10 ** (-5) * Mstbof(ql1) * qo1 / (TiD / 12)

    # Friction factor
    fm0 = 4 * 10 ** (1.444 - 2.5 * np.log10(Drv0))

    # Friction term
    ftq0 = fm0 * qo1 ** 2 * Mstbof(ql1) ** 2 / 7.4137 / 10 ** 10 / (TiD / 12) ** 5

    return ftq0


# Error in depth
'''def TPRf1(px):
    global pwh, Twh, Tbh, gsg, API, gso, WOR, Bw, GOR, Ppc, Tpc, SGR, TD

    q1 = (J * (pr - px) if px > pb else (qb + qmax * (1 - 0.2 * (px / pb) - 0.8 * (px / pb) ** 2)))
    ft = ftf(q1)

    # wellhead
    RsWH = Rs(pwh, Twh)  # Solution gas-oil ratio at wellhead
    BoWH = Bo(RsWH, Twh)  # Oil formation volume factor at wellhead
    VmWH = Vm(BoWH, RsWH, pwh, Twh)  # Volume associated with 1 stb of oil at wellhead
    rhoWH = Mstbof(q1) / VmWH  # Fluid density at wellhead

    # bottom hole
    RsBH = Rs(px, Tbh)  # Solution gas-oil ratio at bottom hole
    BoBH = Bo(RsBH, Tbh)  # Oil formation volume factor at bottom hole
    VmBH = Vm(BoBH, RsBH, px, Tbh)  # Volume associated with 1 stb of oil at bottom hole
    rhoBH = Mstbof(q1) / VmBH  # Fluid density at wellhead
    rhoavr = (rhoWH + rhoBH) / 2
    dif = 144 * (px - pwh) / (rhoavr + ft / rhoavr) - TD
    return dif'''


def TPRf(px, q0):
    global pwh, Twh, Tbh, gsg, API, gso, WOR, Bw, GOR, Ppc, Tpc, SGR, TD

    ft = ftf(q0)

    # wellhead
    RsWH = Rsf(pwh, Twh)  # Solution gas-oil ratio at wellhead
    BoWH = Bof(RsWH, Twh)  # Oil formation volume factor at wellhead
    VmWH = Vmf(BoWH, RsWH, pwh, Twh)  # Volume associated with 1 stb of oil at wellhead
    rhoWH = Mstbof(q0) / VmWH  # Fluid density at wellhead

    # bottom hole
    RsBH = Rsf(px, Tbh)  # Solution gas-oil ratio at bottom hole
    BoBH = Bof(RsBH, Tbh)  # Oil formation volume factor at bottom hole
    VmBH = Vmf(BoBH, RsBH, px, Tbh)  # Volume associated with 1 stb of oil at bottom hole
    rhoBH = Mstbof(q0) / VmBH  # Fluid density at wellhead

    rhoavr = (rhoWH + rhoBH) / 2

    diff = 144 * (px - pwh) / (rhoavr + ft / rhoavr) - TD
    return diff


def IPRVogelf(q0):
    global pr, J, pb, qb, qmax

    if q0 < qb:
        IPR0 = pr - q0 / J
    else:
        IPR0 = 0.125 * pb * ((81 - 80 * (q0 - qb) / (qmax - qb))**0.5 - 1)

    return IPR0


def nodalf62(q0):
    nodalf1 = IPRVogelf(q0) - fsolve(TPRf, pr, args=q0)[0]
    return nodalf1


qop62 = fsolve(nodalf62, qb)[0]
pop62 = IPRVogelf(qop62)
print(f'Operation Liquid Rate = {qop62:.1f} stb/d')
print(f'Operation Bottomhole Pressure Rate = {pop62:.1f} psia')

pointsnum = 101
rates62 = np.linspace(200, 3000, pointsnum)

TPR = []
IPR = []
for q0 in rates62:

    IPRq = IPRVogelf(q0)
    IPR.append(IPRq)

    TPRq = fsolve(TPRf, pr, args=q0)[0]
    TPR.append(TPRq)

# pwfv = np.linspace(0, pr, pointsnum)

df62 = pd.DataFrame({
    'rate': rates62,
    'IPR': IPR,
    'TPR': TPR,
})


print('Writing results to Excel file.')
curdate = cur_date()
path_res = f'Nodal_out'
file_out_xls = f'{path_res}/62 BottomHoleNodalOil-PC-{curdate}.xlsx'
file_out_fig = f'{path_res}/62 BottomHoleNodalOil-PC-{curdate}.png'

with pd.ExcelWriter(file_out_xls) as writer:
    df62.to_excel(writer, sheet_name='BottomholeNodalOil-PC')


fig, ax = plt.subplots(figsize=figsize)  # Create a figure containing a single axes.
ax.plot(df62['rate'], df62['IPR'], color='k', linewidth=2, linestyle='-', label='IPR (Vogel)')
ax.plot(df62['rate'], df62['TPR'], color='b', linewidth=1, linestyle='--', label='TPR (Poettmann-Carpenter)')
ax.plot([qop62, qop62], [0, pop62],
        color='r', linewidth=1, linestyle=':', label=f'Operating Liquid Rate = {qop62:.0f}, stb/d')
ax.plot([0, qop62], [pop62, pop62],
        color='r', linewidth=1, linestyle=':', label=f'Operating BottomHole Pressure = {pop62:.0f}, psia')
ax.set_title('Oil Bottom Hole Nodal by Poettmann-Carpenter Method')
ax.set_xlim([0.0, qmax])
ax.set_ylim([0, pr])
ax.grid(True)
ax.set_xlabel('Liquid Production Rate, stb/d')
ax.set_ylabel('Bottom Hole Pressure, psia')
plt.legend()
plt.savefig(file_out_fig)
plt.show()
