"""Gas-side convection inputs for ANSYS Mechanical; no solid thermal solver.
CSV coordinates: x = axial, r = radius, both in metres.
T_aw is the convection bulk/recovery temperature, not solid wall temperature.
HTC is convective only: gas radiation is not included.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import set_rocketcea_data_dir

# ============================================================
# USTAWIENIA UZYTKOWNIKA - jednostki SI, katy w stopniach
# ============================================================
OXIDIZER = 'N2O'
FUEL = 'Isopropanol'
F_t = 300.0          # [N] wymagany ciag - wyznacza srednice gardla
P_ch = 20e5         # [Pa] cisnienie komory (stagnacyjne)
MR = 2.35           # [-] stosunek masowy utleniacz/paliwo
P3 = 101325.0       # [Pa] cisnienie otoczenia
eta_n = 0.98        # [-] wspolczynnik predkosci, jak w oryginale
eta_cs = 0.85       # [-] wspolczynnik osiagow, jak w oryginale

T_WALL_REF = 300.0  # [K] temperatura MATERIALU tylko do poprawki Bartza
N_AX = 1500        # liczba probek + dokladne punkty polaczen

d_ch = 0.032        # [m] srednica cylindrycznej komory
L_ch = 0.090        # [m] dlugosc cylindrycznej komory
R1 = 0.0048         # [m] promien komora -> odcinek zbiezny
R2 = 0.0030         # [m] promien odcinek zbiezny -> wiekszy cylinder
BETA_DEG = 45.0     # [deg] kat odcinka zbieznego WZGLEDEM OSI
PRE_THROAT_RATIO = 1.10  # [-] SREDNICA wiekszego cylindra / srednica gardla
L_pre = 0.020       # [m] dlugosc wiekszego cylindra przed fazka
ALPHA_DEG = 30.0    # [deg] kat fazki WZGLEDEM OSI, nie kat wierzcholkowy
L_th = 0.003        # [m] dlugosc koncowego cylindra (wlasciwego gardla)

# Bartz wymaga skali krzywizny. Fazka ma ostre krawedzie, wiec nie ma
# jednoznacznego promienia gardla. To jawny parametr modelu, nie promien fazki.
# None: uzyj R2 jako przyblizonej skali, zachowujac podejscie oryginalu.
BARTZ_RADIUS_M = None  # [m] mozna podac niezaleznie od R2

# ============================================================
# OBLICZENIA - ponizej zwykle nie trzeba nic zmieniac
# ============================================================
R_univ = 8.31446261815324
g0 = 9.80665
beta = np.deg2rad(BETA_DEG)
alpha = np.deg2rad(ALPHA_DEG)


def require_positive(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


for name in ['F_t', 'P_ch', 'MR', 'T_WALL_REF', 'd_ch', 'L_ch',
             'R1', 'R2', 'L_pre', 'L_th']:
    require_positive(name, globals()[name])
if not np.isfinite(P3) or P3 < 0 or P3 >= P_ch:
    raise ValueError('Require 0 <= ambient pressure P3 < chamber pressure P_ch')
if not (0 < eta_n <= 1 and 0 < eta_cs <= 1):
    raise ValueError('Efficiency factors must be in (0, 1]')
if not (0 < BETA_DEG < 90 and 0 < ALPHA_DEG < 90):
    raise ValueError('Angles must be between 0 and 90 degrees, relative to axis')
if not np.isfinite(PRE_THROAT_RATIO) or PRE_THROAT_RATIO <= 1:
    raise ValueError('PRE_THROAT_RATIO must be greater than 1 (diameter ratio)')

# Separate working files; copying this .py alone is sufficient.
_CEA_WORK = Path(__file__).resolve().parent / 'cea_work'
_CEA_WORK.mkdir(parents=True, exist_ok=True)
set_rocketcea_data_dir(str(_CEA_WORK), do_print=False)
ispObj = CEA_Obj(
    oxName=OXIDIZER, fuelName=FUEL, isp_units='sec', cstar_units='m/s',
    pressure_units='Pa', temperature_units='K', sonic_velocity_units='m/s',
    enthalpy_units='J/kg', density_units='kg/m^3', specific_heat_units='J/kg-K',
    viscosity_units='millipoise', thermal_cond_units='W/cm-degC')

# No divergent nozzle: exit area equals final throat area, Ae/At = 1.
# Keep the original sizing method and efficiency convention, but use the
# actual sonic exit instead of a virtual nozzle expanded to ambient pressure.
eps = 1.0
c_star = ispObj.get_Cstar(Pc=P_ch, MR=MR)
pc_over_pe = ispObj.get_PcOvPe(Pc=P_ch, MR=MR, eps=eps,
                              frozen=1, frozenAtThroat=0)
Pe = P_ch / pc_over_pe  # calculated sonic exit pressure, no longer an input
if P3 >= Pe:
    raise ValueError('Ambient pressure is too high for the assumed choked flow')
_, _, a_e = ispObj.get_SonicVelocities(Pc=P_ch, MR=MR, eps=eps,
                                      frozen=1, frozenAtThroat=0)
v_e = a_e  # M=1 at final throat / exit
Isp = eta_cs * (eta_n*v_e + eps*c_star*(Pe-P3)/P_ch) / g0
require_positive('Isp', Isp)
T_ch = ispObj.get_Tcomb(Pc=P_ch, MR=MR)
mdot = F_t / (Isp*g0)
mw, k_gas = ispObj.get_Chamber_MolWt_gamma(Pc=P_ch, MR=MR, eps=eps)
A_th = mdot / P_ch * np.sqrt(
    R_univ*T_ch/(mw/1e3) /
    (k_gas*(2.0/(k_gas+1.0))**((k_gas+1.0)/(k_gas-1.0))))
d_th = np.sqrt(4.0*A_th/np.pi)
d_pre = PRE_THROAT_RATIO*d_th
r_b2 = R2  # retained name for the Bartz reference scale
bartz_radius = R2 if BARTZ_RADIUS_M is None else BARTZ_RADIUS_M
require_positive('BARTZ_RADIUS_M', bartz_radius)
if d_pre >= d_ch:
    raise ValueError('Chamber diameter must exceed pre-throat cylinder diameter')

# Seven segments, tangent circular fillets at the main converging section.
r_ch, r_pre, r_th = d_ch/2, d_pre/2, d_th/2
x1 = L_ch
x2 = R1*np.sin(beta)
y2 = r_ch-R1*(1-np.cos(beta))
y3 = r_pre+R2*(1-np.cos(beta))
if y2 <= y3:
    raise ValueError('No space for straight converging segment: reduce R1/R2 or angle, '
                     'or increase chamber diameter')
x3 = (y2-y3)/np.tan(beta)
x4 = R2*np.sin(beta)
x5 = L_pre
x6 = (r_pre-r_th)/np.tan(alpha)  # chamfer length derived from angle/diameters
JOINS = np.cumsum([0., x1, x2, x3, x4, x5, x6, L_th])
x_throat = float(JOINS[6])
L_tot = float(JOINS[-1])
SEGMENT_NAMES = ['Chamber', 'R1', 'Convergent', 'R2',
                 'Pre-throat', 'Chamfer', 'Throat']


def chmb_curve(x):
    if not 0 <= x <= L_tot:
        return np.nan
    if x < JOINS[1]:
        return r_ch
    if x < JOINS[2]:
        u = x-JOINS[1]
        return r_ch-R1+np.sqrt(max(0., R1**2-u**2))
    if x < JOINS[3]:
        return y2-(x-JOINS[2])*np.tan(beta)
    if x < JOINS[4]:
        u = JOINS[4]-x
        return r_pre+R2-np.sqrt(max(0., R2**2-u**2))
    if x < JOINS[5]:
        return r_pre
    if x < JOINS[6]:
        return r_pre-(x-JOINS[5])*np.tan(alpha)
    return r_th


def area_mach_function(M, k):
    expo = (k + 1.0) / (2.0 * (k - 1.0))
    return (1.0 / M) * ((1.0 + ((k - 1.0) / 2.0) * M**2) / ((k + 1.0) / 2.0))**expo


def mach_from_area_ratio(area_ratio, k, supersonic=False, tol=1e-8, max_iter=200):
    if abs(area_ratio-1.0) <= 1e-10:
        return 1.0

    if supersonic:
        lo, hi = 1.000001, 20.0
    else:
        lo, hi = 1e-6, 0.999999

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_lo = area_mach_function(lo, k) - area_ratio
        f_mid = area_mach_function(mid, k) - area_ratio

        if abs(f_mid) < tol:
            return mid

        if f_lo * f_mid < 0.0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


def recovery_temperature(T_0, M, k, Pr):
    r_rec = Pr ** (1.0 / 3.0)
    num = 1.0 + r_rec * 0.5 * (k - 1.0) * M**2
    den = 1.0 + 0.5 * (k - 1.0) * M**2
    return T_0 * num / den


cp_gas, mu_gas, _, Pr_gas = ispObj.get_Chamber_Transport(
    Pc=P_ch, MR=MR, eps=eps, frozen=1)
mu_gas /= 10000.0  # millipoise -> Pa*s

def bartz_sigma(T_wg, T_0, k, M):
    term_M = 1.0 + 0.5 * (k - 1.0) * M**2
    base = 0.5 * (T_wg / T_0) * term_M + 0.5
    sigma = 1.0 / (base**0.68 * term_M**0.12)
    return sigma


def bartz_htc(T_wg, T_0, k, M, mu, cp, Pr, mdot, A_th, d_th, r_local, A_local):
    sigma = bartz_sigma(T_wg, T_0, k, M)
    hg = (
        (0.026 / (d_th**0.2))
        * ((mu**0.2) * cp / (Pr**0.6))
        * ((mdot / A_th)**0.8)
        * ((d_th / r_local)**0.1)
        * ((A_th / A_local)**0.9)
        * sigma
    )
    return hg


def calculate_profile(t_wall_ref=T_WALL_REF, n_ax=N_AX):
    if not np.isfinite(t_wall_ref) or t_wall_ref <= 0:
        raise ValueError("Wall reference temperature must be positive [K]")
    if n_ax < 2:
        raise ValueError("At least two axial samples are required")
    # Every segment receives samples, including a short chamfer.
    # Joins are included exactly and coordinates are shared by all outputs.
    x = np.unique(np.concatenate([np.linspace(0., L_tot, n_ax),
        *[np.linspace(a, b, 12) for a, b in zip(JOINS[:-1], JOINS[1:])]]))
    r = np.array([chmb_curve(float(v)) for v in x])
    area = np.pi*r**2
    ratio = area/A_th
    if np.any(~np.isfinite(r)) or np.any(r <= 0) or np.any(ratio < 1.-1e-8):
        raise ValueError("Invalid inner geometry / throat area")
    mach = np.array([
        1.0 if x_throat <= v <= x_throat+L_th else
        mach_from_area_ratio(ar, k_gas, supersonic=False)
        for v, ar in zip(x, ratio)
    ])
    t_aw = recovery_temperature(T_ch, mach, k_gas, Pr_gas)
    htc = bartz_htc(t_wall_ref, T_ch, k_gas, mach, mu_gas, cp_gas,
                    Pr_gas, mdot, A_th, d_th, bartz_radius, area)
    data = np.column_stack([x, r, htc, t_aw])
    if not np.all(np.isfinite(data)) or np.any(data[:, 2:] <= 0):
        raise ValueError("Invalid HTC or temperature results")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-temperature', type=float, default=T_WALL_REF,
                        help='Solid wall reference temperature for Bartz [K]')
    parser.add_argument('--points', type=int, default=N_AX)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'thermal_results')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    data = calculate_profile(args.wall_temperature, args.points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir/'wall_convection_SI.csv', data, delimiter=',',
               header='x_m,r_m,htc_W_m2_K,T_aw_K', comments='', fmt='%.12e')
    np.savez(args.output_dir/'wall_convection_SI.npz',
             x_m=data[:,0], r_m=data[:,1], htc_W_m2_K=data[:,2], T_aw_K=data[:,3],
             T_wall_ref_K=args.wall_temperature)
    try:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                                 layout='constrained')
    except Exception as exc:
        print(f'Interactive plots unavailable ({type(exc).__name__}); saving PNG files.')
        plt.switch_backend('Agg')
        args.no_show = True
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                                 layout='constrained')
    for ax, col, label in zip(axes, [1,2,3],
            ['r [m]', 'HTC [W/(m^2 K)]', 'T_aw [K]']):
        ax.plot(data[:,0], data[:,col])
        ax.set_ylabel(label)
        ax.grid(True)
        ax.axvspan(x_throat, x_throat+L_th, color='grey', alpha=.15)
    axes[-1].set_xlabel('x [m]')
    fig.suptitle(f'Gas-side convection; Bartz wall reference {args.wall_temperature:g} K')
    fig.savefig(args.output_dir/'wall_convection_SI.png', dpi=180)
    geometry_fig, ax = plt.subplots(figsize=(12, 5))
    geometry_fig.subplots_adjust(bottom=.22, top=.86)
    ax.plot(data[:,0], data[:,1], color='tab:blue')
    ax.plot(data[:,0], -data[:,1], color='tab:blue')
    ax.axhline(0, color='grey', linewidth=.8, linestyle='--')
    ax.axvspan(x_throat, L_tot, color='grey', alpha=.15)
    for i, (a, b) in enumerate(zip(JOINS[:-1], JOINS[1:]), start=1):
        ax.annotate(str(i), ((a+b)/2, chmb_curve((a+b)/2)),
                    xytext=(0, 12 if i%2 else 28), textcoords='offset points',
                    ha='center', fontsize=9,
                    arrowprops=dict(arrowstyle='-', color='grey', linewidth=.6))
    ax.set(xlabel='x [m]', ylabel='r [m]',
           title=f'Inner profile | Dt={d_th*1000:.3f} mm | '
                 f'Dpre={d_pre*1000:.3f} mm | chamfer={ALPHA_DEG:g} deg to axis')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=.3)
    geometry_fig.text(.5, .03, '1 Chamber   2 R1   3 Convergent   4 R2   '
                      '5 Pre-throat   6 Chamfer   7 Final throat', ha='center', fontsize=9)
    geometry_fig.savefig(args.output_dir/'geometry_SI.png', dpi=180)
    np.savetxt(args.output_dir/'geometry_segments_SI.csv',
               np.column_stack([np.arange(1,8), JOINS[:-1], JOINS[1:],
                                np.diff(JOINS)]), delimiter=',',
               header='segment,x_start_m,x_end_m,length_m', comments='',
               fmt=['%d','%.12e','%.12e','%.12e'])
    (args.output_dir/'settings.txt').write_text(
        f'T_wall_ref_K={args.wall_temperature}\n'
        f'Rows={len(data)}\n'
        f'd_th_m={d_th:.12g}\nd_pre_m={d_pre:.12g}\n'
        f'F_t_N={F_t}\nP_ch_Pa={P_ch}\nMR={MR}\n'
        f'P3_Pa={P3}\nPe_calculated_Pa={Pe:.12g}\n'
        f'Isp_s={Isp:.12g}\nmdot_kg_s={mdot:.12g}\n'
        f'eta_n={eta_n}\neta_cs={eta_cs}\n'
        f'd_ch_m={d_ch}\nL_ch_m={L_ch}\nR1_m={R1}\nR2_m={R2}\n'
        f'beta_deg={BETA_DEG}\nalpha_deg={ALPHA_DEG}\n'
        f'pre_throat_diameter_ratio={PRE_THROAT_RATIO}\n'
        f'L_pre_m={L_pre}\nL_th_m={L_th}\nL_chamfer_m={x6:.12g}\n'
        f'bartz_reference_radius_m={bartz_radius}\n'
        'Sizing=original method with sonic exit, Ae/At=1\n'
        'Flow=quasi-1D isentropic; no friction/heat feedback\n'
        'Bartz=engineering approximation for chamfer geometry\n'
        'Temperature=T_aw (convection bulk/recovery temperature)\n'
        'HTC=convective Bartz coefficient; radiation excluded\n'
        'Coordinates=x axial, r radial, units=m\n', encoding='utf-8')
    print(f'Calculated throat diameter: {d_th*1000:.3f} mm; pre-throat: {d_pre*1000:.3f} mm')
    print(f'Saved {len(data)} rows to {args.output_dir.resolve()}')
    print(f'HTC range: {data[:,2].min():.6g} .. {data[:,2].max():.6g} W/(m^2 K)')
    print(f'T_aw range: {data[:,3].min():.6g} .. {data[:,3].max():.6g} K')
    if not args.no_show:
        plt.show()
    plt.close(fig)
    plt.close(geometry_fig)


if __name__ == '__main__':
    main()
