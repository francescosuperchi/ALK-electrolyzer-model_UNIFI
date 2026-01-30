# -*- coding: utf-8 -*-
"""

MIT License

Copyright (c) 2026 Università degli Studi di Firenze

Created: 2026-01-30

"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

# --- Electrochemistry constants ---
V_TN_CELL = 1.48  # [V] thermoneutral per cell (use a better T/P law if desired)

# Temperature-dependent Faraday efficiency data (user supplied)
_FARADAY_TEMP_POINTS = np.array([40.0, 60.0, 80.0], dtype=float)
_FARADAY_F1_POINTS = np.array([150.0, 200.0, 250.0], dtype=float)
_FARADAY_F2_POINTS = np.array([0.99, 0.985, 0.98], dtype=float)
_PLOT_CURRENT_POINTS = 200

def _faraday_coeffs_from_temp(
                                    temp_c: float,
                                    temps: Optional[np.ndarray],
                                    f1_curve: Optional[np.ndarray],
                                    f2_curve: Optional[np.ndarray],
                                    fallback_f1: float,
                                    fallback_f2: float,
                                ) -> Tuple[float, float]:
    """
    Interpolate Faraday efficiency coefficients vs temperature.
    Falls back to the provided defaults if the tables are missing/misaligned.
    """
    if temps is None or f1_curve is None or f2_curve is None:
        return fallback_f1, fallback_f2

    temps = np.asarray(temps, dtype=float).ravel()
    f1_curve = np.asarray(f1_curve, dtype=float).ravel()
    f2_curve = np.asarray(f2_curve, dtype=float).ravel()

    if (
        temps.size == 0
        or f1_curve.size != temps.size
        or f2_curve.size != temps.size
    ):
        return fallback_f1, fallback_f2

    order = np.argsort(temps)
    temps = temps[order]
    f1_curve = f1_curve[order]
    f2_curve = f2_curve[order]

    clamped_temp = np.clip(temp_c, temps[0], temps[-1])
    f1 = np.interp(clamped_temp, temps, f1_curve)
    f2 = np.interp(clamped_temp, temps, f2_curve)
    return float(f1), float(f2)

class BaseElectroCell(ABC):
    """
    Common interface for electrolyzer cell types.
    Subclasses must implement build_curves() and faraday_efficiency().
    Arrays are defined after build_curves().
    """
    # numeric knobs (must exist on subclasses)
    iNumCurrent: int
    rA_cell: float           # cm^2
    rI_rated: float          # A/cm^2
    rT: float                # °C working temperature used for curve gen

    # universal constants
    rR: float = 8.314
    rF: float = 96485.0

    # outputs populated by build_curves()
    arCurrentDensity: Optional[np.ndarray] = None
    arE_min: Optional[np.ndarray] = None
    arR_cell: Optional[np.ndarray] = None
    arV_cell: Optional[np.ndarray] = None

    @abstractmethod
    def build_curves(self) -> "BaseElectroCell":
        ...

    @abstractmethod
    def faraday_efficiency(self, J: np.ndarray) -> np.ndarray:
        ...

@dataclass
class ElectroCellALK(BaseElectroCell):
    # geometry & grid
    iNumCurrent: int = 1000             # resolutin of performance curves
    
    area_m2: float = 0.1                # electrode active area in m2 - representative value
    rJ_rated: float = 0.8               # rated current density in A/cm2
    rV_cellNom: float = 2               # Nominal voltage of the cell [V]

    rA_cell: float = area_m2 * 1e4      # cell area in   cm^2      

    # temperature references °C
    rT_0: float = 70.0                  # Nominal operating temperature [°C]
    rT: float  = 70.0                   # Initial operating temperature [°C]

    # Electrochemical parameters (default values)
    pressure_bar: float = 30.0             # pressure of the produced H2 [bar]
    koh_mass_frac: float = 0.3             # 30 wt% of KOH in the electorlyte solution
    
    L_anode_m: float = 0.4e-3              # anode thickness [m]      - representative value
    L_cathode_m: float = 0.4e-3            # cathode thickness [m]    - representative value
    cond_Ni: float = 14.6e6                 # [S/m]

    # Empirical resistance contribution  https://doi.org/10.1016/S0360-3199(97)00069-4
    henao_prefactor: float = 0.37
    henao_temp_coeff: float = -0.011

    # Faraday eff params https://doi.org/10.3390/en13184792
    rF1: float = 0.05
    rF2: float = 0.99
    arFaradayTemp_C: np.ndarray = field(default_factory=lambda: _FARADAY_TEMP_POINTS.copy())
    arFaradayF1: np.ndarray = field(default_factory=lambda: _FARADAY_F1_POINTS.copy())
    arFaradayF2: np.ndarray = field(default_factory=lambda: _FARADAY_F2_POINTS.copy())

    def build_curves(self) -> "ElectroCellALK":
        ec = self
        ec.arCurrentDensity = np.linspace(0.0, ec.rJ_rated, ec.iNumCurrent)
        ec.arE_min = np.empty(ec.iNumCurrent)
        ec.arR_cell = np.empty(ec.iNumCurrent)
        ec.arV_cell = np.empty(ec.iNumCurrent)

        # Temperature conversions
        T_C = ec.rT
        T_K = T_C + 273.15

        # Constants
        R = ec.rR            # J/mol/K
        F = ec.rF            # C/mol
        n = 2.0              # n of electrons

        # Geometry/resistance terms
        S_a = ec.area_m2  #anode's surface                                          
        S_c = ec.area_m2  #cathodes's surface
        R_a = (1.0 / ec.cond_Ni) * (ec.L_anode_m / max(S_a, 1e-12))     # anode's resistance
        R_c = (1.0 / ec.cond_Ni) * (ec.L_cathode_m / max(S_c, 1e-12))   # cathode's resistance

        # Concentration / vapor pressure terms
        C_KOH = ec.koh_mass_frac
        
        # formulas from https://doi.org/10.1016/j.ijhydene.2012.07.015
        M = (C_KOH * (183.1221 - 0.56845 * T_K + 984.5679 * np.exp(C_KOH / 115.96277))) / (100.0 * 56.105)    # molarity of the solution                      
        P_w = (T_K ** -3.498) * np.exp(37.93 - (6426.32 / T_K)) * np.exp(0.016214 - 0.13802 * M + 0.19330 * np.sqrt(max(M, 0.0)))   # partial pressure of gaseous solution
        P_vapor = (T_K ** -3.4159) * np.exp(37.043 - (6275.7 / T_K))   # partial pressure of water vapour at temperature T
        pressure_Pa = ec.pressure_bar * 1e5      # pressure in Pa

        # transfer-charge coefficients defined with the Nickel electrode correlation 
        
        # formulation from https://doi.org/10.1016/j.ijhydene.2012.07.015
        alfa_a = 0.0675 + 0.00095 * T_K
        alfa_c = 0.1175 + 0.00095 * T_K
        
        # Tafel equation from https://doi.org/10.1016/j.ijhydene.2012.07.015
        b_a = 2.3026 * R * T_K / (n * F * max(alfa_a, 1e-12))
        b_c = 2.3026 * R * T_K / (n * F * max(alfa_c, 1e-12))

        # Reversible voltage in standard conditions from https://doi.org/10.3390/pr12122616
        V_rev0 = 1.50342 - 9.956e-4 * T_K + 2.5e-7 * (T_K ** 2)
        
        # Reversible voltage through Nernst equation 10.1149/1.2130044
        term_log = ((pressure_Pa - P_w) ** 1.5) * P_vapor / max(P_w, 1e-12)
        V_rev = V_rev0 + ((R * T_K) / (n * F)) * np.log(max(term_log, 1e-12))
        
        #anode and cathore exchange current densities  https://doi.org/10.1016/j.jpowsour.2013.10.086
        i_0a = 13.72491 - 0.09055*T_C + 0.0009055*T_C**2 * 10 # *10 to convert mA/cm2 to A/m2
        i_0c = 30.4 - 0.206*T_C + 0.00035*T_C**2 * 10         # *10 to convert mA/cm2 to A/m2

        for idx, J in enumerate(ec.arCurrentDensity):
            # Current conversions
            I_total = J * ec.rA_cell                # [A]
            i_1 = I_total / max(ec.area_m2, 1e-12)  # [A/m^2] = J*1e4

            if J <= 0.0:
                V_act = 0.0
            else:
                ratio_a = max(i_1, 1e-30) / max(i_0a, 1e-30)
                Vact_a = b_a * np.log10(ratio_a)
                
                ratio_c = max(i_1, 1e-30) / max(i_0c, 1e-30)
                Vact_c = b_c * np.log10(ratio_c)
                V_act = Vact_a + Vact_c

            R_Henao = (ec.henao_prefactor * np.exp(ec.henao_temp_coeff * T_C)) / (max(ec.area_m2, 1e-12) * 1e4)
            Res = R_a + R_c + R_Henao
            V_ohm = Res * I_total

            V_cell = V_rev + V_act + V_ohm
            
            ec.arE_min[idx] = V_rev
            ec.arR_cell[idx] = Res
            ec.arV_cell[idx] = V_cell

        return ec

    def faraday_efficiency(self, J: np.ndarray) -> np.ndarray:
        J_mA = J * 1e3  # convert from A/cm^2 to mA/cm^2 for the fitted curve
        f1, f2 = _faraday_coeffs_from_temp(
            self.rT,
            self.arFaradayTemp_C,
            self.arFaradayF1,
            self.arFaradayF2,
            self.rF1,
            self.rF2,
        )
        return (J_mA**2 / (f1 + J_mA**2)) * f2



#%% Plot of Behavior

def _cell_efficiency_vs_load(cell: BaseElectroCell) -> Tuple[np.ndarray, np.ndarray]:
    cell.build_curves()
    current_density = cell.arCurrentDensity
    current = current_density * cell.rA_cell
    faraday_eff = cell.faraday_efficiency(current_density)

    rLHV_H2 = 119_988.0  # J/g
    rMu = 2.01588        # g/mol
    rN = 2.0
    rLossDry = 0.03
    rConstantPart = rMu / cell.rF / rN * (1.0 - rLossDry)

    h2_dot = faraday_eff * rConstantPart * current
    p_total = current * cell.arV_cell
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = (rLHV_H2 * h2_dot) / p_total
    efficiency = np.nan_to_num(efficiency, nan=0.0, posinf=0.0, neginf=0.0)

    rated_power = max(float(p_total[-1]), 1e-6)
    load_pct = np.clip(p_total / rated_power, 0.0, None) * 100.0
    return load_pct, efficiency * 100.0

def plot_alk_efficiency_vs_load(
    ax=None,
    temps_C: Optional[np.ndarray] = None,
    title: str = "ALK Efficiency vs Load",
    xlim: Optional[Tuple[float, float]] = (0.0, 100.0),
    ylim: Optional[Tuple[float, float]] = (0.0, 70.0),
    min_temp: float = 40.0,
    max_temp: float = 80.0,
    n_temps: int = 15,
):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if temps_C is None:
        temps_C = np.linspace(min_temp, max_temp, n_temps)
    else:
        temps_C = np.asarray(temps_C, dtype=float).ravel()
        if temps_C.size == 0:
            temps_C = np.linspace(min_temp, max_temp, n_temps)
        min_temp = float(np.min(temps_C))
        max_temp = float(np.max(temps_C))

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5))

    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=min_temp, vmax=max_temp)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for T in temps_C:
        cell = ElectroCellALK()
        cell.rT = float(T)
        cell.iNumCurrent = _PLOT_CURRENT_POINTS
        load_pct, efficiency_pct = _cell_efficiency_vs_load(cell)
        color_val = cmap(norm(T))
        ax.plot(load_pct, efficiency_pct, color=color_val, linewidth=1.5, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Load [% of rated power]")
    ax.set_ylabel("Efficiency [% LHV]")
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Temperature [C]")

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    return ax

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ax = plot_alk_efficiency_vs_load()
    ax.figure.tight_layout()
    plt.show()









