"""Расчёт токсодозы при аварийном выбросе АХОВ — Сценарий 1.
Разрушение оборудования с выбросом всего объема АХОВ, образование
первичного облака, рассеяние первичного облака и воздействие
на окружающую среду.
"""

from __future__ import annotations

from dataclasses import dataclass
import numexpr as ne
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from reference_data import (
    get_OHV_values,
    get_coefficients_A1A2B1B2C3,
    get_coefficients_C1C2D1D2,
)
from plot_utils import plot_conc_doze_Ox, plot_dose_with_contours


# --------------------- dataclass параметров -------------------------- #


@dataclass
class ReleaseParams:
    """Группирует все численные параметры выброса."""

    Q: float
    M_mol: float
    T: float
    P: float
    U: float
    z0: float
    A1: float
    A2: float
    B1: float
    B2: float
    C3: float
    C1: float
    C2: float
    D1: float
    D2: float
    h: float = 0.0

    _R: float = 8.314  # газовая постоянная

    # Кэшируемые свойства --------------------------------------------------
    @property
    def rho_gas(self) -> float:
        return (self.P * self.M_mol) / (self._R * self.T)

    @property
    def R1(self) -> float:
        return (3 * self.Q / (4 * np.pi * self.rho_gas)) ** (1 / 3)


# --------------------- дисперсии σ(x) -------------------------------- #


def sigma_x(x: NDArray, p: ReleaseParams) -> NDArray:
    return p.C3 * x / np.sqrt(1 + 1.0e-4 * x)


def sigma_y(x: NDArray, p: ReleaseParams) -> NDArray:
    sx = sigma_x(x, p)
    return np.where(
        x / p.U >= 600,
        sx * (220.2 * 60 + x / p.U) / (220.2 * 60 + 600),
        sx,
    )


def g_x(x: NDArray, p: ReleaseParams) -> NDArray:
    return p.A1 * x**p.B1 / (1 + p.A2 * x**p.B2)


def f_z0_x(x: NDArray, p: ReleaseParams) -> NDArray:
    if p.z0 < 0.1:
        return np.log(p.C1 * x**p.D1 * (1 + p.C2 * x**p.D2))
    return np.log(p.C1 * x**p.D1 / (1 + p.C2 * x**p.D2))


def sigma_z(x: NDArray, p: ReleaseParams) -> NDArray:
    return f_z0_x(x, p) * g_x(x, p)


# --------------------- базовый знаменатель --------------------------- #


def denom(
    x: NDArray,
    p: ReleaseParams,
    sx: NDArray | None = None,
    sy: NDArray | None = None,
    sz: NDArray | None = None,
) -> NDArray:
    if sx is None:
        sx = sigma_x(x, p)
    if sy is None:
        sy = sigma_y(x, p)
    if sz is None:
        sz = sigma_z(x, p)
    return (8 / 3) * np.pi * p.R1**3 + (2 * np.pi) ** 1.5 * sx * sy * sz


# --------------------- C(x) и D(x) ----------------------------------- #


def c_max(x: NDArray, p: ReleaseParams) -> NDArray:
    sz = sigma_z(x, p)
    g0 = ne.evaluate(
        "exp(-h2 / (2 * sz ** 2))",
        {
            "h2": p.h**2,
            "sz": sz,
        },
    )
    return 2 * p.Q / denom(x, p, sz=sz) * g0


def d_max(x: NDArray, p: ReleaseParams) -> NDArray:
    sx, sz = sigma_x(x, p), sigma_z(x, p)
    g0 = ne.evaluate(
        "exp(-h2 / (2 * sz ** 2))",
        {
            "h2": p.h**2,
            "sz": sz,
        },
    )
    return (
        2
        * p.Q
        * np.sqrt(2 * np.pi)
        * sx
        / (p.U * denom(x, p, sx=sx, sz=sz))
        * g0
        * 1000
        / 60
    )


# --------------------- полноразмерные поля --------------------------- #


def c_field(x: NDArray, y: NDArray, t: float, p: ReleaseParams) -> NDArray:
    """C(x,y,t) для z=0 c использованием broadcasting."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    sx, sy, sz = sigma_x(X, p), sigma_y(X, p), sigma_z(X, p)
    den = denom(X, p, sx, sy, sz)
    expr = ne.evaluate(
        "Q * exp(-(X - U*t) ** 2 / (2*sx**2) - Y**2 / (2*sy**2))",
        {
            "Q": p.Q,
            "X": X,
            "Y": Y,
            "U": p.U,
            "t": t,
            "sx": sx,
            "sy": sy,
        },
    )
    gz = ne.evaluate(
        "exp(-h2 / (2*sz**2)) + exp(-h2 / (2*sz**2))",
        {
            "h2": p.h**2,
            "sz": sz,
        },
    )
    return expr * gz / den


def d_field(x: NDArray, y: NDArray, p: ReleaseParams) -> NDArray:
    X, Y = np.meshgrid(x, y, indexing="ij")
    sx, sy, sz = sigma_x(X, p), sigma_y(X, p), sigma_z(X, p)
    gn = ne.evaluate(
        "exp(-Y**2/(2*sy**2)) * (exp(-h2/(2*sz**2)) + exp(-h2/(2*sz**2)))",
        {
            "Y": Y,
            "sy": sy,
            "sz": sz,
            "h2": p.h**2,
        },
    )
    return (
        p.Q * np.sqrt(2 * np.pi) * sx / (p.U * denom(X, p, sx, sy, sz)) * gn * 1000 / 60
    )


# --------------------- основной сценарий ----------------------------- #


def run_scenario() -> None:
    """Запрашивает данные у пользователя, затем строит графики."""

    # --- ввод данных (оставлен без изменений) ------------------------ #
    allowed = [
        "аммиак",
        "мышьяковистый водород",
        "фтористый водород",
        "хлористый водород",
        "бромистый водород",
        "цианистый водород",
        "сероводород",
        "сероуглерод",
        "формальдегид",
        "фосген",
        "фтор",
        "хлор",
        "хлорциан",
        "окись углерода",
        "окись этилена",
    ]
    while True:
        aho = input("Введите название АХОВ: ").lower()
        if aho in allowed:
            break
        print("Неверный ввод. Пожалуйста, выберите из:", ", ".join(allowed))

    M_mol = get_OHV_values(aho, "M") * 1e-3
    PCt50 = get_OHV_values(aho, "PCt50")
    LCt50 = get_OHV_values(aho, "LCt50")

    Q1 = float(input("Масса АХОВ, кг: "))
    T = float(input("Температура хранения, °C: ")) + 273.15
    P = float(input("Давление, атм: ")) * 101_325
    z0 = float(input("Шероховатость поверхности, м: "))
    U = float(input("Скорость ветра, м/с: "))

    while True:
        svua = input(
            "Вертикальная устойчивость (изотермия/конвекция/инверсия): "
        ).lower()
        if svua in ["изотермия", "конвекция", "инверсия"]:
            break
        print("Повторите ввод.")

    A1, A2, B1, B2, C3 = get_coefficients_A1A2B1B2C3(svua)
    C1, C2, D1, D2 = get_coefficients_C1C2D1D2(z0)

    # --- расчёт ------------------------------------------------------- #
    p = ReleaseParams(
        Q=Q1,
        M_mol=M_mol,
        T=T,
        P=P,
        U=U,
        z0=z0,
        A1=A1,
        A2=A2,
        B1=B1,
        B2=B2,
        C3=C3,
        C1=C1,
        C2=C2,
        D1=D1,
        D2=D2,
    )

    x = np.linspace(10, 10_000, 1_001)
    C_line = c_max(x, p)
    D_line = d_max(x, p)

    # NaN-защита
    mask = ~np.isnan(D_line)
    x, C_line, D_line = x[mask], C_line[mask], D_line[mask]

    inv = interp1d(D_line, x, kind="linear", fill_value="extrapolate")
    x_deadly = float(inv(LCt50))
    x_threshold = float(inv(PCt50))

    print(f"Смертельные поражения: {x_deadly:.0f} м")
    print(f"Пороговые поражения: {x_threshold:.0f} м")

    plot_conc_doze_Ox(x, C_line, D_line)

    # --- поля 2‑D ----------------------------------------------------- #
    y = np.linspace(10, 5_000, 1_001)

    dose = d_field(x, y, p)
    dose_full = np.concatenate([dose[:, ::-1], dose], axis=1)
    y_full = np.concatenate([-y[::-1], y])

    dose_mat = dose_full.T
    mask_x = x <= 1_000
    mask_y = (y_full >= -500) & (y_full <= 500)

    idx_x = np.where(mask_x)[0]  # индексы по X
    idx_y = np.where(mask_y)[0]  # индексы по Y

    dose_cut = dose_mat[np.ix_(idx_y, idx_x)]  # (M_cut , N_cut)
    x_cut = x[idx_x]
    y_cut = y_full[idx_y]

    plot_dose_with_contours(
        x=x_cut,
        y_full=y_cut,
        dose_full=dose_cut,
        levels=[PCt50, LCt50],
    )


# --------------------- точка входа ----------------------------------- #

if __name__ == "__main__":
    run_scenario()
