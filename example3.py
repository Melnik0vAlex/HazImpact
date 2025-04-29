"""Расчёт токсодозы при аварийном выбросе АХОВ — Сценарий 3.
Разрушение оборудования с жидким АХОВ, выброс АХОВ в окружающую
среду, при наличии перегрева у жидкой фазы возможно
ее вскипание с образованием в атмосфере газокапельного облака.
Часть жидкой фазы может пролиться на подстилающую поверхность
— либо в обваловку, либо на неограниченную площадь.
Если жидкость при этом имеет температуру кипения меньше температуры
поверхности, то произойдет вскипание жидкости при
ее соприкосновении с подстилающей поверхностью. Из газовой
фазы, содержавшейся в оборудовании, из образовавшейся при
вскипании за счет перегрева жидкой фазы газокапельной фазы и
из газа, образующегося при кипении пролива, образуется первичное
облако, которое рассеивается в атмосфере и воздействует на
окружающую среду.
Из пролива происходит испарение АХОВ, в результате чего образуется
вторичное облако, которое также рассеивается в атмосфере
и воздействует на окружающую среду.

Представлен пример с аммиаком. Ввод исходным данных и алгоритм расчетов 
не оптимизирован и требует доработки аналогично example1.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import plot_dose_with_contours

# Таблица свойств АХОВ
tbl_7 = {
    "АХОВ": [
        "Аммиак",
        "Мышьяковистый водород",
        "Фтористый водород",
        "Хлористый водород",
        "Бромистый водород",
        "Цианистый водород",
        "Сероводород",
        "Сероуглерод",
        "Формальдегид",
        "Фосген",
        "Фтор",
        "Хлор",
        "Хлорциан",
        "Окись углерода",
        "Окись этилена",
    ],
    "M": [
        17.0,
        77.9,
        20.4,
        36.5,
        80.9,
        27.0,
        34.1,
        76.1,
        30,
        98.9,
        38.0,
        70.1,
        61.5,
        28,
        44,
    ],
    "rg": [
        0.8,
        3.5,
        0.92,
        1.64,
        3.50,
        0.9,
        1.5,
        6.0,
        1.03,
        3.48,
        1.7,
        3.2,
        2.1,
        0.97,
        1.7,
    ],
    "rgid": [
        681,
        1640,
        989,
        1191,
        1490,
        689,
        964,
        1263,
        815,
        1420,
        1512,
        1553,
        1258,
        1000,
        882,
    ],
    "Tkip": [
        -33.4,
        -62.5,
        19.4,
        -85.1,
        -67.8,
        25.6,
        -60.4,
        46.2,
        -19.3,
        8.2,
        -188.0,
        -34.1,
        12.6,
        -191.6,
        10.7,
    ],
    "Cp": [
        4.6,
        0.5,
        1.42,
        0.8,
        0.36,
        1.33,
        1.04,
        0.67,
        1.32,
        0.67,
        3.32,
        0.96,
        0.73,
        1.04,
        1.72,
    ],
    "a": [
        1.34,
        1.3,
        1.3,
        1.41,
        1.42,
        1.31,
        1.3,
        1.24,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.29,
        1.3,
    ],
    "PCt50": [
        15.0,
        0.2,
        4.0,
        2.0,
        2.4,
        0.2,
        1.0,
        30.0,
        0.6,
        0.55,
        0.2,
        0.6,
        0.75,
        10.0,
        2.2,
    ],
    "LCt50": [150, 6, 40, 20, 24, 6, 15, 500, 6, 3.2, 3, 6, 11, 37.5, 25],
    "Hisp": [
        1360,
        242,
        1560,
        300,
        217,
        933,
        310,
        352,
        273,
        158,
        727,
        288,
        208,
        216,
        320,
    ],
}
tbl_7_df = pd.DataFrame(tbl_7)


# Функция для получения значений свойств
def get_OHV_values(aho, parameter):
    try:
        return tbl_7_df.loc[
            tbl_7_df["АХОВ"].str.lower() == aho.lower(), parameter
        ].iloc[0]
    except IndexError:
        print(f"АХОВ '{aho}' не найден в таблице.")
        return None


# Данные для аммиака
aho = "Аммиак"
M = get_OHV_values(aho, "M")  # молярная масса
T_kip = get_OHV_values(aho, "Tkip")  # температура кипения (°C)
Cp = get_OHV_values(aho, "Cp")  # перевод в Дж/(кг·К)
H_kip = get_OHV_values(aho, "Hisp")  # теплота парообразования аммиака,кДж/кг
rho_star = get_OHV_values(aho, "rgid")  # плотность жидкого аммиака при 20°C, кг/м³
PCt50 = get_OHV_values(aho, "PCt50")
LCt50 = get_OHV_values(aho, "LCt50")

# Перевод температуры кипения в Кельвины
T_kip_K = T_kip + 273.15

# Условные входные данные процесса разрушения
alpha = 0.1  # объемная доля газа
V3 = 10.0  # объем оборудования, м³
P3 = 1.5e6  # давление в оборудовании, Па
R = 8.314  # универсальная газовая постоянная
T3 = 20.0  # температура оборудования, °C
T3_K = T3 + 273.15
Q_star = 1500.0  # полная масса жидкости в оборудовании, кг
Tp = 25.0  # температура подстилающей поверхности, °C
lamb_p = 1.6  # теплопроводность подстилающей поверхности, Вт/(м·К)
cp_p = 1000.0  # теплоемкость подстилающей поверхности, Дж/(кг·К)
rho_p = 2500.0  # плотность подстилающей поверхности, кг/м³
U = 2.0  # скорость ветра, м/с
T_vnesh = 20.0  # температура внешней среды, °C
P0 = 101325.0  # атмосферное давление, Па
gamma = 1.4  # показатель адиабаты

z0 = 0.001
h = 0
A1, A2, B1, B2, C3 = 0.098, 0.00135, 0.889, 0.688, 0.08
C1, C2, D1, D2 = 1.56, 0.000625, 0.048, 0.045

# Сохранение всех данных для дальнейшего пошагового расчёта
initial_conditions = {
    "alpha": alpha,
    "mu": M,
    "V3": V3,
    "P3": P3,
    "R": R,
    "T3": T3,
    "Q_star": Q_star,
    "Cp": Cp,
    "T_kip": T_kip,
    "T_kip_K": T_kip_K,
    "H_kip": H_kip,
    "Tp": Tp,
    "lamb_p": lamb_p,
    "cp_p": cp_p,
    "rho_p": rho_p,
    "U": U,
    "T_vnesh": T_vnesh,
    "P0": P0,
    "gamma": gamma,
}


# Формулы расчёта выброса
def Q_total(Q3, Q3_star, Q3_i, Q_g):  # (19)
    return Q3 + Q3_star + Q3_i + Q_g


def Q_gas(alpha, mu, V3, P3, R, T3):  # (20)
    return alpha * mu * V3 * P3 / (R * (T3 + 273.15))


def Q3_evap(Q_star, Cp, T3, T_kip, H_kip):  # (21)
    delta_T = T3 - T_kip
    return Q_star * (1 - np.exp(-Cp * (delta_T + abs(delta_T)) / (2 * H_kip)))


def Q3_star(Q3, Q_star):  # (22)
    return min(Q3, Q_star - Q3)


def F_area(Q_star, Q3, Q3_star, rho_star):  # (24)
    return (Q_star - Q3 - Q3_star) / (0.05 * rho_star)


def p_n_sat(Delta_H_kip, mu, T_kip, T_vnesh, R):  # (26)
    return 760 * np.exp(
        (Delta_H_kip * mu * (1 / (T_kip + 273.15) - 1 / (T_vnesh + 273.15))) / R
    )


def t_kip_calc(Tp, T_kip, H_kip, lamb_p, cp_p, rho_p, F_kont, F, U, mu, p_n):  # (25)
    term1 = (Tp - T_kip + abs(Tp - T_kip)) / H_kip
    term2 = np.sqrt(lamb_p * cp_p * rho_p / np.pi)
    term3 = 1 / (np.sqrt(mu * 1e-6 * (5.83 + 4.1 * U) * p_n))
    term4 = (F_kont / F) * np.sqrt(2 * F / U)
    return min(term1 * term2 * term3 * term4, 1.0)


def Q3_i(
    Tp, T_kip, H_kip, lamb_p, cp_p, rho_p, F_kont, F, t_kip, Q_star, Q3, Q3_star
):  # (23)
    delta_T = Tp - T_kip
    term1 = (delta_T + abs(delta_T)) / H_kip
    term2 = np.sqrt(lamb_p * cp_p * rho_p / np.pi)
    term3 = (F_kont**2 / F) * np.sqrt(t_kip)
    return min(term1 * term2 * term3, Q_star - Q3 - Q3_star)


# Выполним расчёты для аммиака
alpha = initial_conditions["alpha"]
mu = initial_conditions["mu"]
V3 = initial_conditions["V3"]
P3 = initial_conditions["P3"]
R = initial_conditions["R"]
T3 = initial_conditions["T3"]
Q_star = initial_conditions["Q_star"]
Cp = initial_conditions["Cp"]
T_kip = initial_conditions["T_kip"]
H_kip = initial_conditions["H_kip"]
Tp = initial_conditions["Tp"]
lamb_p = initial_conditions["lamb_p"]
cp_p = initial_conditions["cp_p"]
rho_p = initial_conditions["rho_p"]
U = initial_conditions["U"]
T_vnesh = initial_conditions["T_vnesh"]
P0 = initial_conditions["P0"]
gamma = initial_conditions["gamma"]

# Расчёт массы газовой фазы
Q_g = Q_gas(alpha, mu, V3, P3, R, T3)

# Расчёт массы мгновенного испарения жидкости (из-за перегрева)
Q3 = Q3_evap(Q_star, Cp, T3, T_kip, H_kip)

# Остаточная жидкая масса (доиспарение)
Q3s = Q3_star(Q3, Q_star)

# Определение площади пролива
F = F_area(Q_star, Q3, Q3s, rho_star)
F_kont = F  # в случае пролива на неограниченную поверхность

# Расчёт давления насыщенного пара
p_n = p_n_sat(H_kip, mu, T_kip, T_vnesh, R)

# Расчёт времени кипения пролива
t_kip = t_kip_calc(Tp, T_kip, H_kip, lamb_p, cp_p, rho_p, F_kont, F, U, mu, p_n)

# Расчёт массы образования пара с подогревом пролива
Q3i = Q3_i(Tp, T_kip, H_kip, lamb_p, cp_p, rho_p, F_kont, F, t_kip, Q_star, Q3, Q3s)

# Суммарная масса вещества в первичном облаке
Q_total_val = Q_total(Q3, Q3s, Q3i, Q_g)

# Результаты
results_ammonia = {
    "Qг (газовая фаза), кг": Q_g,
    "Q3 (испарение из-за перегрева), кг": Q3,
    "Q3* (доиспарение), кг": Q3s,
    "F (площадь пролива), м²": F,
    "pн (давление насыщенного пара), мм рт.ст.": p_n,
    "tкип (время кипения пролива), с": t_kip,
    "Q3и (массоперенос с поверхности), кг": Q3i,
    "Q_total (суммарная масса первичного облака), кг": Q_total_val,
}


def rho3_vyb(
    Q3, Q3_star, Q3_i, Q_g, rho_kip, mu, P3, R, T3, P0, gamma, Tp, T_kip
):  # (31)
    if Tp > T_kip:
        return rho_kip * Q3 / (Q3 + Q3_i + Q_g)
    else:
        return mu * P3 / (R * (T3 + 273.15)) * (P0 / P3) ** (1 / gamma)


def R3(Q3, rho3_vyb):  # (34)
    return (3 * Q3 / (4 * np.pi * rho3_vyb)) ** (1 / 3)


def rho_kip(mu, R, P0, T_kip):  # (32)
    return mu / R * P0 / (T_kip + 273.15)


# Расчет плотности жидкоти при температуре кипения
rho_kip_val = rho_kip(mu, R, P0, T_kip)

# Расчёт плотности первичного облака
rho3_v = rho3_vyb(Q3, Q3s, Q3i, Q_g, rho_kip_val, mu, P3, R, T3, P0, gamma, Tp, T_kip)

# Расчёт радиуса первичного облака
R3_val = R3(Q3, rho3_v)

# Сохранение результатов
results_cloud = {
    "ρ3выб (плотность облака), кг/м³": rho3_v,
    "R3 (радиус облака), м": R3_val,
    "Масса первичного облака, кг": Q_total_val,
}


# Формулы для испарения с поверхности пролива и образования вторичного облака
def q3_i(F, mu, U, p_n):  # (27)
    # Массовый расход испарения с поверхности пролива
    return F * np.sqrt(mu * 1e-6 * (5.83 + 4.1 * U)) * p_n


def t3(Q, Q3, q3_i):  # (29)
    # Время полного испарения остаточной жидкости
    return (Q - Q3) / q3_i if q3_i > 0 else float("inf")


def rho3_i(mu, P0, R, T_kip):  # (32)
    # Плотность вторичного облака при кипении
    return mu * P0 / (R * (T_kip + 273.15))


def R3_i(F):  # (35)
    # Радиус вторичного облака по площади пролива
    return 0.5 * np.sqrt(F)


# Расчёт массового расхода испарения
q3i_val = q3_i(F, mu, U, p_n)

# Расчёт времени испарения остаточной массы
t3_val = t3(Q_star, Q3, q3i_val)

# Расчёт плотности вторичного облака
rho3_i_val = rho3_i(mu, P0, R, T_kip)

# Расчёт радиуса вторичного облака
R3i_val = R3_i(F)

# Сохраняем и выводим результаты
results_secondary = {
    "q3и (массовый расход испарения), кг/с": q3i_val,
    "t3 (время испарения), с": t3_val,
    "ρ3и (плотность вторичного облака), кг/м³": rho3_i_val,
    "R3и (радиус вторичного облака), м": R3i_val,
}


# Расчет распространения облака
def sigma_x(x):  # (78)
    return C3 * x / np.sqrt(1 + 0.0001 * x)


def sigma_y(x):  # (79)
    if x / U >= 600:
        return sigma_x(x) * (220.2 * 60 + x / U) / (220.2 * 60 + 600)
    else:
        return sigma_x(x)


def g_x(x):  # (81)
    return A1 * x**B1 / (1 + A2 * x**B2)


def f_z0_x(x):  # (82)
    if z0 < 0.1:
        return np.log(C1 * x**D1 * (1 + C2 * x**D2))
    else:
        return np.log(C1 * x**D1 / (1 + C2 * x**D2))


def sigma_z(x):  # (80)
    return f_z0_x(x) * g_x(x)


def G_0(x):  #  (86)
    sz = sigma_z(x)
    return np.exp(-(h**2) / (2 * sz**2))


def G_H(x, y, z):  #  (88)
    sy = sigma_y(x)
    sz = sigma_z(x)
    part_y = np.exp(-(y**2) / (2 * sy**2))
    part_z = np.exp(-((z - h) ** 2) / (2 * sz**2)) + np.exp(
        -((z + h) ** 2) / (2 * sz**2)
    )
    return part_y * part_z


def x_rp(t_star, C3, U):  # (89)
    term1 = (1e-4) * (t_star**2) * U**2
    term2 = np.sqrt(
        8 * np.pi * (C3**2) * (t_star**2) * U**2 + 1e-8 * (t_star**4) * U**4
    )
    denom = 4 * np.pi * C3**2
    return (term1 + term2) / denom


def C_i_max_primary(x, Q1, R1):  # (85)
    sx = sigma_x(x)
    sy = sigma_y(x)
    sz = sigma_z(x)
    denom = (8 / 3) * np.pi * R1**3 + (2 * np.pi)**(3/2) * sx * sy * sz
    return (2 * Q1 / denom) * G_0(x)


def C_i_max_star(x, q_i_star, t_i_star, R_i_star, U):  # (90)
    sy = sigma_y(x)
    sz = sigma_z(x)
    if x <= x_rp(t_i_star, C3, U):
        denom = U * (2 * np.pi * R_i_star**2 + 2 * np.pi * sy * sz)
        return (2 * q_i_star / denom) * G_0(x)
    else:
        denom = (
            2 * np.pi * R_i_star**2 * t_i_star * U
            + (2 * np.pi) ** (3 / 2) * sigma_x(x) * sy * sz
        )
        return (2 * q_i_star * t_i_star / denom) * G_0(x)


def G_3(x, y, z, t):  #  (84)
    sx = sigma_x(x)
    sy = sigma_y(x)
    sz = sigma_z(x)
    part_xy = np.exp(-((x - U * t) ** 2 / (2 * sx**2) + y**2 / (2 * sy**2)))
    part_z = np.exp(-((z - h) ** 2) / (2 * sz**2)) + np.exp(
        -((z + h) ** 2) / (2 * sz**2)
    )
    return part_xy * part_z


def C_i_star(x, y, z, t, q_i_star, t_i_star, R_i_star, U):  # (87)
    xrp_val = x_rp(t_i_star, C3, U)
    sy = sigma_y(x)
    sz = sigma_z(x)
    if x <= xrp_val and t > x / U:
        denom = U * (2 * np.pi * R_i_star**2 + 2 * np.pi * sy * sz)
        return (q_i_star / denom) * G_H(x, y, z)
    elif x > xrp_val and x / U < t <= x / U + t_i_star:
        denom = (
            2 * np.pi * R_i_star**2 * t_i_star * U
            + (2 * np.pi) ** (3 / 2) * sigma_x(x) * sy * sz
        )
        return (q_i_star * t_i_star / denom) * G_3(x, y, z, t)
    else:
        return 0.0


def C_xyzt_primary(x, y, z, t, Q1, R1, U):
    """
    Расчет концентрации вещества в пространственно-временной точке (x,y,z,t)
    для ПЕРВИЧНОГО облака (формула 83).
    """
    sx = sigma_x(x)
    sy = sigma_y(x)
    sz = sigma_z(x)
    
    # знаменатель
    denom = (8 / 3) * np.pi * R1**3 + (2 * np.pi)**(3/2) * sx * sy * sz
    
    # экспоненциальные слагаемые
    sl1 = np.exp(-((x - U * t)**2) / (2 * sx**2) - (y**2) / (2 * sy**2))
    sl2 = np.exp(-((z - h)**2) / (2 * sz**2))
    sl3 = np.exp(-((z + h)**2) / (2 * sz**2))
    
    # итоговая формула
    gz_xyzt = sl1 * (sl2 + sl3)
    
    return (Q1 / denom) * gz_xyzt

def C_xyzt_secondary(x, y, z, t, q_i_star, t_i_star, R_i_star, U):
    """
    Расчет концентрации вещества в пространственно-временной точке (x,y,z,t)
    для ВТОРИЧНОГО облака (по аналогии с первичным, но согласно формулам 87-88-90).
    """
    xrp_val = x_rp(t_i_star, C3, U)
    sx = sigma_x(x)
    sy = sigma_y(x)
    sz = sigma_z(x)

    if x <= xrp_val and t > x / U:
        denom = U * (2 * np.pi * R_i_star**2 + 2 * np.pi * sy * sz)
        sl1 = np.exp(-((x - U * t)**2) / (2 * sx**2) - (y**2) / (2 * sy**2))
        sl2 = np.exp(-((z - h)**2) / (2 * sz**2))
        sl3 = np.exp(-((z + h)**2) / (2 * sz**2))
        gz = sl1 * (sl2 + sl3)
        return (q_i_star / denom) * gz
    
    elif x > xrp_val and x / U < t <= x / U + t_i_star:
        denom = 2 * np.pi * R_i_star**2 * t_i_star * U + (2 * np.pi)**(3/2) * sx * sy * sz
        sl1 = np.exp(-((x - U * t)**2) / (2 * sx**2) - (y**2) / (2 * sy**2))
        sl2 = np.exp(-((z - h)**2) / (2 * sz**2))
        sl3 = np.exp(-((z + h)**2) / (2 * sz**2))
        gz = sl1 * (sl2 + sl3)
        return (q_i_star * t_i_star / denom) * gz
    
    else:
        return 0.0
    
# Расчет токсодоз для первичного облака
def Gn_xyz(x, y, z, h):
    e1 = np.exp(-y**2 / 2 / sigma_y(x)**2)
    e2 = np.exp(-pow(z - h, 2) / 2 / sigma_z(x)**2)
    e3 = np.exp(-pow(z + h, 2) / 2 / sigma_z(x)**2)
    
    return e1 * (e2 + e3)

def D_max(x, Q1, R1):
    
    chisl = 2 * Q1 * (2*np.pi)**0.5 * sigma_x(x)
    znam = U * (8/3 * np.pi * R1**3 + (2*np.pi)**(3/2)*sigma_x(x)*sigma_y(x)*sigma_z(x))
    g0_x = np.exp(-h**2 / 2 / sigma_z(x)**2)

    return chisl / znam * g0_x * 1000 / 60

def D_xyzt(x, y, z, h, Q1, R1):
    
    chisl = Q1 * (2*np.pi)**0.5 * sigma_x(x)
    znam = U * (8/3 * np.pi * R1**3 + (2*np.pi)**(3/2)*sigma_x(x)*sigma_y(x)*sigma_z(x))

    return chisl / znam * Gn_xyz(x, y, z, h) * 1000 / 60


# Расчет токсодозы для вторичного облака (формулы 105–106)
def D_max_star(x, q_i_star, t_i_star, R_i_star, U, t_expo):
    """
    Максимальная токсодоза D*_i_max(x, 0, 0) для вторичного облака (формула 106)
    """
    sx = sigma_x(x)
    sy = sigma_y(x)
    sz = sigma_z(x)

    t_eff = min(t_i_star, t_expo)
    xrp_val = x_rp(t_i_star, C3, U)

    if x <= xrp_val:
        denom = U * (2 * np.pi * R_i_star**2 + 2 * np.pi * sy * sz)
        return (2 * q_i_star * t_eff / denom) * G_0(x) * 1000 / 60
    else:
        denom = U * (2 * np.pi * R_i_star**2 * t_i_star + (2 * np.pi)**(3/2) * sx * sy * sz)
        return (2 * q_i_star * (2 * np.pi)**0.5 * sx * t_eff / denom) * G_0(x) * 1000 / 60

def D_xyz_star(x, y, z, t_i_star, q_i_star, R_i_star, U, t_expo):
    """
    Поле токсодозы D*_i(x,y,z) для вторичного облака (формула 105)
    """
    sx = sigma_x(x)
    sy = sigma_y(x)
    sz = sigma_z(x)

    t_eff = min(t_i_star, t_expo)
    xrp_val = x_rp(t_i_star, C3, U)

    G_h_val = G_H(x, y, z)

    if x <= xrp_val:
        denom = U * (2 * np.pi * R_i_star**2 + 2 * np.pi * sy * sz)
        return (q_i_star * t_eff / denom) * G_h_val * 1000 / 60
    else:
        denom = U * (2 * np.pi * R_i_star**2 * t_i_star + (2 * np.pi)**(3/2) * sx * sy * sz)
        return (q_i_star * (2 * np.pi)**0.5 * sx * t_eff / denom) * G_h_val * 1000 / 60


# Расчёт значений
x_values = np.linspace(1, 1000, 200)

c_max_secondary = [C_i_max_star(x, q3i_val, t3_val, R3i_val, U) for x in x_values]
c_max_primary = [C_i_max_primary(x, Q_total_val, R3_val) for x in x_values]

plt.figure(figsize=(10, 6))
plt.plot(x_values, c_max_secondary, label='Вторичное облако', color='orange')
plt.plot(x_values, c_max_primary, label='Первичное облако', color='blue')
plt.xlabel("x (м)")
plt.ylabel("Максимальная концентрация $c_{i,max}(x)$, кг/м³")
plt.title("Сравнение концентраций первичного и вторичного облака аммиака\nпо оси y=0, z=0")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Расчёт поля токсодозы вдоль осей X и Y (Z = 0)
x = np.linspace(10, 1000, 500)
y = np.linspace(0, 500, 500)
t_expo = 10 * 60  # экспозиция

dose_primary = np.array([[D_xyzt(xi, yi, 0, h, Q_total_val, R3_val) for yi in y] for xi in x])
dose_secondary = np.array([[D_xyz_star(xi, yi, 0, t3_val, q3i_val, R3i_val, U, t_expo) for yi in y] for xi in x])

# Создание симметрии по оси Y
dose_primary_full = np.concatenate([dose_primary[:, ::-1], dose_primary], axis=1)
dose_secondary_full = np.concatenate([dose_secondary[:, ::-1], dose_secondary], axis=1)
y_full = np.concatenate([-y[::-1], y])

# Визуализация
levels = [PCt50, LCt50]

plot_dose_with_contours(
    x=x,
    y_full=y_full,
    dose_full=dose_primary_full.T,
    levels=levels,
    title="Поле токсодозы от первичного облака"
)

plot_dose_with_contours(
    x=x,
    y_full=y_full,
    dose_full=dose_secondary_full.T,
    levels=levels,
    title="Поле токсодозы от вторичного облака"
)