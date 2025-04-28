import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray


def _make_custom_cmap() -> LinearSegmentedColormap:
    """
    Кастомная палитра «белый → зелёный → жёлтый → оранжевый → красный».
    """
    colors = ["white", "green", "yellow", "orange", "red"]
    return LinearSegmentedColormap.from_list("Wgyor", colors)


def plot_dose_with_contours(
    x: NDArray,
    y_full: NDArray,
    dose_full: NDArray,
    levels: list[float],
) -> None:
    """
    Отрисовывает тепловую карту токсодозы D(x, y) с изолиниями *levels*.
    **Требование наличия y = 0 убрано** — функция работает с любым
    симметричным либо произвольным набором значений `y_full`.

    Параметры
    ----------
    x        : 1-D массив, координата вдоль ветра (м).
    y_full   : 1-D массив, координата поперёк ветра (м).
               Может не содержать 0.
    dose_full: 2-D массив формы (len(y_full), len(x)) — значения доз.
    levels   : список уровней доз (мг·мин/л) для построения изолиний.
    """
    # Создаём сетку
    X, Y = np.meshgrid(x, y_full, indexing="xy")
    cmap = _make_custom_cmap()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Тепловая карта
    pcm = ax.pcolormesh(X, Y, dose_full, shading="auto", cmap=cmap)
    fig.colorbar(pcm, ax=ax, label="Токсодоза, мг·мин/л")

    # Изолинии
    cs = ax.contour(
        X,
        Y,
        dose_full,
        levels=levels,
        colors="black",
        linestyles="--",
        linewidths=1.0,
    )
    ax.clabel(
        cs,
        fmt={lvl: f"{lvl:.1f}" for lvl in levels},
        inline=True,
        fontsize=10,
    )

    # Подписи осей и прочий «косметический» минимум
    ax.set_title("Поле токсодозы")
    ax.set_xlabel("Расстояние вдоль ветра, м")
    ax.set_ylabel("Расстояние поперёк ветра, м")
    ax.set_aspect("equal", adjustable="box")  # квадратные клетки
    fig.tight_layout()
    plt.show()


def plot_conc_doze_Ox(x: NDArray, conc: NDArray, dose: NDArray) -> None:
    """
    Профили максимальной концентрации и токсодозы вдоль оси факела.

    Параметры
    ----------
    x    : расстояние вдоль ветра, м.
    conc : C_max(x), кг/м³.
    dose : D_max(x), мг·мин/л.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"hspace": 0.35}
    )

    # ── График концентрации ───────────────────────────────────────────────
    ax1.semilogy(x, conc, lw=1.2)
    ax1.set_title("Максимальная концентрация на оси облака")
    ax1.set_ylabel("Концентрация, кг/м³")
    ax1.set_xlim(x.min(), x.max())
    ax1.grid(which="both", ls=":", lw=0.5)

    # ── График дозы ───────────────────────────────────────────────────────
    ax2.semilogy(x, dose, lw=1.2)
    ax2.set_title("Токсодоза на оси облака")
    ax2.set_xlabel("Расстояние, м")
    ax2.set_ylabel("Токсодоза, мг·мин/л")

    # диапазон Y, как в методичке (100 … 0.001 мг·мин/л)
    ax2.set_ylim(1e-3, 1e2)
    ax2.set_xlim(x.min(), x.max())

    # базовая линия 1 мг·мин/л (для визуальной привязки)
    ax2.axhline(1, color="k", lw=0.8)

    # мелкая сетка
    ax2.grid(which="both", ls=":", lw=0.5)

    plt.tight_layout()
    plt.show()