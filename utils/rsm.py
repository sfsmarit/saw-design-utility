from dataclasses import dataclass, field
import re
import numpy as np
import plotly.graph_objects as go
import pandas as pd


@dataclass
class Track:
    name: str = ""
    x: list[float] = field(default_factory=list)
    l: list[float] = field(default_factory=list)
    df: list[float] = field(default_factory=list)
    sign: list[int] = field(default_factory=list)
    pol: list[str] = field(default_factory=list)

    def generate_graph_points(self):
        gx, gl, gdf = [], [], []
        for x_, l_, df_ in zip(self.x, self.l, self.df):
            gx += [x_-l_/4, x_+l_/4]
            gl += [l_]*2
            gdf += [df_]*2
        return gx, gl, gdf

    def set_left_polarity(self, left: str):
        pols = ["HOT", "GND"]
        self.pol = [left]
        for sign in self.sign[1:]:
            if sign == self.sign[0]:
                self.pol.append(left)
            else:
                self.pol.append(pols[not pols.index(left)])

    def calc_grad_ratio(self):
        return 1 - min(self.l) / max(self.l)


def split(text, newline="\r\n"):
    lines = text.split(newline)
    config_idx = lines.index("*Configuration")
    headers = lines[:config_idx+2]

    profile = []
    for line in lines[config_idx+2:]:
        x = re.split("[ \t,]", line)
        idt = [v for v in x if v]
        if not idt:
            continue
        if idt[0] not in ["S", "R"]:
            idt.insert(0, "")
        if len(idt) == 4:
            idt.append("")
        profile.append(idt)

    return headers, profile


def reconstruct_rsm(headers, profile):
    result = "\n".join(headers) + "\n"

    for idt in profile:
        line = ",\t".join(idt) + "\n"
        # 不要な先頭のカンマを削除
        if line[0] == ",":
            line = line[1:]
        # 先頭にタブ追加
        line = "\t" + line
        result += line

    return result


def expand(profile):
    """定ピッチ部を0.5本単位に展開"""
    new_profile = []

    for idt in profile:
        # 符号
        first_sign = idt[-1][0] if idt[-1] else ""

        if first_sign == "+":
            opp_sign = "-"
        elif first_sign == "-":
            opp_sign = "+"
        else:
            opp_sign = ""

        # 展開
        n = int(float(idt[1])*2)
        for i in range(n):
            d = idt.copy()
            if i:
                d[0] = ""
            d[1] = '0.5'
            d[-1] = first_sign if i % 2 == 0 else opp_sign
            d[-1] += "\\"
            new_profile.append(d)

    # ブロック末の\\を削除
    n = len(new_profile)
    for i in range(n-1):
        idt1 = new_profile[i]
        idt2 = new_profile[i+1]
        if idt2[0]:
            idt1[-1] = idt1[-1][:-1]
    new_profile[-1] = new_profile[-1][:-1]

    return new_profile


def convert_MM(profile):
    """MM変換"""
    # L: Lambda
    # DF: duty factor
    # P: L/2
    # HalfS: Half IDT spacing
    L, DF = [], []
    for i, d in enumerate(profile):
        L.append(float(d[2][:-1]))
        DF.append(float(d[3]))

    L = np.array(L)
    DF = np.array(DF)
    P = L / 2
    HalfS = P * (1 - DF) / 2

    # S23(S12): IDTの右(左)側のスペース
    # P23(P12): IDTの右(左)隣りの平均ピッチ
    n = len(HalfS)

    S23, P23 = np.zeros(n), np.zeros(n)
    for i in range(n - 1):
        S23[i] = HalfS[i] + HalfS[i+1]
        P23[i] = (P[i] + P[i+1]) / 2

    S12 = np.zeros(n)
    P12 = np.zeros(n)

    # 左端のIDTは左右対称とみなす
    S12[0] = S23[0]
    P12[0] = P23[0]

    for i in range(1, n):
        S12[i] = S23[i-1]
        P12[i] = P23[i-1]

    # 右端のIDTも左右対称とみなす
    S23[-1] = S12[-1]
    P23[-1] = P12[-1]

    # MMモデルにおけるIDT寸法
    # w23_MM (w12_MM) : 右(左)半分のIDT幅
    # DF_MM : duty factor
    # halfS_MM : IDTスペースの半値
    # P_MM : Lambda / 2
    # L_MM : Lambda
    w23_MM = (P23 - S23) / 2
    w12_MM = (P12 - S12) / 2
    w_MM = w23_MM + w12_MM  # 全幅
    DF_MM = np.zeros(n)
    halfS_MM = np.zeros(n)
    P_MM = np.zeros(n)

    # 1本目のDFはGGモデルと一致させる
    DF_MM[0] = DF[0]
    P_MM[0] = w_MM[0] / DF_MM[0]
    halfS_MM[0] = P_MM[0] * (1-DF_MM[0]) / 2

    # 2本目以降の計算では、スペース半値を決定してからピッチとDFを求める
    for i in range(1, n):
        halfS_MM[i] = S12[i] - halfS_MM[i-1]
        P_MM[i] = w_MM[i] + 2 * halfS_MM[i]
        DF_MM[i] = w_MM[i] / P_MM[i]

    L_MM = 2 * P_MM

    new_profile = profile.copy()
    for idt, l, df in zip(new_profile, L_MM, DF_MM):
        idt[2] = f"{l:.5f}u"
        idt[3] = f"{df:.4f}"

    return new_profile


def divide_into_tracks(profile):
    tracks: list[Track] = []
    signs = {
        "": 0,
        "+": 1,
        "-": -1,
    }

    x_prev = l_prev = 0
    for idt in profile:
        l = float(idt[2][:-1])
        df = float(idt[3])
        x = x_prev + l_prev/4 + l/4
        if len(idt) == 4:
            s = ""
        else:
            s = idt[-1].replace("\\", "")
        sign = signs[s]

        if idt[0]:
            tracks.append(Track())

        tracks[-1].x.append(x)
        tracks[-1].l.append(l)
        tracks[-1].df.append(df)
        tracks[-1].sign.append(sign)

        x_prev = x
        l_prev = l

    for i, tr in enumerate(tracks):
        if i == 0:
            tr.name = "L"
        elif i == len(tracks) - 1:
            tr.name = "R"
        else:
            tr.name = f"T{i}"

    return tracks


def plot_pitch_profile(tracks: list[Track]):
    fig = go.Figure()

    # DF (right axis)
    x, df = [], []
    for tr in tracks:
        gx, gl, gdf = tr.generate_graph_points()
        x += gx
        df += gdf

    fig.add_trace(
        go.Scatter(x=x, y=df, mode="lines", name=f"DF", yaxis="y2", line=dict(width=1, color="lightgray"))  # ← y2 に描く
    )

    df_range = [
        max(0, round(10*min(df)) / 10 - 0.2),
        min(1, round(10*max(df)) / 10 + 0.2),
    ]

    # Pitch (left axis)
    for i, tr in enumerate(tracks):
        gx, gl, gdf = tr.generate_graph_points()
        fig.add_trace(
            go.Scatter(x=gx, y=gl, mode="lines", name=tr.name, yaxis="y")
        )

    # Plot settings
    fig.update_layout(
        title="Pitch Profile",
        xaxis_title="X [um]",
        # template="plotly_white",
        yaxis=dict(side='left', title="Pitch [um]", dtick=0.2),
        yaxis2=dict(side='right', title="DF", dtick=0.1, range=df_range, showgrid=False, overlaying='y'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
        ),
        showlegend=False,
    )

    return fig


def generate_polarity_data(tracks: list[Track], left: str):
    if len(tracks) < 2:
        return pd.DataFrame()

    tracks[1].set_left_polarity(left)  # type: ignore
    right = tracks[1].pol[-1]
    for tr in tracks[2:-1]:
        tr.set_left_polarity(right)
        right = tr.pol[-1]

    tracks[0].set_left_polarity("GND")
    tracks[-1].set_left_polarity("GND")

    pol_table = {
        f"{tracks[i].name} - {tracks[i+1].name}":
            f"{tracks[i].pol[-1][0]} - {tracks[i+1].pol[0][0]}"
            for i in range(len(tracks)-1)
    }

    return pd.DataFrame([pol_table])


if __name__ == "__main__":
    newline = "\n"
    file = r"C:\Users\marit\Documents\project\DG036N\final_sim\rsm\=BTM__DG036N_standalone@Filter@B71_D1=B71_D1.rsm"
    with open(file, encoding="utf-8") as f:
        text = f.read()

    headers, profile = split(text, newline=newline)
    exp_profile = expand(profile)
    tracks = divide_into_tracks(exp_profile)

    pol0 = "HOT"
    tracks[0].set_left_polarity(pol0)
    last_pol = tracks[0].pol[-1]
    for tr in tracks[1:]:
        tr.set_left_polarity(last_pol)
        last_pol = tr.pol[-1]

    for tr in tracks:
        print(tr.pol[0], tr.pol[-1])
