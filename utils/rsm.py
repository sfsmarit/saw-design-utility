import re
import numpy as np
import streamlit as st
from copy import deepcopy
from matplotlib import pyplot as plt


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


def plot_profile(profile):
    new_profile = deepcopy(profile)
    for i, idt in enumerate(new_profile):
        if i:
            idt_prev = new_profile[i-1]
            x_prev = float(idt_prev[-1])
            l_prev = float(idt_prev[2][:-1])
        else:
            x_prev = l_prev = 0

        l = float(idt[2][:-1])
        x = x_prev + l_prev/4 + l/4
        new_profile[i].append(x)

    x_list, l_list, df_list = [], [], []
    for idt in new_profile:
        x_list.append(idt[-1])
        l_list.append(idt[2][:-1])
        df_list.append(idt[3])

    st.write(l_list[:40])
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_list, l_list)
    st.pyplot(fig)
