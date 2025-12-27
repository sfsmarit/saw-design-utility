import skrf as rf
import numpy as np
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def dB(x):
    return 20 * np.log10(np.abs(x) + 1e-300)


def generate_freq_table_text(nw: rf.Network) -> str:
    segments = [[]]
    for i, f in enumerate(nw.f):
        last_segm = segments[-1]
        if i < 2 or f - last_segm[-1] == last_segm[-1] - last_segm[-2]:
            last_segm.append(f)
        else:
            segments.append([last_segm[-1], f])

    text = ""
    for segm in segments:
        sss = [segm[0],
               segm[-1],
               segm[1] - segm[0] if len(segm) > 1 else 0]
        sss = [float(v*1e-6) for v in sss]
        sss = [int(v) if v.is_integer() else v for v in sss]
        text += "\tsweep {" + f"start={sss[0]}MEG, stop={sss[1]}MEG, step={sss[2]}MEG" + "}\n"

    return text


def plot_port_char(
    nw: rf.Network,
    p: int,
    fmin: float,
    fmax: float,
    log_scale: list[bool],
):
    f_MHz = nw.f * 1e-6

    z0 = np.abs(nw.z0[0][0])
    y11 = nw.y[:, p-1, p-1]
    s11 = nw.s[:, p-1, p-1]
    z = z0 * (1 + s11) / (1 - s11)

    Ls = np.maximum(0, np.imag(z) / (2 * np.pi * nw.f))
    Cs = np.maximum(0, -1 / np.imag(z) / (2 * np.pi * nw.f))
    Rs = np.real(z) - z0

    fig = make_subplots(
        rows=1,
        cols=5,
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}, {"secondary_y": False}]],
        subplot_titles=(
            "Y [dB]",
            "Re[Y]",
            "L [nH]",
            "C [fF]",
            "R [Ω]"
        )
    )

    # --- col=1: Y11 [dB] ---
    fig.add_trace(
        go.Scatter(x=f_MHz, y=dB(y11), mode="lines", name="Y11 [dB]"),
        row=1, col=1
    )

    # --- col=2: Re[Y11]
    fig.add_trace(
        go.Scatter(x=f_MHz, y=np.real(y11), mode="lines", name="Re[Y11]"),
        row=1, col=2
    )

    # --- col=3: 空欄 ---
    fig.add_trace(
        go.Scatter(x=f_MHz, y=Ls * 1e9, mode="lines", name="L [nH]"),
        row=1, col=3
    )

    # --- col=4: Ls / Cs） ---
    fig.add_trace(
        go.Scatter(x=f_MHz, y=Cs * 1e12, mode="lines", name="C [fF]"),
        row=1, col=4
    )

    # --- col=5: Rs [Ω] ---
    fig.add_trace(
        go.Scatter(x=f_MHz, y=Rs, mode="lines", name="R [Ω]"),
        row=1, col=5
    )

    # ==== 軸レンジ・ラベル・グリッド ====
    for c in range(1, 6):
        fig.update_xaxes(title_text="[MHz]", range=[fmin, fmax], row=1, col=c, showgrid=True)
        fig.update_yaxes(showgrid=True, row=1, col=c)
        if log_scale[c-1]:
            fig.update_yaxes(type="log", row=1, col=c)

    # ==== レイアウト ====
    fig.update_layout(
        title=f"Port {p}",
        # template="plotly_white",
        width=1500, height=300,
        # legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
