import pandas as pd
import streamlit as st
from utils import rsm


st.markdown(
    """
    ---
    ### Pitch Profile
    ---    
    """
)


if True:
    newline = "\r\n"
    file = st.file_uploader("rsm File", type="rsm")
    if not file:
        st.stop()
    text = file.getvalue().decode("utf-8", errors="replace")
else:
    newline = "\n"
    file = r"C:\Users\marit\Documents\project\DG036N\final_sim\rsm\=BTM__DG036N_standalone@Filter@B71_D1=B71_D1.rsm"
    with open(file, encoding="utf-8") as f:
        text = f.read()

cols = st.columns(2)
with cols[0]:
    mm_conv = st.toggle("MM Conversion", value=True)
with cols[1]:
    rsm_download = st.empty()


headers, profile = rsm.split(text, newline=newline)
exp_profile = rsm.expand(profile)
if mm_conv:
    exp_profile = rsm.convert_MM(exp_profile)
    mm_rsm = rsm.reconstruct_rsm(headers, exp_profile)
    rsm_download.download_button(
        ":material/download: MM rsm",
        data=mm_rsm,
        file_name=f"MM_test.rsm",
        mime="text/plain",
    )

tracks = rsm.divide_into_tracks(exp_profile)

# Polarity
left = st.segmented_control("First IDT Polarity", ["HOT", "GND"], default="HOT")
df = rsm.generate_polarity_data(tracks, left)  # type: ignore
st.markdown("**Polarity**")
st.dataframe(df)


# Pitch profile
fig = rsm.plot_pitch_profile(tracks)
st.plotly_chart(fig)


# Gradation ratio
names = [tr.name for tr in tracks]
grads = [tr.calc_grad_ratio() for tr in tracks]
max_grad = max(grads)
max_track_name = names[grads.index(max_grad)]
data = {name: f"{gr:.1%}" for name, gr in zip(names, grads)}
df = pd.DataFrame([data])
st.markdown("**Pitch modulation**")
st.caption("modulation = 1 - min(L)/max(L)")
st.dataframe(df)
st.markdown(f"Max = {max_grad:.1%} ({max_track_name})")
