import streamlit as st
import skrf as rf
from utils import sp
import tempfile
from pathlib import Path


st.markdown(
    """
    ---
    ### S-parameter Viewer
    ---
    """
)

file = st.file_uploader("Touchstone")
if not file:
    st.stop()

# Load touchstone
suffix = Path(file.name).suffix
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(file.getbuffer())
nw = rf.Network(tmp.name)

# Frequency table
text = sp.generate_freq_table_text(nw)
st.write("**Frequency**")
st.code(text)

# Frequency slidebar
fmin, fmax = int(nw.f[0]*1e-6), int(nw.f[-1]*1e-6)
fstart, fstop = 500, 1000
fstep = 5

f1, f2 = st.slider(
    "**Frequency Range [MHz]**",
    fmin, fmax,
    value=(max(fstart, fmin), min(fstop, fmax)),
    step=fstep,
)

# Characteristics
n_cols = 5
log_scale = [False, True, False, False, False]

cols = st.columns(n_cols)
for i, col in enumerate(cols):
    with col:
        log_scale[i] = st.toggle("Log", value=log_scale[i], key=f"log_{i}")

for p in range(nw.nports):
    fig = sp.plot_port_char(nw, p+1, f1, f2, log_scale)
    st.plotly_chart(fig)
