import streamlit as st
import re
from matplotlib import pyplot as plt
from utils import CornerLot


st.set_page_config("Corner Lot Sim", layout="wide")
st.title("Corner Lot Simulation")


result_file = st.file_uploader("Monte Calro Result", type="csv")
spec_file = st.file_uploader("FI Spec", type="csv")

# 周波数スライダー
fmin, fmax = st.slider("Frequency range [MHz]", 0, 10000, value=(0, 10000), step=100)
fmin *= 1e6
fmax *= 1e6

# 除外ワード入力
text = st.text_input("Spec words to exlude", "IR, OPB, SHB, RIPPLE")
spec_words_to_exclude = re.split(r"[,\s]+", text)

# Corner Lot Sim アプリで出力されたファイルかどうかの判定
result_filename = result_file.name if result_file else ""
is_mont = "mont" in result_filename

# Plot button
if st.button("Plot Failure Rate", width="stretch"):
    if result_file and spec_file:
        cornerlot = CornerLot()
        with st.spinner("Calculating..."):
            cornerlot.load(result_file, spec_file, is_mont=is_mont)
            cornerlot.filter(fmin, fmax, spec_words_to_exclude)
            fig, axes = cornerlot.get_failure_rate_figure()
            st.pyplot(plt)  # type: ignore
    else:
        if not result_file:
            st.warning("Please upload Monte Calro Result file.")
        if not spec_file:
            st.warning("Please upload FI Spec file.")
