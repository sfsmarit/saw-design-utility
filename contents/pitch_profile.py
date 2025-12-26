import streamlit as st
from matplotlib import pyplot as plt
from utils import rsm


st.markdown(
    """
    ### Pitch Profile
    """
)


if False:
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

headers, profile = rsm.split(text, newline=newline)
exp_profile = rsm.expand(profile)
mm_profile = rsm.convert_MM(exp_profile)

mm_rsm = rsm.reconstruct_rsm(headers, mm_profile)
st.download_button(
    ":material/download: MM rsm",
    data=mm_rsm,
    file_name=f"MM_test.rsm",
    mime="text/plain",
)

rsm.plot_profile(mm_profile)

# st.write(headers)
# st.write(profile)
# st.write(exp_profile)
# st.write(mm_rsm)
