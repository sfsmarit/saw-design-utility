import streamlit as st


st.set_page_config("SAW Design Utility", page_icon=":toolbox:", layout="wide")
st.title("SAW Design Utility")


pages = [
    st.Page("contents/sp_viewer.py", title="Sp Viewer"),
    st.Page("contents/pitch_profile.py", title="Pitch Profile"),
    st.Page("contents/FI_spec.py", title="FI Spec"),
    st.Page("contents/cornerlotsim.py", title="Corner Lot Sim"),
]
st.navigation(pages).run()
