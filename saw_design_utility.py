import streamlit as st


st.set_page_config("SAW Design Utility", page_icon=":toolbox:", layout="wide")
st.title("SAW Design Utility")


pages = [
    st.Page("contents/pitch_profile.py", title="Pitch Profile", icon=":material/home:"),
    st.Page("contents/cornerlotsim.py", title="Corner Lot Sim", icon=":material/equalizer:"),
    st.Page("contents/develop.py", title="Develop", icon=":material/construction:"),
]
st.navigation(pages).run()
