import streamlit as st


pages = [
    st.Page("contents/cornerlot.py", title="Corner Lot Sim", icon=":material/equalizer:"),
    st.Page("contents/develop.py", title="Develop", icon=":material/construction:"),
]
st.navigation(pages).run()
