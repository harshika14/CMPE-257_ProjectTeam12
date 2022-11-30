import streamlit as st



st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
st.write("## Welcome to our Review and Tagging Prediction app based on Yelp Dataset 👋")
st.markdown(
    """
    **👈 Select a demo from the sidebar** to see our different types of models (Review and Tagging)
    of what our models can do!
    ### Want to peek at our codebase?
    - Check out our team [Git repo](https://github.com/harshika14/CMPE-257_ProjectTeam12)
    - Dataset has been obtained from [Yelp](https://www.yelp.com/dataset)


    ### Want to look at our other Repos?
    - [Raju](https://github.com/rajuptvs)
    - [Sanika](https://github.com/sanika-karwa)
    - [Harshika](https://github.com/harshika14)
    - [Swapna](https://github.com/kothaswapna)


"""
)
st.sidebar.success("Select a demo above.")
# Add all your application here
