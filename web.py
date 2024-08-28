import streamlit as st
from streamlit_option_menu import option_menu


import fdopa, mri, multi
st.set_page_config(
        page_title="Parkinson Disease Detection",
)

class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='Models',
                options=['MRI','FDOPA','MULTI'],
                default_index=0,
                styles={
                    "container": {"padding": "5!important","background-color":'black'},
        "icon": {"color": "white", "font-size": "15px"}, 
        "nav-link": {"color":"white","font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
        "nav-link-selected": {"background-color": "#02ab21"},}
                )

        
        if app == "MRI":
            mri.main()
        if app == "FDOPA":
            fdopa.main()    
        if app == "MULTI":
            multi.main() 
             
    run()