import streamlit as st 

# reg_page = st.Page("./pgs/registration.py", title="register", icon=":material/person_add:")
# signin_page = st.Page("./pgs/signin.py", title="sign in", icon=":material/login:")
# home_page = st.Page("./pgs/main.py", title="home page", icon=":material/home:")
td_page = st.Page("./pgs/quest.py", title="Self service", icon=":material/language:")
# chatbot_page = st.Page("./pgs/chatbot.py", title="chatbot", icon=":material/chat:")



pg = st.navigation([td_page])

st.set_page_config(
    page_title="Huduma AI",
    page_icon="🇰🇪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.echominds.africa',
        'Report a bug': "https://www.echominds.africa",
        'About': "Preserving and promoting Kenyan indigenous languages, culture, and heritage through a tech-driven approach."
    }
)

pg.run()


