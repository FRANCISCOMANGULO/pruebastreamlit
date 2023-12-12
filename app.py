import streamlit as st
import pandas as pd
import folium
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from gtts import gTTS
import os


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.sidebar.header("Configuration")

ruta_archivo = r'C:\Users\baby_\OneDrive\Escritorio\streamlitproyect\restaurante_google.csv'
df_restaurantes = pd.read_csv("nombres_restaurantes_latitud_longitud.csv")

agent = create_csv_agent(
    OpenAI(temperature=1),
    "restaurante_google_index.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

st.image("datasolu.jpg", width=200)
st.title('Recomendador de Restaurantes en California')


st.subheader('Mapa de California')
coord_california = [36.7783, -119.4179]
m = folium.Map(location=coord_california, zoom_start=6)
st.write(m)


query = st.text_input("Escribe aquí tu pregunta")

if st.button('Buscar'):
    with st.spinner('Procesando...'):
        
        results = agent.run(query)
        tts = gTTS(text=results, lang='es')
        tts.save("results.mp3")

        
        os.system("start results.mp3")
        
        st.subheader('Resultados:')
        st.write(results)
        
        
        if isinstance(results, str):
            
            names = results.split(', ')
            
            
            matching_restaurants = df_restaurantes[df_restaurantes['name'].isin(names)]
            
            if not matching_restaurants.empty:
                
                st.subheader('Mapa de Restaurantes')
                m = folium.Map(location=[matching_restaurants['latitude'].mean(), matching_restaurants['longitude'].mean()], zoom_start=12)
                
                
                for index, row in matching_restaurants.iterrows():
                    folium.Marker([row['latitude'], row['longitude']], popup=row['name']).add_to(m)
                
                
                st.write(m)
            else:
                st.write("No se encontraron restaurantes para esa consulta.")
        else:
            st.write("La consulta no devolvió una cadena de resultados esperada.")

