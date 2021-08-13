import streamlit as st  ## Used to build the web app
import pandas as pd     ## Used to handle the dataframe and web scraping
import base64  ## Used to handle the data download for the CSV file to encode the ASCII to Byte conversion.
import matplotlib.pyplot as plt  ## Used to create the heatmap plot
import seaborn as sns   ## Used to create the heatmap plot
import numpy as np    ## Used to create the heatmap plot
from PIL import Image

image = Image.open('nba_player.png')

st.image(image, use_column_width=True)

st.title('NBA Player Statistics Explorer')

st.markdown("""
    This app performs simple webscraping and data visualization of NBA player stats data.
    * **Python libraries used includes:** base64, pandas, streamlit, matplotlib, seaborn and numpy.
    * **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Select Year', list(reversed(range(1950,2021))))

## Webscraping of the NBA player stats here
@st.cache
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header = 0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)  # This line deletes repeated Age elements
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis = 1)
    return playerstats
playerstats = load_data(selected_year)
playerstats['FG%'] = playerstats['FG%'].astype(str)
playerstats['3P%'] = playerstats['3P%'].astype(str)
playerstats['2P%'] = playerstats['2P%'].astype(str)
playerstats['eFG%'] = playerstats['eFG%'].astype(str)
playerstats['FT%'] = playerstats['FT%'].astype(str)
playerstats['2P%'] = playerstats['2P%'].astype(str)

## Sidebar Team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Select Team', sorted_unique_team, sorted_unique_team)

## Sidebar - Position Selection
unique_position = ['C', 'PF', 'SF', 'PG', 'SG']
selected_position = st.sidebar.multiselect('Team Position', unique_position, unique_position)

## filtering the data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_position))]

st.header('Display Player Statistics of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

# To download the players stats data
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() # strings to bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href


st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# Heatmap
st.write("""
     Click the button below to view the Intercorrelation Matrix Heatmap
""")

if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()


    
st.subheader('Completed by Adubi Olubunmi')
st.set_option('deprecation.showPyplotGlobalUse', False)