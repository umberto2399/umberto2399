import streamlit as st
import pandas as pd
from openai import OpenAI

# Load the player stats data
df = pd.read_csv('player_stats.csv')

client = OpenAI(api_key='YOUR API KEY HERE')

# Ensure the DataFrame has a 'playing_style' column
if 'playing_style' not in df.columns:
    df['playing_style'] = ''

# Function to get the roster of a team for a specific season
def get_team_roster(data, team_name, season_year):
    if team_name == "All Teams":
        return data[data['SEASON'] == season_year]
    return data[(data['TEAM'] == team_name) & (data['SEASON'] == season_year)]

# Function to save the DataFrame
def save_dataframe(data):
    data.to_csv('player_stats.csv', index=False)

# Function to call ChatGPT API and get insights
def generate_insights(players_give_away, players_acquire):
    prompt = f"Analyze the following potential trade:\n\nPlayers to Give Away:\n{players_give_away}\n\nPlayers to Acquire:\n{players_acquire}\n\nProvide the pros and cons of this trade."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a basketball player specialist who helps managers to take decisions about players."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# Streamlit app
st.title("NBA Player Stats")

# Display the entire table at the beginning
st.write("### All Player Stats")
st.dataframe(df)

# Filter options
st.write("### Filter Options")
filter_option = st.selectbox("Filter by", ["None", "Player", "Team"])

if filter_option == "Player":
    player_name = st.selectbox("Select Player", ["All Players"] + df['PLAYER'].unique().tolist())
    if player_name == "All Players":
        filtered_stats = df
    else:
        filtered_stats = df[df['PLAYER'] == player_name]
    st.dataframe(filtered_stats)

elif filter_option == "Team":
    team_name = st.selectbox("Select Team", ["All Teams"] + df['TEAM'].unique().tolist())
    season_year = st.selectbox("Select Season", df['SEASON'].unique())
    roster = get_team_roster(df, team_name, season_year)
    st.dataframe(roster)

# Section to add playing style for each player
st.write("### Add Playing Style")
player_name = st.text_input("Player Name")
playing_style = st.text_area("Playing Style")

if st.button("Save Playing Style"):
    if player_name in df['PLAYER'].values:
        df.loc[df['PLAYER'] == player_name, 'playing_style'] = playing_style
        save_dataframe(df)
        st.success(f"Playing style for {player_name} saved successfully!")
    else:
        st.error(f"Player {player_name} not found in the dataset.")

# Display the updated DataFrame
st.write("### Updated Player Stats with Playing Style")
st.dataframe(df)

# Section for analyzing potential trades using ChatGPT
st.write("### Analyze Potential Trade")

st.write("#### Select Players to Give Away")
give_away_players = st.multiselect("Players to Give Away", df['PLAYER'].unique().tolist())

st.write("#### Select Players to Acquire")
acquire_players = st.multiselect("Players to Acquire", df['PLAYER'].unique().tolist())

if st.button("Generate Trade Insights"):
    if give_away_players and acquire_players:
        give_away_data = df[df['PLAYER'].isin(give_away_players)].to_dict(orient='records')
        acquire_data = df[df['PLAYER'].isin(acquire_players)].to_dict(orient='records')
        give_away_data_str = "\n".join([str(player) for player in give_away_data])
        acquire_data_str = "\n".join([str(player) for player in acquire_data])
        insights = generate_insights(give_away_data_str, acquire_data_str)
        st.write("### Trade Insights")
        st.write(insights)
    else:
        st.error("Please select at least one player to give away and one player to acquire.")
