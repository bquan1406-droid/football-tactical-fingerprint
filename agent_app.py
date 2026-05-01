import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import joblib
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from difflib import get_close_matches

st.set_page_config(page_title="Football Tactical AI Agent", layout="wide")

st.title("Football Tactical AI Agent")
st.markdown("Ask anything about player archetypes, comparisons, and similar players.")

# Load data and models
@st.cache_resource
def load_data_and_models():
    df = pd.read_csv('data/player_profiles.csv')
    
    with open('models/archetype_names.pkl', 'rb') as f:
        archetype_names = pickle.load(f)
    
    unique_players = df.drop_duplicates(subset=['player']).copy()
    
    return df, unique_players, archetype_names

# Country mapping
country_names = {
    'ENG': 'England', 'EGY': 'Egypt', 'FRA': 'France', 'ESP': 'Spain',
    'GER': 'Germany', 'ITA': 'Italy', 'BEL': 'Belgium', 'NED': 'Netherlands',
    'POR': 'Portugal', 'BRA': 'Brazil', 'ARG': 'Argentina', 'URU': 'Uruguay',
    'SEN': 'Senegal', 'NGA': 'Nigeria', 'CIV': 'Ivory Coast', 'MAR': 'Morocco',
    'KOR': 'South Korea', 'JPN': 'Japan', 'USA': 'United States', 'MEX': 'Mexico',
    'POL': 'Poland', 'CRO': 'Croatia', 'SUI': 'Switzerland', 'SWE': 'Sweden',
    'DEN': 'Denmark', 'NOR': 'Norway', 'IRL': 'Ireland', 'SCO': 'Scotland',
    'WAL': 'Wales', 'NIR': 'Northern Ireland'
}

# Load data
df, unique_players, archetype_names = load_data_and_models()

def get_player_archetype(player_name):
    player_row = unique_players[unique_players['player'] == player_name]
    if player_row.empty:
        return f"Player '{player_name}' not found."
    archetype_id = int(player_row['archetype'].iloc[0])
    return archetype_names[archetype_id]

def get_player_nationality(player_name):
    player_row = unique_players[unique_players['player'] == player_name]
    if player_row.empty:
        return "Unknown"
    code = player_row['nation_'].iloc[0] if 'nation_' in player_row.columns else ""
    return country_names.get(code, code)

def get_player_stats_summary(player_name):
    player_row = unique_players[unique_players['player'] == player_name]
    if player_row.empty:
        return f"Player '{player_name}' not found."
    
    archetype = get_player_archetype(player_name)
    nationality = get_player_nationality(player_name)
    
    goals = player_row['Performance_Gls_norm'].iloc[0] if 'Performance_Gls_norm' in player_row else 0
    tackles = player_row['Performance_TklW_norm'].iloc[0] if 'Performance_TklW_norm' in player_row else 0
    passing = player_row['Total_Cmp%_norm'].iloc[0] if 'Total_Cmp%_norm' in player_row else 0
    creativity = player_row['Creativity_Score_norm'].iloc[0] if 'Creativity_Score_norm' in player_row else 0
    
    def interpret(value):
        if value > 2: return "Elite"
        elif value > 0.5: return "Above Average"
        elif value > -0.5: return "Average"
        else: return "Below Average"
    
    return f"""Player: {player_name}
Nationality: {nationality}
Archetype: {archetype}

Key Stats:
- Goals: {interpret(goals)} ({round(goals, 2)})
- Tackles: {interpret(tackles)} ({round(tackles, 2)})
- Passing: {interpret(passing)} ({round(passing, 2)})
- Creativity: {interpret(creativity)} ({round(creativity, 2)})"""

def find_similar_scouting(player_name, top_n=5):
    player_row = unique_players[unique_players['player'] == player_name]
    if player_row.empty:
        return f"Player '{player_name}' not found."
    
    pc1_target = player_row['PC1'].iloc[0]
    pc2_target = player_row['PC2'].iloc[0]
    goals_target = player_row['Performance_Gls_norm'].iloc[0] if 'Performance_Gls_norm' in player_row else 0
    tackles_target = player_row['Performance_TklW_norm'].iloc[0] if 'Performance_TklW_norm' in player_row else 0
    
    results = []
    for _, row in unique_players.iterrows():
        if row['player'] == player_name:
            continue
        
        pca_dist = np.sqrt((row['PC1'] - pc1_target)**2 + (row['PC2'] - pc2_target)**2)
        pca_sim = max(0, 100 - (pca_dist * 15))
        
        goals_dist = abs(row.get('Performance_Gls_norm', 0) - goals_target)
        tackles_dist = abs(row.get('Performance_TklW_norm', 0) - tackles_target)
        
        goals_sim = max(0, 100 - (goals_dist * 30))
        tackles_sim = max(0, 100 - (tackles_dist * 30))
        
        total_sim = (pca_sim * 0.5 + goals_sim * 0.25 + tackles_sim * 0.25)
        
        results.append({
            'name': row['player'],
            'archetype': archetype_names[int(row['archetype'])],
            'similarity': round(total_sim, 1),
            'team': row.get('team', 'Unknown')
        })
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]

def compare_players(player1_name, player2_name):
    player1 = unique_players[unique_players['player'] == player1_name]
    player2 = unique_players[unique_players['player'] == player2_name]
    
    if player1.empty:
        return f"Player '{player1_name}' not found."
    if player2.empty:
        return f"Player '{player2_name}' not found."
    
    arch1 = archetype_names[int(player1['archetype'].iloc[0])]
    arch2 = archetype_names[int(player2['archetype'].iloc[0])]
    nat1 = get_player_nationality(player1_name)
    nat2 = get_player_nationality(player2_name)
    
    stats = ['Performance_Gls_norm', 'Performance_TklW_norm', 'Total_Cmp%_norm', 'Creativity_Score_norm']
    labels = ['Goals', 'Tackles', 'Passing', 'Creativity']
    
    result = f"{player1_name} ({arch1} - {nat1}) vs {player2_name} ({arch2} - {nat2})\n\n"
    
    for stat, label in zip(stats, labels):
        val1 = player1[stat].iloc[0] if stat in player1.columns else 0
        val2 = player2[stat].iloc[0] if stat in player2.columns else 0
        result += f"{label}: {round(val1, 2)} vs {round(val2, 2)}\n"
    
    return result

# Setup Groq API
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# Define tools for agent
@tool
def get_player_archetype_tool(player_name: str) -> str:
    """Get the tactical archetype of a player (e.g., Complete Forward, Central Midfielder)."""
    return get_player_archetype(player_name)

@tool
def get_player_stats_summary_tool(player_name: str) -> str:
    """Get a summary of a player's key stats including goals, tackles, passing, and creativity."""
    return get_player_stats_summary(player_name)

@tool
def compare_players_tool(player1: str, player2: str) -> str:
    """Compare two players side by side including archetypes and key stats."""
    return compare_players(player1, player2)

@tool
def find_similar_players_tool(player_name: str, top_n: int = 5) -> str:
    """Find players who have a similar playing style to a given player."""
    results = find_similar_scouting(player_name, top_n)
    if isinstance(results, list):
        output = f"Players similar to {player_name}:\n"
        for i, p in enumerate(results, 1):
            output += f"{i}. {p['name']} ({p['archetype']}) - {p['similarity']}% similar\n"
        return output
    return str(results)

tools = [
    get_player_archetype_tool,
    get_player_stats_summary_tool,
    compare_players_tool,
    find_similar_players_tool,
]

# Create agent
agent = create_react_agent(model=llm, tools=tools)

# Chat interface
st.subheader("Chat with the Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a player..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.invoke({"messages": [("user", prompt)]})
            answer = response["messages"][-1].content
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
