import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from difflib import get_close_matches
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

st.set_page_config(page_title="True AI Agent - Football Tactical Analyst", layout="wide")

st.title("True AI Agent")
st.markdown("Ask anything naturally. The AI decides what to do.")

@st.cache_resource
def load_data():
    df = pd.read_csv('data/player_profiles_full.csv')
    with open('models/archetype_names.pkl', 'rb') as f:
        archetype_names = pickle.load(f)
    unique_players = df.drop_duplicates(subset=['player']).copy()
    return df, unique_players, archetype_names

@st.cache_resource
def load_llm():
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)

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

df, unique_players, archetype_names = load_data()
llm = load_llm()

def get_player_stats_summary(player_name):
    player_row = unique_players[unique_players['player'] == player_name]
    if player_row.empty:
        return None
    
    archetype_id = int(player_row['archetype'].iloc[0])
    archetype = archetype_names[archetype_id]
    
    code = player_row['nation_'].iloc[0] if 'nation_' in player_row.columns else ""
    nationality = country_names.get(code, code)
    
    goals = player_row['Performance_Gls_norm'].iloc[0] if 'Performance_Gls_norm' in player_row else 0
    tackles = player_row['Performance_TklW_norm'].iloc[0] if 'Performance_TklW_norm' in player_row else 0
    passing = player_row['Total_Cmp%_norm'].iloc[0] if 'Total_Cmp%_norm' in player_row else 0
    creativity = player_row['Creativity_Score_norm'].iloc[0] if 'Creativity_Score_norm' in player_row else 0
    
    def interpret(v):
        if v > 2: return "Elite"
        elif v > 0.5: return "Above Average"
        elif v > -0.5: return "Average"
        else: return "Below Average"
    
    return f"""Player: {player_name}
Nationality: {nationality}
Archetype: {archetype}

Key Stats:
- Goals: {interpret(goals)} ({round(goals, 2)})
- Tackles: {interpret(tackles)} ({round(tackles, 2)})
- Passing: {interpret(passing)} ({round(passing, 2)})
- Creativity: {interpret(creativity)} ({round(creativity, 2)})"""

def compare_players(p1, p2):
    row1 = unique_players[unique_players['player'] == p1]
    row2 = unique_players[unique_players['player'] == p2]
    
    if row1.empty or row2.empty:
        return None
    
    arch1 = archetype_names[int(row1['archetype'].iloc[0])]
    arch2 = archetype_names[int(row2['archetype'].iloc[0])]
    
    nat1_code = row1['nation_'].iloc[0] if 'nation_' in row1.columns else ""
    nat2_code = row2['nation_'].iloc[0] if 'nation_' in row2.columns else ""
    nat1 = country_names.get(nat1_code, nat1_code)
    nat2 = country_names.get(nat2_code, nat2_code)
    
    stats = ['Performance_Gls_norm', 'Performance_TklW_norm', 'Total_Cmp%_norm', 'Creativity_Score_norm']
    labels = ['Goals', 'Tackles', 'Passing', 'Creativity']
    
    result = f"{p1} ({arch1} - {nat1}) vs {p2} ({arch2} - {nat2})\n\n"
    
    for stat, label in zip(stats, labels):
        v1 = row1[stat].iloc[0] if stat in row1.columns else 0
        v2 = row2[stat].iloc[0] if stat in row2.columns else 0
        diff = v1 - v2
        result += f"{label}: {round(v1, 2)} vs {round(v2, 2)}"
        if diff > 0:
            result += f" ({p1} leads by {round(diff, 2)})\n"
        elif diff < 0:
            result += f" ({p2} leads by {round(-diff, 2)})\n"
        else:
            result += " (Equal)\n"
    
    return result

def find_similar(p_name, top_n=5):
    player_row = unique_players[unique_players['player'] == p_name]
    if player_row.empty:
        return None
    
    pc1_t = player_row['PC1'].iloc[0]
    pc2_t = player_row['PC2'].iloc[0]
    goals_t = player_row['Performance_Gls_norm'].iloc[0] if 'Performance_Gls_norm' in player_row else 0
    tackles_t = player_row['Performance_TklW_norm'].iloc[0] if 'Performance_TklW_norm' in player_row else 0
    
    results = []
    for _, row in unique_players.iterrows():
        if row['player'] == p_name:
            continue
        
        pca_dist = np.sqrt((row['PC1'] - pc1_t)**2 + (row['PC2'] - pc2_t)**2)
        pca_sim = max(0, 100 - (pca_dist * 15))
        
        goals_sim = max(0, 100 - (abs(row.get('Performance_Gls_norm', 0) - goals_t) * 30))
        tackles_sim = max(0, 100 - (abs(row.get('Performance_TklW_norm', 0) - tackles_t) * 30))
        
        total_sim = (pca_sim * 0.5 + goals_sim * 0.25 + tackles_sim * 0.25)
        
        results.append({
            'name': row['player'],
            'archetype': archetype_names[int(row['archetype'])],
            'similarity': round(total_sim, 1)
        })
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]

# Simple tool functions (no decorators needed)
def get_player_stats_tool(player_name: str) -> str:
    result = get_player_stats_summary(player_name)
    return result if result else f"Player '{player_name}' not found."

def compare_players_tool(player1: str, player2: str) -> str:
    result = compare_players(player1, player2)
    return result if result else f"One or both players not found."

def find_similar_players_tool(player_name: str) -> str:
    results = find_similar(player_name, top_n=5)
    if not results:
        return f"No similar players found for '{player_name}'."
    output = f"Players similar to {player_name}:\n"
    for i, p in enumerate(results, 1):
        output += f"{i}. {p['name']} ({p['archetype']}) - {p['similarity']}% similar\n"
    return output

tools = [get_player_stats_tool, compare_players_tool, find_similar_players_tool]

# Create the agent
agent_executor = create_react_agent(model=llm, tools=tools)

# Chat interface
st.subheader("Chat with the True AI Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about football players..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent_executor.invoke({"messages": [("user", prompt)]})
                answer = response["messages"][-1].content
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {e}")
                answer = "Sorry, I encountered an error."
                st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
