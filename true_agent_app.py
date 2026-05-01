import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from difflib import get_close_matches
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

st.set_page_config(page_title="True AI Agent - Football Tactical Analyst", layout="wide")

st.title("True AI Agent")
st.markdown("Ask anything naturally. The AI decides what to do.")

# Load data and models
@st.cache_resource
def load_data():
    df = pd.read_csv('data/player_profiles_full.csv')
    with open('models/archetype_names.pkl', 'rb') as f:
        archetype_names = pickle.load(f)
    unique_players = df.drop_duplicates(subset=['player']).copy()
    return df, unique_players, archetype_names

# Initialize LLM
@st.cache_resource
def load_llm():
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)

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
df, unique_players, archetype_names = load_data()
llm = load_llm()


# KEEP EXISTING FUNCTIONS 


def get_player_archetype(player_name):
    player_row = unique_players[unique_players['player'] == player_name]
    if player_row.empty:
        return None
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
        return None
    
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

def compare_players(player1_name, player2_name):
    player1 = unique_players[unique_players['player'] == player1_name]
    player2 = unique_players[unique_players['player'] == player2_name]
    
    if player1.empty or player2.empty:
        return None
    
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
        diff = val1 - val2
        result += f"{label}: {round(val1, 2)} vs {round(val2, 2)}"
        if diff > 0:
            result += f" ({player1_name} leads by {round(diff, 2)})\n"
        elif diff < 0:
            result += f" ({player2_name} leads by {round(-diff, 2)})\n"
        else:
            result += " (Equal)\n"
    
    return result

def find_similar_scouting(player_name, top_n=5):
    player_row = unique_players[unique_players['player'] == player_name]
    if player_row.empty:
        return None
    
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


# STEP 3: WRAP FUNCTIONS AS TOOLS FOR THE AGENT


@tool
def get_player_stats(player_name: str) -> str:
    """Get the tactical archetype and key stats (goals, tackles, passing, creativity) for a football player."""
    result = get_player_stats_summary(player_name)
    if result is None:
        return f"Player '{player_name}' not found in database."
    return result

@tool
def compare_two_players(player1: str, player2: str) -> str:
    """Compare two football players side-by-side including their stats and archetypes."""
    result = compare_players(player1, player2)
    if result is None:
        return f"One or both players not found. Please check the names."
    return result

@tool
def find_similar_players(player_name: str) -> str:
    """Find players with a similar playing style to a given player. Returns top 5 similar players with similarity percentages."""
    results = find_similar_scouting(player_name, top_n=5)
    if results is None or len(results) == 0:
        return f"No similar players found for '{player_name}'."
    
    output = f"Players similar to {player_name}:\n"
    for i, p in enumerate(results, 1):
        output += f"{i}. {p['name']} ({p['archetype']}) - {p['similarity']}% similar\n"
    return output

# List of all tools the agent can use
tools = [get_player_stats, compare_two_players, find_similar_players]


# STEP 4: CREATE THE AGENT (NO IF/ELSE)


# Create the ReAct agent - LLM decides which tool to call
agent_executor = create_react_agent(model=llm, tools=tools)


# STEP 5: CHAT INTERFACE

st.subheader("Chat with the True AI Agent")

# Add example prompts
with st.expander("Example questions (click to try)"):
    st.markdown("""
    - What is the playing style of Mohamed Salah?
    - Compare Messi and Ronaldo
    - Who is better defensively, Rodri or Casemiro?
    - Find players similar to Kevin De Bruyne
    - What archetype is Virgil van Dijk?
    - Tell me about Kylian Mbappe
    - Which player has elite creativity?
    """)

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
                # Invoke the agent - the LLM decides what to do
                response = agent_executor.invoke({"messages": [("user", prompt)]})
                answer = response["messages"][-1].content
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {e}")
                answer = f"Sorry, I encountered an error. Please try again."
                st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
