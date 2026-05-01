import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from difflib import get_close_matches
from langchain_groq import ChatGroq

st.set_page_config(page_title="Football Tactical AI Agent", layout="wide")

st.title("Football Tactical AI Agent")
st.markdown("Ask anything about player archetypes, comparisons, and similar players.")

# Load data and models
@st.cache_resource
def load_data():
    df = pd.read_csv('data/player_profiles_full.csv')
    
    with open('models/archetype_names.pkl', 'rb') as f:
        archetype_names = pickle.load(f)
    
    unique_players = df.drop_duplicates(subset=['player']).copy()
    
    return df, unique_players, archetype_names

# Initialize LLM for analysis
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

def get_players_by_archetype(archetype_name, limit=10):
    arch_id = None
    for aid, aname in archetype_names.items():
        if aname.lower() == archetype_name.lower():
            arch_id = aid
            break
    
    if arch_id is None:
        return f"Archetype '{archetype_name}' not found."
    
    players = unique_players[unique_players['archetype'] == arch_id].head(limit)
    result = f"Players in {archetype_name} archetype:\n"
    for _, row in players.iterrows():
        result += f"- {row['player']} ({row.get('team', 'Unknown')})\n"
    return result

def find_player_fuzzy(search_term):
    all_players = unique_players['player'].tolist()
    matches = get_close_matches(search_term, all_players, n=5, cutoff=0.4)
    return matches

def analyze_player_profile(player_name, stats_text):
    prompt = f"""You are a football tactical analyst at a top European club. Analyze this player profile.

Player data:
{stats_text}

Write a short, insightful analysis (3-4 sentences) covering:
1. The player's playing style and tactical role
2. Their key strengths and weaknesses
3. What type of team system would maximize their impact

Use professional football terminology. Be specific and data-driven.
"""
    
    response = llm.invoke(prompt)
    return response.content

def analyze_comparison(player1_name, player2_name, comparison_text):
    prompt = f"""You are a football tactical analyst at a top European club. Analyze this player comparison.

Comparison data:
{comparison_text}

Write a short, insightful analysis (4-5 sentences) covering:
1. The key tactical difference between these players
2. What each player does better than the other
3. What kind of team system would suit each player
4. If you were a sporting director, which would you sign and why

Use professional football terminology. Be specific and data-driven.
"""
    
    response = llm.invoke(prompt)
    return response.content

def analyze_similar_players(player_name, similar_text):
    prompt = f"""You are a football tactical analyst. Analyze these similar players.

Target player: {player_name}
Similar players found:
{similar_text}

Write a short analysis (2-3 sentences) explaining:
1. What pattern or playing style connects these players
2. What this suggests about the target player's tactical profile
"""
    
    response = llm.invoke(prompt)
    return response.content

# Direct question answering function
def answer_question(question):
    question_lower = question.lower()
    
    all_players = unique_players['player'].tolist()
    found_players = [p for p in all_players if p.lower() in question_lower]
    
    # Archetype question with analysis
    if "archetype" in question_lower and found_players:
        stats = get_player_stats_summary(found_players[0])
        if stats:
            analysis = analyze_player_profile(found_players[0], stats)
            return f"{stats}\n\n---\n\n**Tactical Analysis:**\n{analysis}"
        return stats
    
    # Stats summary with analysis
    if ("stats" in question_lower or "summary" in question_lower) and found_players:
        stats = get_player_stats_summary(found_players[0])
        if stats:
            analysis = analyze_player_profile(found_players[0], stats)
            return f"{stats}\n\n---\n\n**Tactical Analysis:**\n{analysis}"
        return stats
    
    # Comparison with analysis
    if "compare" in question_lower and len(found_players) >= 2:
        comparison = compare_players(found_players[0], found_players[1])
        if comparison:
            analysis = analyze_comparison(found_players[0], found_players[1], comparison)
            return f"{comparison}\n\n---\n\n**Tactical Analysis:**\n{analysis}"
        return comparison
    
    # Similar players with analysis
    if ("similar" in question_lower or "like" in question_lower) and found_players:
        results = find_similar_scouting(found_players[0], top_n=5)
        if results:
            output = f"Players similar to {found_players[0]}:\n"
            for i, p in enumerate(results, 1):
                output += f"{i}. {p['name']} ({p['archetype']}) - {p['similarity']}% similar\n"
            analysis = analyze_similar_players(found_players[0], output)
            return f"{output}\n\n---\n\n**Analysis:**\n{analysis}"
        return "No similar players found."
    
    # List players by archetype
    for arch_name in archetype_names.values():
        if arch_name.lower() in question_lower:
            return get_players_by_archetype(arch_name, limit=10)
    
    # Fuzzy search
    if not found_players:
        search_term = question_lower.replace("who is ", "").replace("what is ", "").strip()
        matches = find_player_fuzzy(search_term)
        if matches:
            return f"Player '{search_term}' not found. Did you mean: {', '.join(matches[:3])}?"
    
    # Default response
    if found_players:
        stats = get_player_stats_summary(found_players[0])
        if stats:
            analysis = analyze_player_profile(found_players[0], stats)
            return f"{stats}\n\n---\n\n**Tactical Analysis:**\n{analysis}"
        return stats
    
    return "I can help with player archetypes, stats, comparisons, and similar players. Try asking: 'What archetype is Mohamed Salah?'"

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
        with st.spinner("Analyzing..."):
            answer = answer_question(prompt)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
