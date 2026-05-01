import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import re
from difflib import get_close_matches
from langchain_groq import ChatGroq

st.set_page_config(page_title="True AI Agent - Football Tactical Analyst", layout="wide")

st.title("True AI Agent")
st.markdown("Ask anything naturally. The AI combines data and football knowledge.")

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

# ============================================
# NICKNAME MAPPING for famous players
# ============================================
nickname_to_player = {
    'messi': 'Lionel Messi',
    'ronaldo': 'Cristiano Ronaldo',
    'cr7': 'Cristiano Ronaldo',
    'mbappe': 'Kylian Mbappe',
    'neymar': 'Neymar',
    'salah': 'Mohamed Salah',
    'kdb': 'Kevin De Bruyne',
    'vvd': 'Virgil van Dijk',
    'rodri': 'Rodri',
    'haaland': 'Erling Haaland',
    'lewa': 'Robert Lewandowski',
    'lewy': 'Robert Lewandowski',
    'kane': 'Harry Kane',
    'son': 'Son Heung-Min',
    'sterling': 'Raheem Sterling',
    'foden': 'Phil Foden',
    'bernardo': 'Bernardo Silva',
    'gundo': 'Ilkay Gundogan',
    'kante': 'N Golo Kante',
    'hazard': 'Eden Hazard',
    'pique': 'Gerard Pique',
    'ramos': 'Sergio Ramos',
    'modric': 'Luka Modric',
    'kroos': 'Toni Kroos',
    'benzema': 'Karim Benzema',
    'vini': 'Vinicius Junior',
    'saka': 'Bukayo Saka',
    'odegaard': 'Martin Odegaard',
    'rice': 'Declan Rice',
    'bellingham': 'Jude Bellingham',
    'pedri': 'Pedri',
    'gavi': 'Gavi',
}

# Famous players not in dataset (will use LLM knowledge)
famous_players_not_in_dataset = [
    'Lionel Messi', 'Cristiano Ronaldo', 'Neymar', 'Kylian Mbappe',
    'Zlatan Ibrahimovic', 'Andres Iniesta', 'Xavi', 'Carles Puyol'
]

# ============================================
# CORE FUNCTIONS
# ============================================

def resolve_player_name(input_name):
    """Convert nickname or partial name to full player name"""
    input_lower = input_name.lower().strip()
    
    # Check nickname map
    if input_lower in nickname_to_player:
        return nickname_to_player[input_lower]
    
    # Check if name is in dataset
    player_row = unique_players[unique_players['player'] == input_name]
    if not player_row.empty:
        return input_name
    
    # Try fuzzy match
    all_players = unique_players['player'].tolist()
    matches = get_close_matches(input_name, all_players, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    
    return None

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

def get_llm_biography(player_name):
    """Get biographical info from LLM for famous players not in dataset"""
    prompt = f"""Provide a brief biography of football player {player_name}. Include:
- Full name
- Date of birth (year only)
- Position
- Current or most recent club
- Key achievements (Ballon d'Or, Champions League, World Cup, league titles)
- Playing style (3-4 words)

Keep it concise, 4-5 lines maximum.
"""
    response = llm.invoke(prompt)
    return response.content

def get_llm_comparison(p1, p2):
    """Get comparison from LLM for famous players not in dataset"""
    prompt = f"""Compare the two football players {p1} and {p2}. Include:
1. Their playing styles
2. Key strengths of each
3. Key achievements
4. Who is considered better overall and why

Keep it concise, 6-8 lines maximum.
"""
    response = llm.invoke(prompt)
    return response.content

# ============================================
# MAIN ANSWER FUNCTION
# ============================================

def answer_question(question):
    question_lower = question.lower()
    
    # Extract player names using nickname mapping
    found_players = []
    
    # Check nicknames first
    for nickname, full_name in nickname_to_player.items():
        if nickname in question_lower:
            found_players.append(full_name)
    
    # Check exact player names in dataset
    all_players = unique_players['player'].tolist()
    for player in all_players:
        if player.lower() in question_lower and player not in found_players:
            found_players.append(player)
    
    # Remove duplicates
    found_players = list(dict.fromkeys(found_players))
    
    # Check if players are in dataset vs famous
    players_in_dataset = [p for p in found_players if p in all_players]
    players_not_in_dataset = [p for p in found_players if p not in all_players]
    
    # ========================================
    # COMPARISON QUESTIONS
    # ========================================
    if ("compare" in question_lower or "vs" in question_lower or "versus" in question_lower or "better" in question_lower) and len(found_players) >= 2:
        p1, p2 = found_players[0], found_players[1]
        
        # Both in dataset
        if p1 in all_players and p2 in all_players:
            comparison = compare_players(p1, p2)
            if comparison:
                return comparison
        
        # One or both not in dataset - use LLM
        else:
            return get_llm_comparison(p1, p2)
    
    # ========================================
    # SINGLE PLAYER QUESTIONS
    # ========================================
    if found_players:
        player = found_players[0]
        
        # Player is in dataset - use stats + add LLM context
        if player in all_players:
            stats = get_player_stats_summary(player)
            
            # Add LLM context for playing style
            context_prompt = f"""Based on these stats for {player}, describe their playing style in 2 sentences:

Stats: {stats}

Playing style:"""
            style_response = llm.invoke(context_prompt)
            style = style_response.content
            
            return f"{stats}\n\n---\n\n**Playing Style:**\n{style}"
        
        # Player not in dataset - use LLM biography
        else:
            bio = get_llm_biography(player)
            return bio
    
    # ========================================
    # SIMILAR PLAYERS
    # ========================================
    if ("similar" in question_lower or "like" in question_lower):
        for player in all_players:
            if player.lower() in question_lower:
                results = find_similar(player, top_n=5)
                if results:
                    output = f"Players similar to {player}:\n"
                    for i, p in enumerate(results, 1):
                        output += f"{i}. {p['name']} ({p['archetype']}) - {p['similarity']}% similar\n"
                    return output
    
    # ========================================
    # FALLBACK TO LLM
    # ========================================
    prompt = f"""Answer this football question briefly and helpfully. If you don't know, say so.

Question: {question}

Answer:"""
    response = llm.invoke(prompt)
    return response.content

# ============================================
# CHAT INTERFACE
# ============================================

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
                answer = answer_question(prompt)
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {e}")
                answer = "Sorry, I encountered an error."
                st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
