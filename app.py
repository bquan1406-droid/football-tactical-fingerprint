import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px

st.set_page_config(page_title="Football Tactical Fingerprint", layout="wide")

st.title("Football Tactical Fingerprint")
st.markdown("Discover any player's tactical archetype based on 8 seasons of European top 5 league data.")

# Load all models and data
@st.cache_resource
def load_models():
    kmeans = joblib.load('models/kmeans_model.pkl')
    pca = joblib.load('models/pca_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('models/features_list.pkl', 'rb') as f:
        features_list = pickle.load(f)
    with open('models/archetype_names.pkl', 'rb') as f:
        archetype_names = pickle.load(f)
    return kmeans, pca, scaler, features_list, archetype_names

@st.cache_data
def load_data():
    df = pd.read_csv('data/player_profiles.csv')
    return df

kmeans, pca, scaler, features_list, archetype_names = load_models()
player_df = load_data()

# Player selection dropdown
player_list = sorted(player_df['player'].unique())
selected_player = st.selectbox("Select a player", player_list)

# Get player data
player_row = player_df[player_df['player'] == selected_player].iloc[0]
archetype_id = int(player_row['archetype'])
archetype_name = archetype_names[archetype_id]
pc1 = player_row['PC1']
pc2 = player_row['PC2']

# Display results
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader(selected_player)
    st.metric("Archetype", archetype_name)
    st.markdown(f"**PC1 (Attacking vs Defending):** {pc1:.2f}")
    st.markdown(f"**PC2 (Passing vs Carrying):** {pc2:.2f}")

# Archetype descriptions
archetype_descriptions = {
    0: "Central Midfielder - Box-to-box engine, tackles and carries effectively.",
    1: "Center Back - Traditional defender, focuses on stopping attacks.",
    2: "Complete Forward - Elite scorer, creator, and dribbler.",
    3: "Ball-Playing Defender - Defends and builds from the back with passing.",
    4: "Poacher - Pure finisher, stays in the box.",
    5: "Wide Midfielder - Provides width, carries ball forward.",
    6: "Attacking Midfielder - Creative scorer, operates in final third."
}

with col2:
    st.markdown("Description")
    st.info(archetype_descriptions[archetype_id])

# PCA Visualization
st.markdown("---")
st.subheader("PCA Space Visualization")

# Create scatter plot of all players colored by archetype
player_df['archetype_name'] = player_df['archetype'].map(archetype_names)

fig = px.scatter(
    player_df, 
    x='PC1', 
    y='PC2', 
    color='archetype_name',
    hover_name='player',
    title='Player Map - PCA Space',
    labels={'PC1': 'Attacking vs Defending', 'PC2': 'Passing vs Carrying'},
    color_discrete_sequence=px.colors.qualitative.Set1
)

# Highlight selected player
fig.add_scatter(
    x=[pc1], 
    y=[pc2], 
    mode='markers',
    marker=dict(size=20, symbol='star', color='black', line=dict(width=2, color='white')),
    name=selected_player,
    hoverinfo='name'
)

st.plotly_chart(fig, use_container_width=True)

# Summary
st.markdown("---")
st.markdown("**7 Archetypes Discovered**")
st.markdown("""
- Central Midfielder
- Center Back
- Complete Forward
- Ball-Playing Defender
- Poacher
- Wide Midfielder
- Attacking Midfielder
""")
