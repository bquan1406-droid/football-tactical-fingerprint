import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from math import pi

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
    df = pd.read_csv('data/player_profiles_full.csv')
    return df

kmeans, pca, scaler, features_list, archetype_names = load_models()
player_df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Mode", ["Single Player Analysis", "Compare Two Players", "Archetype Explorer"])

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

# Archetype colors for consistent visualization
archetype_colors = {
    0: '#1f77b4',  # blue
    1: '#ff7f0e',  # orange
    2: '#2ca02c',  # green
    3: '#d62728',  # red
    4: '#9467bd',  # purple
    5: '#8c564b',  # brown
    6: '#e377c2'   # pink
}

# Function to create radar chart
def create_radar_chart(player_name, player_row):
    # Normalized stats for radar (0-1 scale)
    radar_stats = {
        'Goals': max(0, min(1, (player_row.get('Performance_Gls_norm', 0) + 3) / 6)),
        'Tackles': max(0, min(1, (player_row.get('Performance_TklW_norm', 0) + 3) / 6)),
        'Passing': max(0, min(1, (player_row.get('Total_Cmp%_norm', 0) + 3) / 6)),
        'Carrying': max(0, min(1, (player_row.get('Progression_PrgC_norm', 0) + 3) / 6)),
        'Creativity': max(0, min(1, (player_row.get('Creativity_Score_norm', 0) + 3) / 6))
    }
    
    categories = list(radar_stats.keys())
    values = list(radar_stats.values())
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    values += values[:1]
    angles += angles[:1]
    
    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=player_name,
            line_color='#2ca02c'
        ),
        layout=go.Layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=f"{player_name} - Tactical Profile"
        )
    )
    return fig

# Mode 1: Single Player Analysis
if mode == "Single Player Analysis":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        player_list = sorted(player_df['player'].unique())
        selected_player = st.selectbox("Select a player", player_list, key="single_select")
    
    player_row = player_df[player_df['player'] == selected_player].iloc[0]
    archetype_id = int(player_row['archetype'])
    archetype_name = archetype_names[archetype_id]
    pc1 = player_row['PC1']
    pc2 = player_row['PC2']
    
    # Display player info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Player", selected_player)
    with col2:
        st.metric("Archetype", archetype_name)
    with col3:
        st.metric("PC1 (Attack-Defend)", f"{pc1:.2f}")
    
    st.markdown("---")
    
    # Two columns for radar and description
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tactical Profile")
        radar_fig = create_radar_chart(selected_player, player_row)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with col2:
        st.subheader("Archetype Description")
        st.info(archetype_descriptions[archetype_id])
        st.markdown("**PC2 (Passing vs Carrying):** " + f"{pc2:.2f}")
        st.markdown("---")
        st.subheader("Similar Players")
        # Find similar players (based on PC1 and PC2 distance)
        player_df['distance'] = np.sqrt((player_df['PC1'] - pc1)**2 + (player_df['PC2'] - pc2)**2)
        similar_players = player_df.nsmallest(6, 'distance')
        similar_players = similar_players[similar_players['player'] != selected_player]
        for _, row in similar_players.head(5).iterrows():
            st.markdown(f"- {row['player']} ({archetype_names[int(row['archetype'])]})")
    
    # PCA Visualization
    st.markdown("---")
    st.subheader("Player Map - PCA Space")
    
    player_df['archetype_name'] = player_df['archetype'].map(archetype_names)
    player_df['color'] = player_df['archetype'].map(archetype_colors)
    
    fig = px.scatter(
        player_df, 
        x='PC1', 
        y='PC2', 
        color='archetype_name',
        hover_name='player',
        title='All Players in PCA Space',
        labels={'PC1': 'Attacking vs Defending (+ = Attacking)', 'PC2': 'Passing vs Carrying (+ = Passing)'},
        color_discrete_sequence=px.colors.qualitative.Set1,
        opacity=0.6
    )
    
    fig.add_scatter(
        x=[pc1], 
        y=[pc2], 
        mode='markers',
        marker=dict(size=15, symbol='star', color='black', line=dict(width=2, color='white')),
        name=selected_player,
        hoverinfo='name'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Mode 2: Compare Two Players
elif mode == "Compare Two Players":
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.selectbox("Select Player 1", sorted(player_df['player'].unique()), key="compare1")
    with col2:
        player2 = st.selectbox("Select Player 2", sorted(player_df['player'].unique()), key="compare2")
    
    if player1 and player2:
        row1 = player_df[player_df['player'] == player1].iloc[0]
        row2 = player_df[player_df['player'] == player2].iloc[0]
        
        archetype1 = archetype_names[int(row1['archetype'])]
        archetype2 = archetype_names[int(row2['archetype'])]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(player1, archetype1)
        with col2:
            st.metric(player2, archetype2)
        with col3:
            distance = np.sqrt((row1['PC1'] - row2['PC1'])**2 + (row1['PC2'] - row2['PC2'])**2)
            st.metric("Tactical Similarity", f"{max(0, 100 - distance*20):.1f}%")
        
        # Radar chart comparison
        st.subheader("Tactical Profile Comparison")
        
        radar_stats1 = {
            'Goals': max(0, min(1, (row1.get('Performance_Gls_norm', 0) + 3) / 6)),
            'Tackles': max(0, min(1, (row1.get('Performance_TklW_norm', 0) + 3) / 6)),
            'Passing': max(0, min(1, (row1.get('Total_Cmp%_norm', 0) + 3) / 6)),
            'Carrying': max(0, min(1, (row1.get('Progression_PrgC_norm', 0) + 3) / 6)),
            'Creativity': max(0, min(1, (row1.get('Creativity_Score_norm', 0) + 3) / 6))
        }
        
        radar_stats2 = {
            'Goals': max(0, min(1, (row2.get('Performance_Gls_norm', 0) + 3) / 6)),
            'Tackles': max(0, min(1, (row2.get('Performance_TklW_norm', 0) + 3) / 6)),
            'Passing': max(0, min(1, (row2.get('Total_Cmp%_norm', 0) + 3) / 6)),
            'Carrying': max(0, min(1, (row2.get('Progression_PrgC_norm', 0) + 3) / 6)),
            'Creativity': max(0, min(1, (row2.get('Creativity_Score_norm', 0) + 3) / 6))
        }
        
        categories = list(radar_stats1.keys())
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(radar_stats1.values()),
            theta=categories,
            fill='toself',
            name=player1,
            line_color='#1f77b4'
        ))
        fig.add_trace(go.Scatterpolar(
            r=list(radar_stats2.values()),
            theta=categories,
            fill='toself',
            name=player2,
            line_color='#ff7f0e'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Player Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # PCA comparison plot
        fig2 = px.scatter(
            player_df, 
            x='PC1', 
            y='PC2', 
            color='archetype_name',
            hover_name='player',
            title='Player Positions in PCA Space',
            labels={'PC1': 'Attacking vs Defending', 'PC2': 'Passing vs Carrying'},
            opacity=0.3
        )
        fig2.add_scatter(x=[row1['PC1']], y=[row1['PC2']], mode='markers', marker=dict(size=15, color='blue', symbol='circle'), name=player1)
        fig2.add_scatter(x=[row2['PC1']], y=[row2['PC2']], mode='markers', marker=dict(size=15, color='orange', symbol='circle'), name=player2)
        st.plotly_chart(fig2, use_container_width=True)

# Mode 3: Archetype Explorer
else:
    st.subheader("Archetype Explorer")
    
    selected_archetype = st.selectbox("Select Archetype", list(archetype_names.values()))
    
    # Find archetype id from name
    archetype_id = [k for k, v in archetype_names.items() if v == selected_archetype][0]
    
    # Get players in this archetype
    archetype_players = player_df[player_df['archetype'] == archetype_id]
    
    st.markdown(f"**{len(archetype_players)} players** belong to this archetype")
    
    st.markdown(archetype_descriptions[archetype_id])
    
    st.subheader("Example Players")
    example_players = archetype_players.head(10)
    for _, row in example_players.iterrows():
        st.markdown(f"- {row['player']}")
    
    # Show PCA distribution for this archetype
    fig = px.scatter(
        player_df, 
        x='PC1', 
        y='PC2', 
        color='archetype_name',
        hover_name='player',
        title=f'{selected_archetype} - PCA Distribution',
        labels={'PC1': 'Attacking vs Defending', 'PC2': 'Passing vs Carrying'},
        color_discrete_sequence=[archetype_colors[archetype_id]] if archetype_id in archetype_colors else None,
        opacity=0.6
    )
    fig.add_scatter(
        x=archetype_players['PC1'], 
        y=archetype_players['PC2'], 
        mode='markers',
        marker=dict(size=10, color='red', symbol='circle'),
        name=selected_archetype,
        hoverinfo='skip'
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with PCA + K-Means clustering on 14k+ players from Europe's top 5 leagues (2017-2025)")
