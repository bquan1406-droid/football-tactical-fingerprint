# Football Tactical Fingerprint

## Project Overview

Every football club faces the same problem: a player who excels in one tactical system often fails in another. Antoine Griezmann was world-class at Atlético Madrid (counter-attacking) but struggled at Barcelona (possession). The player's talent did not change. The system did.

This project solves that problem by building a **tactical fingerprint** for every player in Europe's top 5 leagues. Instead of asking "Is this player good?" we answer "What kind of player are they? Which system do they fit?"

---

## Business & Sports Context

### The Problem

| Stakeholder | Pain Point | Financial Impact |
|-------------|------------|------------------|
| Football Clubs | 40-50% of big transfers fail | €1.5B+ wasted annually |
| Player Agencies | Clients underperform at wrong club | Lower commission fees |
| Betting Operators | Standard odds ignore tactical mismatches | Missed edge opportunities |
| Fantasy Sports | Users pick stars who don't fit systems | Churn and reduced engagement |

### The Solution

A data-driven system that:
1. Reduces 44 performance statistics into 2 interpretable dimensions
2. Groups players into 7 distinct tactical archetypes
3. Enables evidence-based player-club matching

---

## Data Source

| Detail | Information |
|--------|-------------|
| Source | FBref via Kaggle |
| Leagues | Premier League, La Liga, Serie A, Bundesliga, Ligue 1 |
| Time Period | 8 seasons (2017-18 to 2024-25) |
| Raw Data | 22,929 rows, 178 columns |
| Cleaned Data | 14,475 rows, 153 columns (500+ minutes, no goalkeepers) |

---

## Methodology

### Step 1: Exploratory Data Analysis (Power BI)

Created 8+ interactive visualizations to understand the data:

| Plot | Insight Discovered |
|------|-------------------|
| Players per league | Serie A has 23% of players, Bundesliga 17% |
| Passing vs Tackles | Midfielders average 80% passing, 1.2 tackles/90 |
| Age vs Carries | Forwards decline after age 26 |
| Age distribution | Peak player age is 24-27 |
| Goals by position | Forwards: 8-12 goals/season, Defenders: 1-2 |
| Shot efficiency | Elite forwards have higher goals/shot ratio |

### Step 2: Feature Engineering

| Feature Type | Count | Examples |
|--------------|-------|----------|
| Original stats | 153 | Goals, tackles, passes, carries |
| Per 90 normalization | Built-in | Standardized for playing time |
| Season normalization (z-score) | 39 | Removed era bias (2017 vs 2025) |
| Engineered features | 5 | Goal Efficiency, Creativity Score, Defensive Index |

### Step 3: Dimensionality Reduction (PCA)

| Component | Variance | Interpretation |
|-----------|----------|----------------|
| PC1 | 32.0% | Attacking vs Defending |
| PC2 | 18.9% | Passing vs Carrying |
| PC3-PC5 | ~17% | Efficiency, discipline, aerial ability |

**Total explained variance with 2 components:** 50.9% (acceptable for noisy sports data)

### Step 4: Clustering (K-Means)

Compared 4 algorithms (K-Means, GMM, DBSCAN, t-SNE). Selected K-Means with 7 clusters.

| Metric | Score | Benchmark | Status |
|--------|-------|-----------|--------|
| Silhouette | 0.3739 | 0.30 (NIH study) | ✓ Exceeds |
| Davies-Bouldin | 1.5979 | Lower is better | ✓ Acceptable |
| Calinski-Harabasz | 2,991.96 | Higher is better | ✓ Acceptable |

---

## The 7 Archetypes

| # | Archetype | Key Stats | Example Player |
|---|-----------|-----------|----------------|
| 0 | Central Midfielder | High tackles, high carrying | Granit Xhaka |
| 1 | Center Back | High defending, low carrying | Rob Holding |
| 2 | Complete Forward | Elite goals, creativity, carrying | Mohamed Salah |
| 3 | Ball-Playing Defender | High passing, high tackling | Laurent Koscielny |
| 4 | Poacher | Goals, minimal creation | Jermain Defoe |
| 5 | Wide Midfielder | High carrying, moderate creativity | Alex Iwobi |
| 6 | Attacking Midfielder | Goals, creativity, carries | Aaron Ramsey |

### Archetype Visualization

![PCA Clusters](images/pca_clusters.png)

*PCA projection showing 7 distinct player archetypes. PC1 (X-axis) represents Attacking vs Defending. PC2 (Y-axis) represents Passing vs Carrying.*

---

## Results

### Statistical Validation

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Silhouette Score | 0.3739 | Exceeds academic benchmark (0.30) |
| 80% variance reached | 9 components | Efficient representation |
| Players clustered | 14,288 | Large, reliable sample |

### Business Value

| Use Case | How This Project Helps |
|----------|------------------------|
| Scouting | Identify player archetype from statistical profile |
| Recruitment | Find players who fit team's tactical system |
| Squad Building | Balance archetypes across the roster |
| Transfer Decisions | Compare players within same archetype for value |

### Example Insight

> *"A club with 3 Playmakers and 0 Target Forwards has a tactical imbalance. Our system recommends signing from Archetype 2 (Complete Forward) or Archetype 4 (Poacher)."*

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Python (Pandas, NumPy) |
| Machine Learning | Scikit-learn (PCA, K-Means, GMM, DBSCAN) |
| Visualization | Matplotlib, Seaborn, Plotly, Power BI |
| Deployment | Streamlit, Joblib, GitHub |


