# NBA Career Points Prediction

This repository contains a Jupyter notebook implementing a machine learning project to predict NBA players' career average points per game (PTS) using deep learning and ensemble methods. The project compares three models: a Feed-Forward Neural Network (FNN), a Convolutional Neural Network (CNN), and XGBoost, leveraging physical and draft-related attributes from the NBA Players Dataset.

## Project Overview

The goal is to predict career average PTS based on pre-NBA attributes, such as height, weight, position, and draft year, to aid talent scouting. The notebook includes:
- **Data Collection**: Aggregates player data from the NBA Players Dataset to compute career averages.
- **Exploratory Data Analysis (EDA)**: Visualizes distributions, correlations, and cleans data (e.g., handling missing values, encoding categorical variables).
- **Feature Engineering**: Creates features like `HEIGHT_WEIGHT_RATIO`, `POSITION_SCORE`, and `DRAFT_YEAR_NORM`.
- **Model Development**: Implements FNN, CNN, and XGBoost with cross-validation and hyperparameter tuning.
- **Evaluation**: Assesses models using RMSE and R² metrics.
- **Discussion**: Analyzes results, feature importance, limitations, and future work.

### Key Findings

- **Model Performance**: Current models (FNN: RMSE = 2.604043, R² = -0.099646; CNN: RMSE = 2.582621, R² = -0.081628; XGBoost: RMSE = 2.506526, R² = -0.018829) underperform, with negative R² values indicating predictions are worse than the mean PTS. XGBoost shows the best performance but is still inadequate.
- **Feature Importance**: Features like `POSITION_SCORE` and `DRAFT_YEAR_NORM` have limited predictive power for PTS.
- **Limitations**: The dataset lacks performance metrics (e.g., minutes played, shooting efficiency), and restrictive filtering (PTS: 2–20) reduces data diversity (973 training samples).
- **Future Work**: Incorporate features like `DRAFT_NUMBER`, relax filtering, and explore simpler models or additional data sources.

## Dataset

> The data can be found at https://www.kaggle.com/datasets/yagizfiratt/nba-players-database

The dataset (`PlayerIndex_nba_stats.csv`) contains NBA player attributes and seasonal statistics, including:
- **Columns**: `PERSON_ID`, `HEIGHT`, `WEIGHT`, `POSITION`, `DRAFT_YEAR`, `DRAFT_NUMBER`, `PTS`, etc.
- **Source**: Aggregated from NBA statistics platforms (e.g., Basketball-Reference.com).
- **Preprocessing**: 
  - Aggregates by `PERSON_ID` to compute career average PTS.
  - Handles missing values (e.g., `DRAFT_YEAR` imputed with median, `POSITION` encoded).
  - Filters outliers (e.g., PTS: 2–20, HEIGHT: 72–84, WEIGHT: 180–260).

## Results

The current models underperform due to weak feature correlations and a small dataset:
- **FNN**: RMSE = 2.604043, R² = -0.099646
- **CNN**: RMSE = 2.582621, R² = -0.081628
- **XGBoost**: RMSE = 2.506526, R² = -0.018829

Negative R² values indicate the models predict worse than the mean PTS. XGBoost performs best but is limited by feature quality. Visualizations (e.g., `correlation_matrix.png`, `pts_distribution.png`) and diagnostics suggest low feature-PTS correlations and a narrow PTS range (5–14.9).

## Limitations

- **Feature Deficiency**: Lacks performance metrics (e.g., minutes played, shooting efficiency).
- **Small Dataset**: 973 training samples limit deep learning generalization.
- **Data Filtering**: Restrictive ranges reduce data diversity.
- **CNN Suitability**: Less effective for tabular data.

## Future Improvements

- Add features like `DRAFT_NUMBER` (impute 100 for undrafted players) or college statistics.
- Relax filtering (e.g., PTS: 0–35) to increase sample size.
- Experiment with simpler models (e.g., LinearRegression, RandomForest) or transformers.
- Source additional data from platforms like Basketball-Reference.com.
