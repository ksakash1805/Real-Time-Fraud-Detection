"""
Exploratory Data Analysis (EDA) Module.

This module provides functions to visualize and understand the characteristics
of the dataset, focusing on the differences between fraud and legitimate transactions.

MENTOR NOTE: EDA is not just making pretty pictures. It's about finding signal in the noise, 
understanding feature distributions, and deciding how to preprocess the data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Optional

from src.config import PLOTS_DIR, PCA_FEATURES

logger = logging.getLogger(__name__)

def _ensure_dir(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

def plot_class_distribution(df: pd.DataFrame, save_dir: str):
    """
    Plots a bar chart showing the severe class imbalance.
    
    Args:
        df (pd.DataFrame): The dataset.
        save_dir (str): Directory to save the plot.
    """
    _ensure_dir(save_dir)
    plt.figure(figsize=(8, 6))
    
    # MENTOR NOTE: Always annotate counts and percentages for imbalanced data. 
    # A bar chart alone might make the minority class invisible if it's < 0.2%!
    ax = sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Legit, 1: Fraud)')
    
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 1000,
                f'{height} ({height/total:.2%})',
                ha="center", fontsize=10)
                
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
    plt.close()

def plot_feature_distributions(df: pd.DataFrame, save_dir: str):
    """
    Plots the distribution of Amount and Time features, split by class.
    
    Args:
        df (pd.DataFrame): The dataset.
        save_dir (str): Directory to save the plot.
    """
    _ensure_dir(save_dir)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # MENTOR NOTE: Amount and Time are the only non-PCA features. 
    # Looking at their distributions by class can reveal if they hold predictive power out-of-the-box.
    # For instance, do frauds happen at specific times? Are they typically larger or smaller amounts?
    
    sns.histplot(data=df, x='Amount', hue='Class', bins=50, ax=axes[0], log_scale=(False, True))
    axes[0].set_title('Transaction Amount Distribution (Log Scale y)')
    
    sns.histplot(data=df, x='Time', hue='Class', bins=50, ax=axes[1])
    axes[1].set_title('Transaction Time Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_distributions.png'))
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str):
    """
    Plots a correlation heatmap focusing on features highly correlated with Class.
    
    Args:
        df (pd.DataFrame): The dataset.
        save_dir (str): Directory to save the plot.
    """
    _ensure_dir(save_dir)
    plt.figure(figsize=(12, 10))
    
    # Extract correlations with Class
    corr = df.corr()
    
    # MENTOR NOTE: We don't need to see the entire 30x30 matrix. It's often better to 
    # look at the features that have the strongest linear relationship with the target.
    # (Though remember, tree models can find non-linear relationships that correlation misses!)
    top_corr_features = corr.index[abs(corr["Class"]) > 0.1]
    
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap (Highly Correlated Features)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'))
    plt.close()

def plot_outlier_analysis(df: pd.DataFrame, save_dir: str):
    """
    Creates boxplots to visualize outliers in key PCA features.
    
    Args:
        df (pd.DataFrame): The dataset.
        save_dir (str): Directory to save the plot.
    """
    _ensure_dir(save_dir)
    # Pick a few key features that are known to be separated well
    # MENTOR NOTE: Box plots are great for identifying outliers. In fraud detection, 
    # outliers in the legit class might actually be undetected frauds, and outliers in 
    # the fraud class might represent different *types* of fraud.
    
    features = ['V14', 'V11', 'V12', 'V4']
    fig, axes = plt.subplots(1, len(features), figsize=(20, 6))
    
    for i, feature in enumerate(features):
        sns.boxplot(x="Class", y=feature, data=df, ax=axes[i])
        axes[i].set_title(f'{feature} vs Class')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'outlier_analysis.png'))
    plt.close()

def plot_fraud_vs_legit_distributions(df: pd.DataFrame, save_dir: str):
    """
    Plots KDE for the top 9 most discriminative PCA features.
    
    Args:
        df (pd.DataFrame): The dataset.
        save_dir (str): Directory to save the plot.
    """
    _ensure_dir(save_dir)
    
    # MENTOR NOTE: Overlaid KDE plots help us visualize class separability. 
    # If the curves for Legit (0) and Fraud (1) overlap completely, the feature is useless. 
    # If they are distinct, the feature is highly predictive.
    
    top_features = ['V14', 'V11', 'V12', 'V4', 'V3', 'V10', 'V16', 'V17', 'V9']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        sns.kdeplot(df[df['Class'] == 0][feature], label='Legit', ax=axes[i], fill=True, color='blue', alpha=0.3)
        sns.kdeplot(df[df['Class'] == 1][feature], label='Fraud', ax=axes[i], fill=True, color='red', alpha=0.3)
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kde_distributions.png'))
    plt.close()

def run_full_eda(df: pd.DataFrame, save_dir: Optional[str] = None):
    """
    Runs the complete suite of EDA visualizations.
    
    Args:
        df (pd.DataFrame): The dataset.
        save_dir (str, optional): Directory to save plots. Defaults to PLOTS_DIR.
    """
    save_dir = str(save_dir or PLOTS_DIR)
    logger.info(f"Running full EDA, saving plots to {save_dir}...")
    
    plot_class_distribution(df, save_dir)
    plot_feature_distributions(df, save_dir)
    plot_correlation_heatmap(df, save_dir)
    plot_outlier_analysis(df, save_dir)
    plot_fraud_vs_legit_distributions(df, save_dir)
    
    logger.info("EDA completed.")
