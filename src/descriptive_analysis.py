"""
Malaysian Condominium Price Predictive Model
Part 2: Descriptive Analysis + Create ML-Ready Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("PART 2: DESCRIPTIVE ANALYSIS")
print("="*70)

# ============================================================================
# LOAD CLEANED DATA
# ============================================================================

df = pd.read_csv('data/processed/house_cleaned.csv')
print(f"\n✓ Loaded cleaned data: {df.shape[0]} rows × {df.shape[1]} columns")

# Create reports/figures directory if it doesn't exist
import os
os.makedirs('reports/figures', exist_ok=True)
print("✓ Created reports/figures directory")

# ============================================================================
# STEP 1: UNIVARIATE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("STEP 1: UNIVARIATE ANALYSIS (One variable at a time)")
print("="*70)

# -------------------- PRICE ANALYSIS --------------------
print("\n--- PRICE DISTRIBUTION ---")
print(f"Mean Price:       RM {df['price'].mean():,.2f}")
print(f"Median Price:     RM {df['price'].median():,.2f}")
print(f"Std Deviation:    RM {df['price'].std():,.2f}")
print(f"Min Price:        RM {df['price'].min():,.2f}")
print(f"Max Price:        RM {df['price'].max():,.2f}")
print(f"Skewness:         {df['price'].skew():.2f}")
print(f"Kurtosis:         {df['price'].kurtosis():.2f}")

# Interpretation
if df['price'].skew() > 1:
    print("→ Interpretation: Highly right-skewed (many luxury condos)")
elif df['price'].skew() > 0.5:
    print("→ Interpretation: Moderately right-skewed (some high-end properties)")
else:
    print("→ Interpretation: Fairly symmetric distribution")

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram with KDE
sns.histplot(df['price'], kde=True, bins=50, ax=axes[0], color='steelblue')
axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Price (RM)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['price'].mean(), color='red', linestyle='--', label='Mean')
axes[0].axvline(df['price'].median(), color='green', linestyle='--', label='Median')
axes[0].legend()

# Boxplot
sns.boxplot(x=df['price'], ax=axes[1], color='coral')
axes[1].set_title('Price Boxplot (Outlier Detection)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Price (RM)')

# Log-transformed histogram
sns.histplot(np.log10(df['price']), kde=True, bins=50, ax=axes[2], color='green')
axes[2].set_title('Price Distribution (Log Scale)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Log10(Price)')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('reports/figures/01_price_univariate.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/figures/01_price_univariate.png")
plt.close()

# -------------------- OTHER NUMERICAL VARIABLES --------------------
print("\n--- OTHER NUMERICAL FEATURES ---")

numerical_features = ['Property Size', 'Bedroom', 'Bathroom', 'Completion Year', 
                      '# of Floors', 'Total Units', 'Parking Lot']

summary_stats = []
for col in numerical_features:
    if col in df.columns:
        stats = {
            'Feature': col,
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Std Dev': df[col].std(),
            'Min': df[col].min(),
            'Max': df[col].max()
        }
        summary_stats.append(stats)

summary_df = pd.DataFrame(summary_stats)
print(summary_df.to_string(index=False))

# Interpretations
print("\n→ Key Insights:")
if 'Property Size' in df.columns:
    print(f"  • Typical condo size: {df['Property Size'].median():.0f} sq ft")
if 'Bedroom' in df.columns:
    print(f"  • Most common bedrooms: {df['Bedroom'].mode().values[0]:.0f}")
if 'Bathroom' in df.columns:
    print(f"  • Most common bathrooms: {df['Bathroom'].mode().values[0]:.0f}")

# Visualizations for numerical features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

plot_features = ['Property Size', 'Bedroom', 'Bathroom', 
                 '# of Floors', 'Total Units', 'Parking Lot']

for idx, col in enumerate(plot_features):
    if col in df.columns and idx < 6:
        sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[idx], color='teal')
        axes[idx].set_title(f'{col} Distribution', fontweight='bold')
        axes[idx].axvline(df[col].median(), color='red', linestyle='--', label='Median')
        axes[idx].legend()

plt.tight_layout()
plt.savefig('reports/figures/02_numerical_features.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/figures/02_numerical_features.png")
plt.close()

# ============================================================================
# SAVE DETAILED UNIVARIATE ANALYSIS TO CSV
# ============================================================================

print("\n--- SAVING DETAILED UNIVARIATE ANALYSIS ---")

# 1. Price Analysis Details
price_analysis = {
    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range', 
               'Q1 (25%)', 'Q3 (75%)', 'IQR', 'Skewness', 'Kurtosis'],
    'Value': [
        df['price'].mean(),
        df['price'].median(),
        df['price'].std(),
        df['price'].min(),
        df['price'].max(),
        df['price'].max() - df['price'].min(),
        df['price'].quantile(0.25),
        df['price'].quantile(0.75),
        df['price'].quantile(0.75) - df['price'].quantile(0.25),
        df['price'].skew(),
        df['price'].kurtosis()
    ]
}

# Add interpretations
skew_val = df['price'].skew()
kurt_val = df['price'].kurtosis()

if skew_val > 1:
    skew_interp = "Highly right-skewed (many luxury condos)"
elif skew_val > 0.5:
    skew_interp = "Moderately right-skewed (some high-end properties)"
elif skew_val < -1:
    skew_interp = "Highly left-skewed (many budget condos)"
elif skew_val < -0.5:
    skew_interp = "Moderately left-skewed"
else:
    skew_interp = "Approximately symmetric distribution"

if kurt_val > 3:
    kurt_interp = "Leptokurtic (heavy tails, more outliers)"
elif kurt_val < 3:
    kurt_interp = "Platykurtic (light tails, fewer outliers)"
else:
    kurt_interp = "Mesokurtic (similar to normal distribution)"

price_analysis['Interpretation'] = [
    f"Average condominium price in dataset",
    f"Middle value - 50% of condos cost less than this",
    f"Typical deviation from mean",
    f"Cheapest condominium in dataset",
    f"Most expensive condominium in dataset",
    f"Price range spans RM {(df['price'].max() - df['price'].min()):,.0f}",
    f"25% of condos cost less than this",
    f"75% of condos cost less than this",
    f"Middle 50% of prices fall within this range",
    skew_interp,
    kurt_interp
]

price_df = pd.DataFrame(price_analysis)
price_df.to_csv('data/processed/analysis_price_detailed.csv', index=False)
print("✓ Saved: data/processed/analysis_price_detailed.csv")

# 2. Detailed Numerical Features Analysis with Interpretations
detailed_stats = []
for col in numerical_features:
    if col in df.columns:
        skew_val = df[col].skew()
        kurt_val = df[col].kurtosis()
        
        # Skewness interpretation
        if skew_val > 1:
            skew_interp = "Highly right-skewed"
        elif skew_val > 0.5:
            skew_interp = "Moderately right-skewed"
        elif skew_val < -1:
            skew_interp = "Highly left-skewed"
        elif skew_val < -0.5:
            skew_interp = "Moderately left-skewed"
        else:
            skew_interp = "Approximately symmetric"
        
        # Kurtosis interpretation
        if kurt_val > 3:
            kurt_interp = "Leptokurtic (heavy tails)"
        elif kurt_val < 3:
            kurt_interp = "Platykurtic (light tails)"
        else:
            kurt_interp = "Mesokurtic (normal-like)"
        
        # Outlier detection (IQR method)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_pct = (outliers / len(df)) * 100
        
        stats = {
            'Feature': col,
            'Count': df[col].notna().sum(),
            'Missing': df[col].isna().sum(),
            'Missing_Pct': (df[col].isna().sum() / len(df)) * 100,
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Mode': df[col].mode().values[0] if len(df[col].mode()) > 0 else np.nan,
            'Std_Dev': df[col].std(),
            'Variance': df[col].var(),
            'Min': df[col].min(),
            'Q1': Q1,
            'Q3': Q3,
            'Max': df[col].max(),
            'Range': df[col].max() - df[col].min(),
            'IQR': IQR,
            'Skewness': skew_val,
            'Skewness_Interpretation': skew_interp,
            'Kurtosis': kurt_val,
            'Kurtosis_Interpretation': kurt_interp,
            'Outliers_Count': outliers,
            'Outliers_Pct': outlier_pct,
            'CV_Coefficient_of_Variation': (df[col].std() / df[col].mean()) * 100 if df[col].mean() != 0 else 0
        }
        detailed_stats.append(stats)

detailed_df = pd.DataFrame(detailed_stats)
detailed_df.to_csv('data/processed/analysis_numerical_features_detailed.csv', index=False)
print("✓ Saved: data/processed/analysis_numerical_features_detailed.csv")

# 3. Distribution Quality Assessment for ML
ml_feature_quality = []
for col in numerical_features + ['price']:
    if col in df.columns:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        skew_val = abs(df[col].skew())
        
        # Quality assessment
        if missing_pct > 50:
            quality = "Poor - High missing values"
            ml_recommendation = "Consider dropping or imputation with caution"
        elif missing_pct > 20:
            quality = "Fair - Moderate missing values"
            ml_recommendation = "Use robust imputation methods"
        else:
            quality = "Good - Low missing values"
            ml_recommendation = "Safe to use with median/mean imputation"
        
        # Transformation recommendation based on skewness
        if skew_val > 1:
            transform_rec = "Apply log/sqrt transformation to reduce skewness"
        elif skew_val > 0.5:
            transform_rec = "Consider transformation if model requires normality"
        else:
            transform_rec = "No transformation needed"
        
        ml_feature_quality.append({
            'Feature': col,
            'Missing_Pct': missing_pct,
            'Skewness': df[col].skew(),
            'Quality_Assessment': quality,
            'ML_Recommendation': ml_recommendation,
            'Transformation_Recommendation': transform_rec,
            'Variance': df[col].var(),
            'Low_Variance_Flag': 'Yes' if df[col].var() < 0.01 else 'No'
        })

ml_quality_df = pd.DataFrame(ml_feature_quality)
ml_quality_df.to_csv('data/processed/analysis_ml_feature_quality.csv', index=False)
print("✓ Saved: data/processed/analysis_ml_feature_quality.csv")

# ============================================================================
# STEP 2: BIVARIATE ANALYSIS (Price vs Other Features)
# ============================================================================

print("\n" + "="*70)
print("STEP 2: BIVARIATE ANALYSIS (Price vs Other Features)")
print("="*70)

# -------------------- PRICE VS NUMERICAL FEATURES --------------------
print("\n--- PRICE VS NUMERICAL FEATURES ---")

# Correlation with price
numeric_cols = ['price', 'Property Size', 'Bedroom', 'Bathroom', 
                '# of Floors', 'Total Units', 'Parking Lot', 'Completion Year']
available_numeric = [col for col in numeric_cols if col in df.columns]
correlations = df[available_numeric].corr()['price'].sort_values(ascending=False)
print("\nCorrelation with Price:")
print(correlations)

# Scatter plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

scatter_features = ['Property Size', 'Bedroom', 'Bathroom', 
                    'Completion Year', '# of Floors', 'Total Units']

for idx, col in enumerate(scatter_features):
    if col in df.columns and idx < 6:
        sns.scatterplot(x=df[col], y=df['price'], ax=axes[idx], alpha=0.6, color='darkblue')
        axes[idx].set_title(f'Price vs {col}', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Price (RM)')
        
        # Add trend line
        valid_data = df[[col, 'price']].dropna()
        if len(valid_data) > 1:
            z = np.polyfit(valid_data[col], valid_data['price'], 1)
            p = np.poly1d(z)
            axes[idx].plot(valid_data[col].sort_values(), 
                          p(valid_data[col].sort_values()), 
                          "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.savefig('reports/figures/03_price_vs_numerical.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/figures/03_price_vs_numerical.png")
plt.close()

# ============================================================================
# SAVE DETAILED BIVARIATE ANALYSIS TO CSV
# ============================================================================

print("\n--- SAVING DETAILED BIVARIATE ANALYSIS ---")

# 1. Detailed Correlation Analysis with Interpretations
correlation_details = []
for feature in correlations.index[1:]:  # Skip 'price' itself
    corr_val = correlations[feature]
    
    # Interpretation
    if abs(corr_val) > 0.7:
        strength = "Very Strong"
    elif abs(corr_val) > 0.5:
        strength = "Strong"
    elif abs(corr_val) > 0.3:
        strength = "Moderate"
    elif abs(corr_val) > 0.1:
        strength = "Weak"
    else:
        strength = "Very Weak/None"
    
    direction = "Positive" if corr_val > 0 else "Negative"
    
    if abs(corr_val) > 0.5:
        ml_importance = "High - Important predictor"
    elif abs(corr_val) > 0.3:
        ml_importance = "Medium - Useful predictor"
    elif abs(corr_val) > 0.1:
        ml_importance = "Low - Minor predictor"
    else:
        ml_importance = "Negligible - Consider removing"
    
    correlation_details.append({
        'Feature': feature,
        'Correlation': corr_val,
        'Abs_Correlation': abs(corr_val),
        'Direction': direction,
        'Strength': strength,
        'Interpretation': f"{strength} {direction.lower()} relationship with price",
        'ML_Importance': ml_importance,
        'R_Squared': corr_val ** 2,
        'Variance_Explained_Pct': (corr_val ** 2) * 100
    })

corr_details_df = pd.DataFrame(correlation_details).sort_values('Abs_Correlation', ascending=False)
corr_details_df.to_csv('data/processed/analysis_correlations_detailed.csv', index=False)
print("✓ Saved: data/processed/analysis_correlations_detailed.csv")

# -------------------- PRICE VS CATEGORICAL FEATURES --------------------
print("\n--- PRICE VS CATEGORICAL FEATURES ---")

categorical_features = ['Tenure Type', 'Property Type', 'Category', 'Land Title']

categorical_analysis_list = []
for col in categorical_features:
    if col in df.columns:
        print(f"\n{col}:")
        grouped = df.groupby(col)['price'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).sort_values('median', ascending=False)
        print(grouped)
        
        # Save detailed analysis for each category
        for category in grouped.index:
            category_data = df[df[col] == category]['price']
            
            categorical_analysis_list.append({
                'Feature': col,
                'Category': category,
                'Count': grouped.loc[category, 'count'],
                'Percentage': (grouped.loc[category, 'count'] / len(df)) * 100,
                'Mean_Price': grouped.loc[category, 'mean'],
                'Median_Price': grouped.loc[category, 'median'],
                'Std_Price': grouped.loc[category, 'std'],
                'Min_Price': grouped.loc[category, 'min'],
                'Max_Price': grouped.loc[category, 'max'],
                'Price_Range': grouped.loc[category, 'max'] - grouped.loc[category, 'min'],
                'Diff_from_Overall_Median': grouped.loc[category, 'median'] - df['price'].median(),
                'Diff_from_Overall_Median_Pct': ((grouped.loc[category, 'median'] - df['price'].median()) / df['price'].median()) * 100
            })

categorical_analysis_df = pd.DataFrame(categorical_analysis_list)
categorical_analysis_df.to_csv('data/processed/analysis_categorical_features_detailed.csv', index=False)
print("\n✓ Saved: data/processed/analysis_categorical_features_detailed.csv")

# Box plots for categorical features
available_cats = [col for col in categorical_features if col in df.columns]
if available_cats:
    n_cats = len(available_cats)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, col in enumerate(available_cats[:4]):
        sns.boxplot(x=col, y='price', data=df, ax=axes[idx], palette='Set2')
        axes[idx].set_title(f'Price Distribution by {col}', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Price (RM)')
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('reports/figures/04_price_vs_categorical.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: reports/figures/04_price_vs_categorical.png")
    plt.close()

# Save detailed categorical analysis to CSV
print("\n--- SAVING CATEGORICAL ANALYSIS ---")

categorical_analysis_list = []
for col in categorical_features:
    if col in df.columns:
        grouped = df.groupby(col)['price'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).sort_values('median', ascending=False)
        
        # Save detailed analysis for each category
        for category in grouped.index:
            category_data = df[df[col] == category]['price']
            
            # Calculate additional metrics
            count = grouped.loc[category, 'count']
            percentage = (count / len(df)) * 100
            median_price = grouped.loc[category, 'median']
            diff_from_overall = median_price - df['price'].median()
            diff_pct = (diff_from_overall / df['price'].median()) * 100
            
            # Interpretation
            if diff_pct > 20:
                interpretation = "Premium category - significantly above market median"
            elif diff_pct > 10:
                interpretation = "Above average category - moderately above market median"
            elif diff_pct > -10:
                interpretation = "Average category - close to market median"
            elif diff_pct > -20:
                interpretation = "Below average category - moderately below market median"
            else:
                interpretation = "Budget category - significantly below market median"
            
            categorical_analysis_list.append({
                'Feature': col,
                'Category': category,
                'Count': count,
                'Percentage': percentage,
                'Mean_Price': grouped.loc[category, 'mean'],
                'Median_Price': median_price,
                'Std_Price': grouped.loc[category, 'std'],
                'Min_Price': grouped.loc[category, 'min'],
                'Max_Price': grouped.loc[category, 'max'],
                'Price_Range': grouped.loc[category, 'max'] - grouped.loc[category, 'min'],
                'Diff_from_Overall_Median': diff_from_overall,
                'Diff_from_Overall_Median_Pct': diff_pct,
                'Interpretation': interpretation
            })

categorical_analysis_df = pd.DataFrame(categorical_analysis_list)
categorical_analysis_df.to_csv('data/processed/analysis_categorical_features_detailed.csv', index=False)
print("✓ Saved: data/processed/analysis_categorical_features_detailed.csv")

# -------------------- PRICE VS BINARY AMENITIES --------------------
print("\n--- PRICE VS BINARY AMENITIES ---")

# Check both binary columns (0/1) and text columns (empty vs not empty)
binary_features_01 = ['Nearby School', 'Mall', 'Park', 'Hospital', 
                      'Highway', 'Nearby Railway Station']
text_features = ['Bus Stop', 'School', 'Railway Station']

amenity_impact = []

# Process binary (0/1) features
for col in binary_features_01:
    if col in df.columns:
        # Check if column is actually binary (has 0 and 1)
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) > 0 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            has_ones = (df[col] == 1).sum()
            has_zeros = (df[col] == 0).sum()
            
            if has_ones > 0 and has_zeros > 0:  # Only analyze if we have both
                with_amenity = df[df[col] == 1]['price'].median()
                without_amenity = df[df[col] == 0]['price'].median()
                difference = with_amenity - without_amenity
                pct_difference = (difference / without_amenity) * 100 if without_amenity > 0 else 0
                
                amenity_impact.append({
                    'Amenity': col,
                    'With (Median)': with_amenity,
                    'Without (Median)': without_amenity,
                    'Difference (RM)': difference,
                    'Difference (%)': pct_difference,
                    'Count With': has_ones,
                    'Count Without': has_zeros
                })
                
                print(f"\n{col}:")
                print(f"  With amenity:    RM {with_amenity:,.2f} (n={has_ones})")
                print(f"  Without amenity: RM {without_amenity:,.2f} (n={has_zeros})")
                print(f"  Difference:      RM {difference:,.2f} ({pct_difference:.1f}%)")
            else:
                print(f"\n{col}: Skipped (all 0s or all 1s)")

# Process text features (not empty = has amenity)
for col in text_features:
    if col in df.columns:
        # Create temporary binary: 1 if not empty/NaN, 0 if empty/NaN
        df[f'{col}_binary'] = df[col].notna() & (df[col] != '') & (df[col] != '-')
        
        has_amenity_count = df[f'{col}_binary'].sum()
        no_amenity_count = (~df[f'{col}_binary']).sum()
        
        if has_amenity_count > 0 and no_amenity_count > 0:  # Only analyze if we have both
            with_amenity = df[df[f'{col}_binary']]['price'].median()
            without_amenity = df[~df[f'{col}_binary']]['price'].median()
            difference = with_amenity - without_amenity
            pct_difference = (difference / without_amenity) * 100 if without_amenity > 0 else 0
            
            amenity_impact.append({
                'Amenity': col,
                'With (Median)': with_amenity,
                'Without (Median)': without_amenity,
                'Difference (RM)': difference,
                'Difference (%)': pct_difference,
                'Count With': has_amenity_count,
                'Count Without': no_amenity_count
            })
            
            print(f"\n{col}:")
            print(f"  With amenity:    RM {with_amenity:,.2f} (n={has_amenity_count})")
            print(f"  Without amenity: RM {without_amenity:,.2f} (n={no_amenity_count})")
            print(f"  Difference:      RM {difference:,.2f} ({pct_difference:.1f}%)")
        else:
            print(f"\n{col}: Skipped (all have or all don't have)")
        
        # Drop temporary column
        df.drop(f'{col}_binary', axis=1, inplace=True)

amenity_df = pd.DataFrame(amenity_impact).sort_values('Difference (%)', ascending=False)
print("\n→ Amenities Ranked by Price Impact:")
print(amenity_df[['Amenity', 'Difference (%)']].to_string(index=False))

# -------------------- NUMBER OF FACILITIES VS PRICE --------------------
print("\n--- NUMBER OF FACILITIES VS PRICE ---")

if 'num_facilities' in df.columns:
    # Group by number of facilities
    facilities_analysis = df.groupby('num_facilities')['price'].agg([
        'count', 'mean', 'median', 'std'
    ]).sort_index()
    
    print("\nPrice Statistics by Number of Facilities:")
    print(facilities_analysis)
    
    # Calculate correlation
    facilities_corr = df[['num_facilities', 'price']].corr().iloc[0, 1]
    print(f"\nCorrelation between num_facilities and price: {facilities_corr:.3f}")
    
    if facilities_corr > 0.3:
        print("→ Interpretation: MORE facilities → HIGHER price (moderate to strong relationship)")
    elif facilities_corr > 0.1:
        print("→ Interpretation: MORE facilities → HIGHER price (weak relationship)")
    elif facilities_corr < -0.1:
        print("→ Interpretation: MORE facilities → LOWER price (unexpected!)")
    else:
        print("→ Interpretation: Number of facilities has minimal impact on price")
    
    # Show price increase per additional facility
    avg_price_by_facilities = df.groupby('num_facilities')['price'].mean().sort_index()
    if len(avg_price_by_facilities) > 1:
        price_increase = avg_price_by_facilities.diff().mean()
        print(f"\nAverage price increase per additional facility: RM {price_increase:,.2f}")
    
    # Create visualization: num_facilities vs price
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot with trend line
    sns.scatterplot(x=df['num_facilities'], y=df['price'], ax=axes[0], 
                    alpha=0.6, color='purple', s=50)
    axes[0].set_title('Price vs Number of Facilities', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Facilities')
    axes[0].set_ylabel('Price (RM)')
    
    # Add trend line
    valid_data = df[['num_facilities', 'price']].dropna()
    if len(valid_data) > 1:
        z = np.polyfit(valid_data['num_facilities'], valid_data['price'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(valid_data['num_facilities'].min(), 
                             valid_data['num_facilities'].max(), 100)
        axes[0].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
                    label=f'Trend (r={facilities_corr:.2f})')
        axes[0].legend()
    
    # Box plot by facility count
    # Group facilities into ranges for better visualization
    df['facilities_group'] = pd.cut(df['num_facilities'], 
                                    bins=[-1, 0, 3, 6, 10, 100],
                                    labels=['0', '1-3', '4-6', '7-10', '10+'])
    
    sns.boxplot(x='facilities_group', y='price', data=df, ax=axes[1], 
                palette='viridis')
    axes[1].set_title('Price Distribution by Facility Count', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Facilities')
    axes[1].set_ylabel('Price (RM)')
    
    plt.tight_layout()
    plt.savefig('reports/figures/05b_facilities_vs_price.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: reports/figures/05b_facilities_vs_price.png")
    plt.close()
    
    # Clean up temporary column
    df.drop('facilities_group', axis=1, inplace=True, errors='ignore')
    
    # Save detailed facilities analysis to CSV
    print("\n--- SAVING FACILITIES ANALYSIS ---")
    
    facilities_detailed = []
    for facility_count in facilities_analysis.index:
        count = facilities_analysis.loc[facility_count, 'count']
        mean_price = facilities_analysis.loc[facility_count, 'mean']
        median_price = facilities_analysis.loc[facility_count, 'median']
        
        # Calculate difference from overall median
        diff_from_overall = median_price - df['price'].median()
        diff_pct = (diff_from_overall / df['price'].median()) * 100
        
        # Interpretation
        if diff_pct > 15:
            interpretation = "High facility count - premium pricing"
        elif diff_pct > 5:
            interpretation = "Above average facility count - moderately higher pricing"
        elif diff_pct > -5:
            interpretation = "Average facility count - typical pricing"
        else:
            interpretation = "Low facility count - below average pricing"
        
        facilities_detailed.append({
            'Num_Facilities': facility_count,
            'Property_Count': count,
            'Percentage': (count / len(df)) * 100,
            'Mean_Price': mean_price,
            'Median_Price': median_price,
            'Std_Price': facilities_analysis.loc[facility_count, 'std'],
            'Diff_from_Overall_Median': diff_from_overall,
            'Diff_from_Overall_Median_Pct': diff_pct,
            'Interpretation': interpretation
        })
    
    facilities_detailed_df = pd.DataFrame(facilities_detailed)
    
    # Add correlation info to a summary row
    facilities_summary = {
        'Analysis': 'Facilities vs Price Correlation',
        'Correlation': facilities_corr,
        'Strength': 'Strong' if abs(facilities_corr) > 0.5 else ('Moderate' if abs(facilities_corr) > 0.3 else 'Weak'),
        'Interpretation': 'MORE facilities → HIGHER price' if facilities_corr > 0.3 else ('MORE facilities → HIGHER price (weak)' if facilities_corr > 0.1 else 'Minimal impact'),
        'Avg_Price_Increase_Per_Facility': price_increase if len(avg_price_by_facilities) > 1 else 0,
        'ML_Recommendation': 'Important feature - include in model' if abs(facilities_corr) > 0.3 else 'Moderate feature - consider including'
    }
    
    facilities_detailed_df.to_csv('data/processed/analysis_facilities_detailed.csv', index=False)
    print("✓ Saved: data/processed/analysis_facilities_detailed.csv")
    
    # Save summary
    facilities_summary_df = pd.DataFrame([facilities_summary])
    facilities_summary_df.to_csv('data/processed/analysis_facilities_summary.csv', index=False)
    print("✓ Saved: data/processed/analysis_facilities_summary.csv")
    
else:
    print("\nnum_facilities column not found - skipping analysis")

# Visualization
if len(amenity_df) > 0:
    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(amenity_df))
    colors = ['green' if x > 0 else 'red' for x in amenity_df['Difference (%)']]
    bars = ax.bar(x_pos, amenity_df['Difference (%)'], color=colors, alpha=0.7)
    ax.set_xlabel('Amenity', fontweight='bold')
    ax.set_ylabel('Price Difference (%)', fontweight='bold')
    ax.set_title('Impact of Amenities on Price', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(amenity_df['Amenity'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/figures/05_amenity_impact.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: reports/figures/05_amenity_impact.png")
    plt.close()

# ============================================================================
# STEP 3: MULTIVARIATE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("STEP 3: MULTIVARIATE ANALYSIS")
print("="*70)

# -------------------- CORRELATION HEATMAP --------------------
print("\n--- CORRELATION HEATMAP ---")

numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = df[numeric_cols_all].corr()

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/06_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/figures/06_correlation_heatmap.png")
plt.close()

# Save detailed multivariate analysis to CSV
print("\n--- SAVING MULTIVARIATE ANALYSIS ---")

# 1. Full correlation matrix
correlation_matrix.to_csv('data/processed/analysis_correlation_matrix_full.csv')
print("✓ Saved: data/processed/analysis_correlation_matrix_full.csv")

# 2. Feature pairs with high correlation (for multicollinearity detection)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        feature1 = correlation_matrix.columns[i]
        feature2 = correlation_matrix.columns[j]
        corr_val = correlation_matrix.iloc[i, j]
        
        # Only save pairs with correlation > 0.5 (or < -0.5)
        if abs(corr_val) > 0.5:
            # Determine if this is multicollinearity concern
            if feature1 != 'price' and feature2 != 'price':
                concern = "High" if abs(corr_val) > 0.8 else "Moderate"
                recommendation = "Consider removing one feature" if abs(corr_val) > 0.8 else "Monitor for multicollinearity"
            else:
                concern = "None - one feature is target variable"
                recommendation = "Keep - useful predictor"
            
            high_corr_pairs.append({
                'Feature_1': feature1,
                'Feature_2': feature2,
                'Correlation': corr_val,
                'Abs_Correlation': abs(corr_val),
                'Strength': 'Very Strong' if abs(corr_val) > 0.8 else 'Strong',
                'Multicollinearity_Concern': concern,
                'ML_Recommendation': recommendation
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Abs_Correlation', ascending=False)
    high_corr_df.to_csv('data/processed/analysis_multicollinearity.csv', index=False)
    print("✓ Saved: data/processed/analysis_multicollinearity.csv")

# 3. Grouped summaries if available
if 'Property Type' in df.columns and 'Tenure Type' in df.columns:
    print("\nSaving grouped analysis...")
    
    grouped_analysis = df.groupby(['Property Type', 'Tenure Type'])['price'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).reset_index()
    
    # Add interpretations
    grouped_analysis['Price_Range'] = grouped_analysis['max'] - grouped_analysis['min']
    grouped_analysis['Diff_from_Overall_Median'] = grouped_analysis['median'] - df['price'].median()
    grouped_analysis['Diff_from_Overall_Median_Pct'] = (grouped_analysis['Diff_from_Overall_Median'] / df['price'].median()) * 100
    
    grouped_analysis = grouped_analysis.sort_values('median', ascending=False)
    grouped_analysis.to_csv('data/processed/analysis_grouped_property_tenure.csv', index=False)
    print("✓ Saved: data/processed/analysis_grouped_property_tenure.csv")

# -------------------- GROUPED SUMMARIES --------------------
print("\n--- GROUPED SUMMARIES ---")

if 'Property Type' in df.columns and 'Tenure Type' in df.columns:
    print("\nMedian Price by Property Type and Tenure Type:")
    grouped = df.groupby(['Property Type', 'Tenure Type'])['price'].agg(['count', 'median']).sort_values('median', ascending=False)
    print(grouped.head(20))

# ============================================================================
# SAVE ANALYSIS RESULTS AS CSV FILES
# ============================================================================

print("\n" + "="*70)
print("SAVING ANALYSIS RESULTS")
print("="*70)

# 1. Summary statistics
summary_stats_dict = {
    'Total Properties': [len(df)],
    'Average Price': [df['price'].mean()],
    'Median Price': [df['price'].median()],
    'Price Std Dev': [df['price'].std()],
    'Min Price': [df['price'].min()],
    'Max Price': [df['price'].max()],
    'Avg Property Size': [df['Property Size'].mean() if 'Property Size' in df.columns else 0],
    'Most Common Bedrooms': [df['Bedroom'].mode().values[0] if 'Bedroom' in df.columns else 0],
    'Most Common Bathrooms': [df['Bathroom'].mode().values[0] if 'Bathroom' in df.columns else 0]
}
summary_stats_df = pd.DataFrame(summary_stats_dict)
summary_stats_df.to_csv('data/processed/summary_statistics.csv', index=False)
print("✓ Saved: data/processed/summary_statistics.csv")

# 2. Amenity impact analysis
if len(amenity_df) > 0:
    amenity_df.to_csv('data/processed/amenity_impact.csv', index=False)
    print("✓ Saved: data/processed/amenity_impact.csv")

# 3. Price correlations
price_corr_df = correlations.reset_index()
price_corr_df.columns = ['Feature', 'Correlation']
price_corr_df.to_csv('data/processed/price_correlations.csv', index=False)
print("✓ Saved: data/processed/price_correlations.csv")

# 4. Numerical features summary
summary_df.to_csv('data/processed/numerical_features_summary.csv', index=False)
print("✓ Saved: data/processed/numerical_features_summary.csv")

# 5. Save cleaned data ready for ML (this is the main output!)
# This CSV contains all cleaned data with no encoding - ready for Part 3
df.to_csv('data/final/house_model_ready.csv', index=False)
print("✓ Saved: data/final/house_model_ready.csv")

# 6. Create comprehensive feature analysis summary for ML
print("\n--- CREATING COMPREHENSIVE ML FEATURE SUMMARY ---")

ml_summary = []

# Numerical features
for col in numeric_cols_all:
    if col in df.columns:
        feature_type = "Target" if col == "price" else "Numerical"
        
        ml_summary.append({
            'Feature_Name': col,
            'Feature_Type': feature_type,
            'Data_Type': 'Continuous',
            'Missing_Count': df[col].isna().sum(),
            'Missing_Pct': (df[col].isna().sum() / len(df)) * 100,
            'Unique_Values': df[col].nunique(),
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Std': df[col].std(),
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Skewness': df[col].skew(),
            'Correlation_with_Price': df[[col, 'price']].corr().iloc[0, 1] if col != 'price' else 1.0,
            'Recommended_for_ML': 'Yes' if abs(df[[col, 'price']].corr().iloc[0, 1] if col != 'price' else 0) > 0.1 else 'Maybe',
            'Preprocessing_Needed': 'Yes - transform' if abs(df[col].skew()) > 1 else 'No',
            'Notes': f"Skewed distribution" if abs(df[col].skew()) > 1 else "Normal distribution"
        })

# Categorical features  
for col in df.select_dtypes(include=['object']).columns:
    unique_count = df[col].nunique()
    
    ml_summary.append({
        'Feature_Name': col,
        'Feature_Type': 'Categorical',
        'Data_Type': 'Nominal' if unique_count > 10 else 'Nominal/Ordinal',
        'Missing_Count': df[col].isna().sum(),
        'Missing_Pct': (df[col].isna().sum() / len(df)) * 100,
        'Unique_Values': unique_count,
        'Mean': np.nan,
        'Median': np.nan,
        'Std': np.nan,
        'Min': np.nan,
        'Max': np.nan,
        'Skewness': np.nan,
        'Correlation_with_Price': np.nan,
        'Recommended_for_ML': 'Yes' if unique_count < 50 else 'Maybe - high cardinality',
        'Preprocessing_Needed': 'Yes - encode',
        'Notes': f"{unique_count} categories - {'One-hot encode' if unique_count < 10 else 'Label encode or target encode'}"
    })

ml_summary_df = pd.DataFrame(ml_summary)
ml_summary_df = ml_summary_df.sort_values('Correlation_with_Price', ascending=False, na_position='last')
ml_summary_df.to_csv('data/processed/analysis_ml_feature_summary.csv', index=False)
print("✓ Saved: data/processed/analysis_ml_feature_summary.csv")

# 7. Create analysis metadata file
metadata = {
    'Analysis_Date': [pd.Timestamp.now()],
    'Total_Records': [len(df)],
    'Total_Features': [len(df.columns)],
    'Numerical_Features': [len(df.select_dtypes(include=[np.number]).columns)],
    'Categorical_Features': [len(df.select_dtypes(include=['object']).columns)],
    'Target_Variable': ['price'],
    'Target_Mean': [df['price'].mean()],
    'Target_Median': [df['price'].median()],
    'Target_Std': [df['price'].std()],
    'Features_High_Correlation': [(abs(correlations) > 0.5).sum() - 1],  # Exclude price itself
    'Features_Medium_Correlation': [((abs(correlations) > 0.3) & (abs(correlations) <= 0.5)).sum()],
    'Features_Low_Correlation': [(abs(correlations) <= 0.3).sum() - 1],
    'Recommended_Features_Count': [ml_summary_df[ml_summary_df['Recommended_for_ML'] == 'Yes'].shape[0]]
}

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv('data/processed/analysis_metadata.csv', index=False)
print("✓ Saved: data/processed/analysis_metadata.csv")

print("\n" + "="*70)
print("✓ PART 2: DESCRIPTIVE ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated Files:")
print("\n  📊 Visualizations (7 files):")
print("    - reports/figures/01_price_univariate.png")
print("    - reports/figures/02_numerical_features.png")
print("    - reports/figures/03_price_vs_numerical.png")
print("    - reports/figures/04_price_vs_categorical.png")
print("    - reports/figures/05_amenity_impact.png")
print("    - reports/figures/05b_facilities_vs_price.png")
print("    - reports/figures/06_correlation_heatmap.png")
print("\n  📄 Basic Summary Files (4 files):")
print("    - data/processed/summary_statistics.csv")
print("    - data/processed/amenity_impact.csv")
print("    - data/processed/price_correlations.csv")
print("    - data/processed/numerical_features_summary.csv")
print("\n  📊 Detailed Analysis for ML (13 files):")
print("    Step 1 - Univariate:")
print("    - data/processed/analysis_price_detailed.csv")
print("    - data/processed/analysis_numerical_features_detailed.csv")
print("    - data/processed/analysis_ml_feature_quality.csv")
print("    Step 2 - Bivariate:")
print("    - data/processed/analysis_correlations_detailed.csv")
print("    - data/processed/analysis_categorical_features_detailed.csv")
print("    - data/processed/analysis_amenity_impact_detailed.csv")
print("    - data/processed/analysis_facilities_detailed.csv")
print("    - data/processed/analysis_facilities_summary.csv")
print("    Step 3 - Multivariate:")
print("    - data/processed/analysis_correlation_matrix_full.csv")
print("    - data/processed/analysis_multicollinearity.csv")
print("    - data/processed/analysis_grouped_property_tenure.csv")
print("    Master Files:")
print("    - data/processed/analysis_ml_feature_summary.csv")
print("    - data/processed/analysis_metadata.csv")
print("\n  🎯 ML-Ready Dataset:")
print("    - data/final/house_model_ready.csv")
print("      (Cleaned data ready for feature engineering & modeling)")
print("\n" + "="*70)
print("WHAT'S IN THE DETAILED CSV FILES:")
print("="*70)
print("✓ Statistical values (mean, median, std, quartiles, etc.)")
print("✓ Distribution characteristics (skewness, kurtosis)")
print("✓ Interpretations (e.g., 'Highly right-skewed distribution')")
print("✓ ML recommendations (e.g., 'Apply log transformation')")
print("✓ Feature importance rankings")
print("✓ Quality assessments (Good/Fair/Poor)")
print("✓ Multicollinearity warnings")
print("✓ Category-level insights")
print("✓ Amenity impact levels (High/Medium/Low) \n")