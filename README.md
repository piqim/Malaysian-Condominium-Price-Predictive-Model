# 🏢 Malaysian Condominium Price Predictive Model
Dataset: https://www.kaggle.com/datasets/mcpenguin/raw-malaysian-housing-prices-data/data


Activate Venv:
venv\Scripts\Activate.ps1

# Analysis CSV Files - Reference Guide (After Descriptive Analysis)

## 📂 File Organization

All analysis CSV files are saved in `data/processed/` for easy access during model training.

**Total Files Generated:**
- 7 Visualizations (PNG)
- 4 Basic Summary CSVs
- 13 Detailed Analysis CSVs
- 1 ML-Ready Dataset

---

## 🎯 ML-Ready Dataset

### **house_model_ready.csv** (`data/final/`)
- **Purpose**: Main dataset for machine learning
- **Contents**: All cleaned data, ready for encoding and modeling
- **Use For**: Training predictive models in Part 3
- **Columns**: All original features after cleaning (no encoding applied yet)

---

## 📊 Detailed Analysis Files (for ML Training)

### **1. analysis_price_detailed.csv**
**What it contains:**
- Complete price statistics (mean, median, std, min, max, quartiles, IQR)
- Skewness and kurtosis values
- **Interpretations**: Text explanations of each metric
- Distribution characteristics

**Use for:**
- Understanding target variable distribution
- Deciding if price needs transformation (log/sqrt)
- Setting price prediction ranges

**Key columns:**
- `Metric`: Statistical measure name
- `Value`: Numerical value
- `Interpretation`: Plain English explanation

---

### **2. analysis_numerical_features_detailed.csv**
**What it contains:**
- Comprehensive stats for ALL numerical features
- Distribution metrics (mean, median, mode, std, variance)
- Skewness & kurtosis with interpretations
- Outlier counts and percentages
- Coefficient of variation

**Use for:**
- Feature selection decisions
- Identifying which features need transformation
- Understanding data quality
- Detecting outliers

**Key columns:**
- `Feature`: Feature name
- `Count`, `Missing`, `Missing_Pct`: Data completeness
- `Skewness_Interpretation`: e.g., "Highly right-skewed"
- `Kurtosis_Interpretation`: e.g., "Leptokurtic (heavy tails)"
- `Outliers_Count`, `Outliers_Pct`: Outlier detection results

---

### **3. analysis_ml_feature_quality.csv**
**What it contains:**
- Quality assessment for each feature
- ML recommendations for handling each feature
- Transformation recommendations
- Variance checks

**Use for:**
- **Feature selection** - which features to keep/drop
- **Preprocessing decisions** - which transformations to apply
- **Imputation strategy** - how to handle missing values

**Key columns:**
- `Feature`: Feature name
- `Missing_Pct`: Percentage of missing data
- `Quality_Assessment`: "Good", "Fair", or "Poor"
- `ML_Recommendation`: Specific advice (e.g., "Use robust imputation")
- `Transformation_Recommendation`: "Apply log transformation" or "No transformation needed"
- `Low_Variance_Flag`: Features with very low variance (may not be useful)

---

### **4. analysis_correlations_detailed.csv**
**What it contains:**
- Correlation between each feature and price
- Strength classification (Very Strong, Strong, Moderate, Weak)
- Direction (Positive/Negative)
- ML importance rating
- R² and variance explained

**Use for:**
- **Feature selection** - prioritize high correlation features
- **Multicollinearity detection** - identify redundant features
- **Feature importance ranking**

**Key columns:**
- `Feature`: Feature name
- `Correlation`: Correlation coefficient with price
- `Strength`: "Very Strong", "Strong", "Moderate", "Weak", "Very Weak"
- `Direction`: "Positive" or "Negative"
- `Interpretation`: Plain English explanation
- `ML_Importance`: "High", "Medium", "Low", "Negligible"
- `Variance_Explained_Pct`: How much of price variance this feature explains

---

### **5. analysis_categorical_features_detailed.csv**
**What it contains:**
- Price statistics for each category within each categorical feature
- Count and percentage of each category
- Price differences from overall median

**Use for:**
- Understanding which categories command higher/lower prices
- Encoding decisions (one-hot vs label encoding)
- Feature engineering opportunities

**Key columns:**
- `Feature`: Feature name (e.g., "Tenure Type")
- `Category`: Specific category (e.g., "Freehold")
- `Count`, `Percentage`: How common this category is
- `Mean_Price`, `Median_Price`: Average prices for this category
- `Diff_from_Overall_Median_Pct`: % difference from overall median price

---

### **6. analysis_amenity_impact_detailed.csv**
**What it contains:**
- Price impact of each amenity
- Median prices with vs without amenity
- Percentage difference
- Impact level classification
- ML recommendations

**Use for:**
- Understanding which amenities matter most
- Feature selection for amenity features
- Creating composite amenity scores

**Key columns:**
- `Amenity`: Amenity name
- `With (Median)`, `Without (Median)`: Price comparison
- `Difference (%)`: Percentage price impact
- `Impact_Level`: "High", "Medium", "Low"
- `Recommendation`: ML-specific advice
- `Count_With`, `Count_Without`: Sample sizes

---

### **7. analysis_facilities_detailed.csv**
**What it contains:**
- Price statistics by number of facilities
- How prices change as facility count increases
- Percentage differences from overall median

**Use for:**
- Understanding facility impact on pricing
- Creating facility-based features
- Binning facilities into groups

**Key columns:**
- `Num_Facilities`: Number of facilities (0, 1, 2, 3, etc.)
- `Property_Count`: How many properties have this many facilities
- `Median_Price`: Median price for properties with this facility count
- `Diff_from_Overall_Median_Pct`: % difference from overall median
- `Interpretation`: Plain English explanation

---

### **8. analysis_facilities_summary.csv**
**What it contains:**
- Overall correlation between facilities and price
- Average price increase per additional facility
- ML recommendation for feature importance

**Use for:**
- Quick reference on facility importance
- Deciding if facilities should be a key feature

**Key columns:**
- `Correlation`: Correlation coefficient with price
- `Strength`: "Strong", "Moderate", or "Weak"
- `Avg_Price_Increase_Per_Facility`: RM increase per facility
- `ML_Recommendation`: Specific guidance

---

### **9. analysis_correlation_matrix_full.csv**
**What it contains:**
- Complete correlation matrix of ALL numerical features
- Every feature correlated with every other feature

**Use for:**
- **Multicollinearity detection** - find redundant features
- Understanding feature relationships
- Creating interaction features

---

### **10. analysis_multicollinearity.csv**
**What it contains:**
- Feature pairs with high correlation (>0.5)
- Multicollinearity concern levels
- Recommendations for handling

**Use for:**
- **Critical for ML** - identifies features that might cause problems
- Deciding which features to drop
- Avoiding model instability

**Key columns:**
- `Feature_1`, `Feature_2`: Correlated feature pair
- `Correlation`: How strongly they're correlated
- `Multicollinearity_Concern`: "High", "Moderate", "None"
- `ML_Recommendation`: What to do about it

---

### **11. analysis_grouped_property_tenure.csv**
**What it contains:**
- Price analysis by Property Type AND Tenure Type combinations
- Identifies most/least expensive combinations

**Use for:**
- Understanding interaction effects
- Creating interaction features
- Market segmentation

**Key columns:**
- `Property_Type`: Type of property
- `Tenure_Type`: Tenure classification
- `Median_Price`: Median for this combination
- `Diff_from_Overall_Median_Pct`: How this combo compares to overall

---

### **12. analysis_ml_feature_summary.csv** ⭐ MASTER FILE
**What it contains:**
- **COMPREHENSIVE OVERVIEW** of ALL features
- Feature types, data types, missing values
- Correlation with price
- Preprocessing recommendations
- ML readiness assessment

**Use for:**
- **Master reference** for feature engineering
- **Quick lookup** of feature characteristics
- **Documentation** for your model

**Key columns:**
- `Feature_Name`: Feature name
- `Feature_Type`: "Target", "Numerical", "Categorical"
- `Missing_Pct`: Missing data percentage
- `Correlation_with_Price`: How it relates to price
- `Recommended_for_ML`: "Yes", "No", "Maybe"
- `Preprocessing_Needed`: What to do before modeling
- `Notes`: Specific guidance

---

### **13. analysis_metadata.csv**
**What it contains:**
- Overall dataset summary
- Analysis date
- Feature counts by type
- Correlation distribution
- Recommended features count

**Use for:**
- Quick dataset overview
- Documentation
- Tracking analysis versions

---

## 📋 Summary Analysis Files (Simple - 4 files)

### **summary_statistics.csv**
- Basic overall statistics
- Good for quick reference

### **price_correlations.csv**
- Simple correlation list
- Feature → Correlation value

### **numerical_features_summary.csv**
- Basic stats for numerical features
- Simplified version of detailed analysis

### **amenity_impact.csv**
- Simple amenity impact summary

---

## 🎯 How to Use These Files for ML

### **Phase 1: Feature Selection**
1. Open `analysis_ml_feature_quality.csv`
   - Keep features with `Quality_Assessment` = "Good" or "Fair"
   - Drop features with >50% missing data
   
2. Open `analysis_correlations_detailed.csv`
   - Prioritize features with `ML_Importance` = "High" or "Medium"
   - Consider dropping features with correlation < 0.1

3. Open `analysis_ml_feature_summary.csv`
   - Use features where `Recommended_for_ML` = "Yes"

### **Phase 2: Preprocessing**
1. Open `analysis_ml_feature_quality.csv`
   - Follow `Transformation_Recommendation` for each feature
   - Apply log/sqrt transforms where suggested

2. Open `analysis_multicollinearity.csv` ⭐ NEW
   - Review feature pairs with high correlation
   - Drop one feature from pairs with `Multicollinearity_Concern` = "High"
   - Keep pairs where one feature is 'price' (these are good predictors)

3. Open `analysis_categorical_features_detailed.csv`
   - Decide encoding strategy based on unique value counts
   - <10 categories → One-hot encoding
   - \>10 categories → Label encoding or target encoding

4. Open `analysis_numerical_features_detailed.csv`
   - Handle outliers based on `Outliers_Count`
   - Impute missing values based on distribution

### **Phase 3: Feature Engineering**
1. Open `analysis_amenity_impact_detailed.csv`
   - Keep amenities with `Impact_Level` = "High" or "Medium"
   - Create composite score: sum of high-impact amenities
   
2. Open `analysis_facilities_summary.csv` ⭐ NEW
   - If correlation is strong (>0.3), keep as important feature
   - Consider creating facility bins (0, 1-3, 4-6, 7-10, 10+)

3. Open `analysis_correlations_detailed.csv`
   - Create interaction features between high-correlation features
   - Example: Property_Size × Bedrooms

4. Open `analysis_grouped_property_tenure.csv` ⭐ NEW
   - Create interaction features: Property_Type × Tenure_Type
   - Or use this for market segmentation

5. Open `analysis_price_detailed.csv`
   - Consider binning price into categories if needed
   - Apply transformation if skewness is high

---

## 💡 Pro Tips

1. **Start with `analysis_ml_feature_summary.csv`** - It's your master reference
2. **Use detailed files for deep dives** - When you need to understand a specific feature
3. **Cross-reference files** - Combine insights from multiple files
4. **Document decisions** - Note which features you keep/drop and why
5. **Version control** - These files track your analysis decisions

---

## 🚀 Quick Checklist for ML Prep

- [ ] Review `analysis_ml_feature_summary.csv` - understand all features
- [ ] Check `analysis_ml_feature_quality.csv` - decide which features to keep
- [ ] Review `analysis_correlations_detailed.csv` - prioritize important features
- [ ] **Check `analysis_multicollinearity.csv`** - identify and remove redundant features
- [ ] Check `analysis_price_detailed.csv` - decide if target needs transformation
- [ ] Plan encoding strategy from `analysis_categorical_features_detailed.csv`
- [ ] Review `analysis_facilities_summary.csv` - assess facility importance
- [ ] Review `analysis_amenity_impact_detailed.csv` - select high-impact amenities
- [ ] Check `analysis_grouped_property_tenure.csv` - consider interaction features
- [ ] Use `house_model_ready.csv` for actual model training

---

## 📞 Need Help?

- All interpretations are in plain English
- Recommendations are actionable
- Files are ready to be read by pandas: `pd.read_csv('file.csv')`