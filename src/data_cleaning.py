"""
Malaysian Condominium Price Predictive Model
Part 1: Environment Setup & Data Cleaning
"""

# ============================================================================
# SECTION 1: INSTALLATION & IMPORTS
# ============================================================================

# Install required packages (run once in terminal):
# pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ All libraries imported successfully!")

# ============================================================================
# SECTION 2: LOAD DATA
# ============================================================================

def load_data(filepath='data/raw/house.csv'):
    """Load the condominium dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded successfully!")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"✗ Error: {filepath} not found")
        print(f"  Please ensure house.csv is in the data/raw/ folder")
        return None

# Load data
df = load_data('data/raw/house.csv')

# Initial data inspection
if df is not None:
    print("\n" + "="*70)
    print("INITIAL DATA OVERVIEW")
    print("="*70)
    print(df.head())
    print()
    df.info()

# ============================================================================
# SECTION 3: DATA CLEANING PIPELINE
# ============================================================================

def clean_condominium_data(df):
    """
    Complete data cleaning pipeline for Malaysian condominium data
    """
    df_clean = df.copy()
    current_year = 2026
    
    print("\n" + "="*70)
    print("STEP 1: STANDARDIZE NUMERIC FIELDS")
    print("="*70)
    
    # Define numeric columns
    numeric_cols = [
        'price', 'Bedroom', 'Bathroom', 'Property Size',
        'Completion Year', '# of Floors', 'Total Units', 'Parking Lot'
    ]
    
    # Special handling for price column (remove RM, spaces, commas)
    if 'price' in df_clean.columns:
        df_clean['price'] = df_clean['price'].astype(str).str.replace('RM', '', regex=False)
        df_clean['price'] = df_clean['price'].str.replace(',', '', regex=False)
        df_clean['price'] = df_clean['price'].str.replace(' ', '', regex=False)
        df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce')
        print(f"  ✓ price: cleaned and converted to numeric")
    
    # Special handling for Property Size (remove sq.ft., sqft, etc.)
    if 'Property Size' in df_clean.columns:
        df_clean['Property Size'] = df_clean['Property Size'].astype(str).str.replace('sq.ft.', '', regex=False)
        df_clean['Property Size'] = df_clean['Property Size'].str.replace('sqft', '', regex=False)
        df_clean['Property Size'] = df_clean['Property Size'].str.replace('sq ft', '', regex=False)
        df_clean['Property Size'] = df_clean['Property Size'].str.replace(',', '', regex=False)
        df_clean['Property Size'] = df_clean['Property Size'].str.replace(' ', '', regex=False)
        df_clean['Property Size'] = pd.to_numeric(df_clean['Property Size'], errors='coerce')
        print(f"  ✓ Property Size: cleaned and converted to numeric")
    
    # Convert other numeric columns
    for col in numeric_cols:
        if col in df_clean.columns and col not in ['price', 'Property Size']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            print(f"  ✓ {col}: converted to numeric")
    
    # Remove impossible values
    if 'price' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean['price'] > 0]
        print(f"  ✓ Removed {before - len(df_clean)} rows with price ≤ 0")
    
    if 'Property Size' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean['Property Size'] > 0]
        print(f"  ✓ Removed {before - len(df_clean)} rows with size ≤ 0")
    
    if 'Completion Year' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[
            (df_clean['Completion Year'] >= 1900) & 
            (df_clean['Completion Year'] <= current_year + 5)
        ]
        print(f"  ✓ Removed {before - len(df_clean)} rows with invalid years")
    
    # Fill missing values with median
    for col in numeric_cols:
        if col in df_clean.columns:
            missing_count = df_clean[col].isna().sum()
            if missing_count > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"  ✓ {col}: filled {missing_count} missing with median ({median_val:.2f})")
    
    print("\n" + "="*70)
    print("STEP 2: STANDARDIZE BINARY/PROXIMITY INDICATORS")
    print("="*70)
    
    # Define binary columns
    binary_cols = [
        'Nearby School', 'Mall', 'Park', 'Hospital', 
        'Highway', 'Nearby Railway Station'
    ]
    
    for col in binary_cols:
        if col in df_clean.columns:
            # Standardize Yes/No, Y/N to 1/0
            df_clean[col] = df_clean[col].replace({
                'Yes': 1, 'No': 0, 'Y': 1, 'N': 0,
                'yes': 1, 'no': 0, 'y': 1, 'n': 0
            })
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
            print(f"  ✓ {col}: standardized to binary (0/1)")
    
    print("\n" + "="*70)
    print("STEP 3: CLEAN CATEGORICAL FEATURES")
    print("="*70)
    
    # Define categorical columns
    categorical_cols = [
        'Tenure Type', 'Property Type', 'Category', 
        'Land Title', 'Developer'
    ]
    
    for col in categorical_cols:
        if col in df_clean.columns:
            # Clean formatting
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
            # Replace 'Nan' string with actual NaN
            df_clean[col] = df_clean[col].replace('Nan', np.nan)
            unique_count = df_clean[col].nunique()
            print(f"  ✓ {col}: cleaned formatting ({unique_count} unique values)")
    
    print("\n" + "="*70)
    print("STEP 4: PROCESS TEXT COLUMNS")
    print("="*70)
    
    # Extract facility count
    if 'Facilities' in df_clean.columns:
        df_clean['num_facilities'] = df_clean['Facilities'].fillna('').apply(
            lambda x: len(str(x).split(',')) if str(x).strip() else 0
        )
        print(f"  ✓ Created 'num_facilities' from Facilities column")
    
    # Drop long description
    if 'description' in df_clean.columns:
        df_clean.drop('description', axis=1, inplace=True)
        print(f"  ✓ Dropped 'description' column")
    
    print("\n" + "="*70)
    print("STEP 5: DROP NON-INFORMATIVE IDENTIFIERS")
    print("="*70)
    
    # Drop identifier columns
    id_cols = ['Firm Number', 'REN Number', 'Ad List']
    for col in id_cols:
        if col in df_clean.columns:
            df_clean.drop(col, axis=1, inplace=True)
            print(f"  ✓ Dropped '{col}'")
    
    print("\n" + "="*70)
    print("CLEANING SUMMARY")
    print("="*70)
    print(f"  Final shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    print(f"  Rows removed: {len(df) - len(df_clean)}")
    
    return df_clean

# Execute cleaning pipeline
if df is not None:
    df_clean = clean_condominium_data(df)
    
    # Save cleaned data
    df_clean.to_csv('data/processed/house_cleaned.csv', index=False)
    print("\n✓ Cleaned data saved to 'data/processed/house_cleaned.csv'")

# ============================================================================
# SECTION 4: DATA QUALITY REPORT
# ============================================================================

def generate_data_quality_report(df):
    """Generate comprehensive data quality report"""
    
    print("\n" + "="*70)
    print("DATA QUALITY REPORT")
    print("="*70)
    
    # Missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
        'Missing Count', ascending=False
    )
    
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("  ✓ No missing values!")
    
    # Data types
    print("\nData Types:")
    print(df.dtypes.value_counts())
    
    # Numeric summary
    print("\nNumeric Features Summary:")
    numeric_df = df.select_dtypes(include=[np.number])
    print(numeric_df.describe().T)
    
    # Categorical summary
    print("\nCategorical Features Summary:")
    categorical_df = df.select_dtypes(include=['object'])
    for col in categorical_df.columns[:5]:  # Show first 5
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().head())

if df_clean is not None:
    generate_data_quality_report(df_clean)

print("\n" + "="*70)
print("✓ PART 1 DATA CLEANING COMPLETE!")
print("="*70)
