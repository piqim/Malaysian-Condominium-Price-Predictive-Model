"""
Malaysian Housing Price Predictive Model
Interactive Streamlit Dashboard - Descriptive Analysis Results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from PIL import Image

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Malaysian Housing Price Analysis",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        text-align: left;
        padding: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD DATA FUNCTIONS
# ============================================================================

@st.cache_data
def load_data():
    """Load all necessary data files"""
    data_dict = {}
    
    # Main dataset
    try:
        data_dict['main'] = pd.read_csv('data/final/house_model_ready.csv')
    except FileNotFoundError:
        st.error("Main dataset not found: data/final/house_model_ready.csv")
        return None
    
    # Load all CSV files with error handling
    files_to_load = {
        'summary': 'data/processed/summary_statistics.csv',
        'price_detailed': 'data/processed/analysis_price_detailed.csv',
        'numerical_detailed': 'data/processed/analysis_numerical_features_detailed.csv',
        'ml_quality': 'data/processed/analysis_ml_feature_quality.csv',
        'correlations': 'data/processed/analysis_correlations_detailed.csv',
        'categorical': 'data/processed/analysis_categorical_features_detailed.csv',
        'amenities': 'data/processed/amenity_impact.csv',
        'facilities_detailed': 'data/processed/analysis_facilities_detailed.csv',
        'facilities_summary': 'data/processed/analysis_facilities_summary.csv',
        'multicollinearity': 'data/processed/analysis_multicollinearity.csv',
        'grouped_analysis': 'data/processed/analysis_grouped_property_tenure.csv',
        'ml_summary': 'data/processed/analysis_ml_feature_summary.csv',
        'metadata': 'data/processed/analysis_metadata.csv',
        'price_correlations': 'data/processed/price_correlations.csv',
        'numerical_summary': 'data/processed/numerical_features_summary.csv'
    }
    
    for key, filepath in files_to_load.items():
        try:
            data_dict[key] = pd.read_csv(filepath)
        except FileNotFoundError:
            st.warning(f"Optional file not found: {filepath}")
            data_dict[key] = None
    
    return data_dict

data = load_data()

if data is None or data['main'] is None:
    st.error("❌ Could not load data files. Please run the analysis first:")
    st.code("python src/descriptive_analysis.py", language="bash")
    st.stop()

df = data['main']

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("# ⏩ Navigation Menu")
st.sidebar.markdown("Use the menu below to explore the analysis results.")
st.sidebar.markdown("---")

st.sidebar.markdown("## 📂 Sections")
page = st.sidebar.radio(
    "Choose a section:",
    [
        "📊 Overview",
        "💰 Price Analysis", 
        "📈 Feature Correlations",
        "📑 Categorical Features",
        "🏪 Amenity & Facilities",
        "📋 Full Reports"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("## ℹ️ About This Dashboard")
st.sidebar.markdown("This interactive dashboard presents the results of a descriptive analysis of Malaysian housing prices. Explore key statistics, visualizations, and detailed reports to understand market trends and price drivers.")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "📊 Overview":
    st.markdown('<h1 class="main-header">🔎 Overview of The Malaysian Housing Market (MHM)</h1>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Metrics
    st.markdown("## 📊 Key Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average Price",
            f"RM {df['price'].mean():,.0f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Median Price",
            f"RM {df['price'].median():,.0f}",
            delta=None
        )
    
    with col3:
        if 'Property Size' in df.columns:
            st.metric(
                "Avg Property Size",
                f"{df['Property Size'].mean():,.0f} sq ft",
                delta=None
            )
    
    with col4:
        if 'Bedroom' in df.columns:
            st.metric(
                "Most Common Bedrooms",
                f"{int(df['Bedroom'].mode().values[0])}",
                delta=None
            )
    
    st.markdown("---")
    
    # Price Distribution Visualization
    st.markdown("## 💰 Price Distribution Overview")
    
    if os.path.exists('reports/figures/01_price_univariate.png'):
        image = Image.open('reports/figures/01_price_univariate.png')
        st.image(image, use_container_width=True)
    
    
    skewness = df['price'].skew()
    col1, col2 = st.columns(2)

    # Key Insights
    with col1:
        st.markdown("#### 💡 Insight(s)")
        st.write(f"➡️ **Prices are heavily right-skewed**: Most housing units cluster in the lower price range, while a small number go very high.This means the market is dominated by affordable-to-mid-range units, with luxury units forming a long tail ")
        st.write(f"➡️ **High Kurtosis Value - Many Outliers - Strong Price Inequality in the Market**: This suggests a segmented market (mass-housing vs premium developments).")
        st.write(f"➡️ **Log Scale of Price Distribution is Roughly Bell-Shaped**: Housing prices grow multiplicatively (location, size, and facilities multiply value)")

    # key variables
    with col2:
        st.markdown("#### ‼️ Key Variable(s)")
        st.write(f"📌 **Price Range**: RM {df['price'].min():,.0f} to RM {df['price'].max():,.0f}")
        st.write(f"📌 **Standard Deviation**: RM {df['price'].std():,.0f}")
        st.write(f"📌 **Skewness**: {skewness:.2f}")
        st.write(f"📌 **Kurtosis**: {df['price'].kurtosis():.2f}")

    # Conclusion
    st.markdown("#### 🎯 Conclusion ➡️ Economic Interpretation")
    st.write(f"Most Malaysians are buying in a relatively narrow low–mid range. Meaning a small high-end segment inflates market statistics. This pattern is typical of urban housing markets with income inequality, speculative investment and premium urban land scarcity (Kuala Lumpur, Penang, Johor Bahru)")

    st.markdown("---")
    
    # Dataset Summary
    if data['metadata'] is not None:
        st.markdown("### 📋 Analysis Summary")
        meta = data['metadata']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total Records**: {int(meta['Total_Records'].values[0]):,}")
            st.write(f"**Total Features**: {int(meta['Total_Features'].values[0])}")
            st.write(f"**Numerical Features**: {int(meta['Numerical_Features'].values[0])}")
            st.write(f"**Categorical Features**: {int(meta['Categorical_Features'].values[0])}")
        
        with col2:
            st.write(f"**Target Variable**: {meta['Target_Variable'].values[0]}")
            st.write(f"**Target Mean**: RM {meta['Target_Mean'].values[0]:,.0f}")
            st.write(f"**Target Median**: RM {meta['Target_Median'].values[0]:,.0f}")
            st.write(f"**Features (High Correlation)**: {int(meta['Features_High_Correlation'].values[0])}")
    
    st.markdown("---")
    
    # Quick Data Preview
    st.markdown("### 📋 Quick Data Preview")
    st.write(f"Preview of up to the first 10 data")
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================================
# PAGE 2: PRICE ANALYSIS
# ============================================================================

elif page == "💰 Price Analysis":
    st.markdown('<h1 class="main-header">💰 Price Analysis of The MHM</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Detailed Price Statistics
    st.markdown("### 🪙 Detailed Price Statistics")
    st.write(f"This table highlights some key interpretation(s) of the metrics in the analysis")
    
    if data['price_detailed'] is not None:
        st.dataframe(data['price_detailed'], use_container_width=True)
        
    
    st.markdown("---")
    
    # Price Distribution Visualizations
    st.markdown("### 📈 A Look Into Price Distribution")
    st.write(f"The histogram and box plot indicate that housing prices in the Malaysian housing market are strongly right-skewed, with most transactions concentrated in a relatively narrow low-to-mid price range and a small number of very high-priced units forming a long upper tail. The mean exceeding the median reflects the disproportionate influence of luxury properties, implying that the average price is not representative of the typical housing unit and that the market is segmented into a dominant mass-market segment and a smaller premium segment. Methodologically, this skewness suggests that median-based measures and log-transformed prices are more appropriate for analysis than raw means. Substantively, the pattern implies that increases in average prices may be driven by high-end developments rather than broad-based affordability deterioration.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = px.histogram(
            df, 
            x='price', 
            nbins=50,
            title='Price Distribution Histogram',
            labels={'price': 'Price (RM)', 'count': 'Frequency'}
        )
        fig.add_vline(x=df['price'].mean(), line_dash="dash", line_color="red", 
                     annotation_text="Mean", annotation_position="top")
        fig.add_vline(x=df['price'].median(), line_dash="dash", line_color="green", 
                     annotation_text="Median", annotation_position="top")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = px.box(
            df, 
            y='price',
            title='Price Box Plot (Outlier Detection)',
            labels={'price': 'Price (RM)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Saved visualizations
    st.markdown("### 📊 Analysis on Price vs Numerical Features")

    # concluding overall interpretation
    st.write("**Overall Interpretation:**")
    st.write("➡️ Structural attributes such as size, bedrooms, and bathrooms show positive relationships with price but do not fully explain its variation.")
    st.write("➡️ Temporal and project-level features reveal a market preference for newer and less dense developments.")
    st.write("➡️ The combined patterns indicate a segmented housing market shaped by both physical attributes and broader urban and socioeconomic factors.")

    if os.path.exists('reports/figures/03_price_vs_numerical.png'):
        image = Image.open('reports/figures/03_price_vs_numerical.png')
        st.image(image, use_container_width=True)

    # Interpretation of the multivariate analysis
    st.write("**Detailed Interpretation:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**1) Price vs Property Size**")
        st.write("➡️ There is a clear positive relationship between property size and price, indicating that larger units generally command higher prices.")
        st.write("➡️ Price dispersion increases for larger properties, suggesting that size interacts with factors such as location and quality.")
        st.write("➡️ Extreme high-price observations likely correspond to large premium or luxury units rather than a purely linear size–price relationship.")
        st.write("**4) Price vs Completion Year**")
        st.write("➡️ Prices show a gradual upward trend for more recently completed developments, suggesting a premium for newer properties.")
        st.write("➡️ Older properties are more concentrated in lower price ranges, consistent with depreciation and outdated building standards.")
        st.write("➡️ High-price observations among newer units indicate strong market preference for modern facilities and contemporary design.")
    with col2:
        st.write("**2) Price vs Bedroom**")
        st.write("➡️ Prices tend to rise with the number of bedrooms, showing that room count is an important determinant of value.")
        st.write("➡️ Substantial variation within each bedroom category suggests that bedroom count alone does not fully explain price differences.")
        st.write("➡️ Units with more bedrooms are more prevalent among high-priced observations, indicating association with upper-market segments.")
        st.write("**5) Price vs Number of Floors**")
        st.write("➡️ The relationship between price and number of floors is weakly positive, indicating that building height alone does not strongly determine unit price.")
        st.write("➡️ Most observations cluster at relatively low floor counts, while taller buildings exhibit wide price variation.")
        st.write("➡️ This suggests that internal unit characteristics and location matter more than total building height.")
    with col3:
        st.write("**3) Price vs Bathroom**")
        st.write("➡️ A positive association exists between the number of bathrooms and price, reflecting higher valuation for increased internal amenities.")
        st.write("➡️ Wide dispersion at each bathroom level implies strong influence from non-structural factors such as location and building quality.")
        st.write("➡️ Units with more bathrooms tend to cluster at higher price levels, signaling a link to luxury or family-oriented properties.")
        st.write("**6) Price vs Total Units**")
        st.write("➡️ A weak negative relationship is observed, with developments containing more units tending to have lower average prices.")
        st.write("➡️ High-density projects are associated with more affordable units, while low-density developments are more often high-priced.")
        st.write("➡️ This reflects a trade-off between scale and exclusivity, where scarcity of units is linked to higher valuation.")

# ============================================================================
# PAGE 3: FEATURE CORRELATIONS
# ============================================================================

elif page == "📈 Feature Correlations":
    st.markdown('<h1 class="main-header">📈 Correlations Analysis of Feature(s) in The MCHM with Price</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Correlation Analysis
    st.markdown("### 🔗 Detailed Correlation Analysis")
    st.write(f"This table highlights some key interpretation(s) of the metrics in the analysis")

    if data['correlations'] is not None:
        # Display correlation table
        st.dataframe(
            data['correlations'].sort_values('Abs_Correlation', ascending=False),
            use_container_width=True
        )

        # Interpretation of the correlation analysis in the table
        st.write("**Interpretation of The Correlation Analysis:**")
        st.write("➡️ Several features show strong positive correlations with price, particularly those related to property size, number of bedrooms and bathrooms, and newer completion years.")
        st.write("➡️ Some features exhibit weak or negligible correlations, suggesting they may not be significant price drivers in the market.")

        st.markdown("---")

        st.markdown("### 📊 Correlation Strength Visualization")
        # Bar chart of correlations
        fig = px.bar(
            data['correlations'].sort_values('Correlation', ascending=False),
            x='Correlation',
            y='Feature',
            orientation='h',
            color='Correlation',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            labels={'Correlation': 'Correlation Coefficient'},
            hover_data=['Strength', 'ML_Importance']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("#### 💡 The Highlighted Predictive Features")
        st.warning("The following features have the strongest correlation with price and are likely to be the most important predictors:")
        top_features = data['correlations'].nlargest(5, 'Abs_Correlation')
        for idx, row in top_features.iterrows():
            st.write(f"**{row['Feature']}**: {row['Interpretation']} (Correlation: {row['Correlation']:.3f})")
    
    st.markdown("---")
    
    # Numerical Features
    st.markdown("### 📊 Numerical Features Distribution")
    
    if os.path.exists('reports/figures/02_numerical_features.png'):
        image = Image.open('reports/figures/02_numerical_features.png')
        st.image(image, use_container_width=True)
    
    if data['numerical_detailed'] is not None:
        with st.expander("📋 View Detailed Numerical Features Statistics"):
            st.dataframe(data['numerical_detailed'], use_container_width=True)
    
    st.write("**Interpretation of Numerical Features Distribution:**")
    st.write("The distribution of numerical features such as property size, number of bedrooms, and completion year shows significant variation, with some features exhibiting skewness or outliers that may influence their relationship with price.")
    st.write("➡️ Overall, the features show a strong positive correlation with price.")
    st.write("➡️ The right skewness in features like property size and number of bedrooms suggests that while most units are smaller and more affordable, there is a tail of larger, more expensive properties that drive up the average values.")
    st.write("➡️ The distribution of completion years indicates a market preference for newer developments, which may command higher prices due to modern amenities and design.")

    st.markdown("---")
    
    # Multicollinearity Check
    st.markdown("### ⚠️ Multicollinearity Detection")
    
    if data['multicollinearity'] is not None and len(data['multicollinearity']) > 0:
        st.warning("**Features with high correlation detected (potential multicollinearity):**")
        
        st.dataframe(data['multicollinearity'], use_container_width=True)
    else:
        st.success("✅ No significant multicollinearity detected!")
    
    # Correlation Heatmap
    st.markdown("### 🔥 Correlation Heatmap")
    
    if os.path.exists('reports/figures/06_correlation_heatmap.png'):
        image = Image.open('reports/figures/06_correlation_heatmap.png')
        st.image(image, use_container_width=True)

    st.write("**Interpretation of Correlation Heatmap and Multicollinearity:**")
    st.write("The correlation heatmap reveals clusters of features that are strongly correlated with each other, indicating potential multicollinearity. For example, features related to property size and number of bedrooms may show high intercorrelation, suggesting they capture similar underlying information about the property. The heatmap also highlights which features have the strongest correlations with price, guiding the focus for further analysis and model development.")
    st.write("➡️ **Strongest Price Drivers**: Property Size (0.62) and Bathroom (0.58) show the strongest positive correlations with price, suggesting these are the most important numerical predictors of property value. Bedroom count has a weaker correlation (0.36), which is interesting since bathrooms appear more valuable than bedrooms.")
    st.write("➡️ **Property Characteristics Cluster**: The features Bedroom, Bathroom, and Property Size form a tight correlation cluster (0.51-0.63 among themselves), which makes intuitive sense - larger properties naturally accommodate more rooms.")
    st.write("➡️ **Parking Premium(s)**: Parking Lot shows a notable correlation with price (0.39) and is positively associated with newer buildings (Completion Year: 0.35) and larger properties. This suggests parking is a valued amenity, particularly in newer developments.")
    st.write("**Counterintuitive Findings**: Total Units shows a slight negative correlation with price (-0.11) and with bedroom/bathroom counts. This suggests that properties in larger complexes may actually be less expensive per unit, possibly due to smaller individual unit sizes or different market segments.")

# ============================================================================
# PAGE 4: CATEGORICAL FEATURES
# ============================================================================

elif page == "📑 Categorical Features":
    st.markdown('<h1 class="main-header">📑 Analyzing Categorical Features of The MCMH</h1>', unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("### 📊 Price by Categories")
    st.write(f"The series of box-plots highlights some key interpretation(s) of the categories in the analysis")

    if os.path.exists('reports/figures/04_price_vs_categorical.png'):
        image = Image.open('reports/figures/04_price_vs_categorical.png')
        st.image(image, use_container_width=True)

    st.write("**Interpretation of Price by Categories:**")
    st.write("➡️ **Property Type**: Condominiums and apartments show similar price distributions, while serviced apartments tend to be more expensive on average, likely due to their premium amenities and target market.")
    st.write("➡️ **Tenure Type**: Freehold properties generally command higher prices than leasehold ones, reflecting the greater long-term value and security associated with freehold ownership.")
    st.write("➡️ **Location**: Properties located in Kuala Lumpur and Penang exhibit higher price distributions compared to those in Johor Bahru, indicating stronger demand and higher land values in these urban centers.")
    st.write("Surprisingly, the land title type (Bumi Lot vs Non-Bumi Lot) does not show a significant difference in price distribution, suggesting that other factors such as location and property type may have a stronger influence on price than land title in the Malaysian housing market.")
    
    st.markdown("---")
    
    # Detailed categorical analysis
    st.markdown("### 📋 Detailed Category Analysis")
    
    if data['categorical'] is not None:
        # Filter selector
        features = data['categorical']['Feature'].unique()
        selected_feature = st.selectbox(
            "Select Feature to Analyze",
            features
        )
        
        filtered_cat = data['categorical'][data['categorical']['Feature'] == selected_feature]
        
        # Display table
        st.dataframe(filtered_cat, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            filtered_cat.sort_values('Median_Price', ascending=False),
            x='Category',
            y='Median_Price',
            title=f'Median Price by {selected_feature}',
            labels={'Median_Price': 'Median Price (RM)'},
            color='Diff_from_Overall_Median_Pct',
            color_continuous_scale='RdYlGn',
            hover_data=['Count', 'Mean_Price', 'Interpretation']
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.markdown(f"### 💡 Insights for {selected_feature}")
        
        highest = filtered_cat.nlargest(1, 'Median_Price').iloc[0]
        lowest = filtered_cat.nsmallest(1, 'Median_Price').iloc[0]
        
        st.write(f"**Highest Price**: {highest['Category']} - RM {highest['Median_Price']:,.0f}")
        st.write(f"  ➡️ {highest['Interpretation']}")
        st.write(f"**Lowest Price**: {lowest['Category']} - RM {lowest['Median_Price']:,.0f}")
        st.write(f"  ➡️ {lowest['Interpretation']}")
    
    st.markdown("---")
    
    # Grouped Analysis
    st.markdown("### 🔀 Combined Feature Analysis")
    st.write(f"This analysis explores the interaction between Property Type and Tenure Type to understand how these combined categories influence price.")
    
    if data['grouped_analysis'] is not None:
        st.markdown("#### Property Type × Tenure Type Analysis")
        st.write("The table below shows the median price for each combination of property type and tenure type, along with the percentage difference from the overall median price. This allows us to identify which combinations are associated with higher or lower prices compared to the market average.")
        st.dataframe(data['grouped_analysis'], use_container_width=True)
        
        # Heatmap
        pivot_data = data['grouped_analysis'].pivot_table(
            values='median',
            index='Property Type',
            columns='Tenure Type',
            aggfunc='first'
        )
        
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Tenure Type", y="Property Type", color="Median Price"),
            title="Price Heatmap: Property Type × Tenure Type",
            color_continuous_scale='YlOrRd',
            aspect="auto"
        )
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, use_container_width=True)

        st.write("**Insights from Combined Feature Analysis:**")
        st.write("➡️ **Serviced Apartments with Freehold Tenure**: This combination commands the highest median price, indicating a strong market preference for premium properties that offer both high-end amenities and long-term ownership security.")
        st.write("➡️ **Housing with Leasehold Tenure**: This combination has the lowest median price, suggesting that the lack of ownership security and potentially fewer amenities make these properties less desirable in the market.")
        st.write("➡️ **Overall Trend**: Freehold properties consistently show higher median prices across all property types, reinforcing the importance of tenure security in driving property values. Additionally, serviced apartments tend to be more expensive than condominiums and apartments, regardless of tenure, highlighting the value placed on premium amenities and services in the Malaysian housing market.")

# ============================================================================
# PAGE 5: AMENITY & FACILITIES
# ============================================================================

elif page == "🏪 Amenity & Facilities":
    st.markdown('<h1 class="main-header">🏪 Impact of Type of Amenities & Number of Facilities on Price</h1>', unsafe_allow_html=True)
    
    st.markdown("---")

    # Amenity Impact
    st.markdown("### 📊 Amenity Impact on Price")
    st.write(f"This analysis evaluates how the presence of specific amenities influences housing prices compared to the overall market median price.")
    
    if data['amenities'] is not None and len(data['amenities']) > 0:
        # Display amenity table
        st.dataframe(data['amenities'], use_container_width=True)
        
        # Visualization
        fig = px.bar(
            data['amenities'].sort_values('Difference (%)', ascending=False),
            x='Amenity',
            y='Difference (%)',
            title='Price Impact of Amenities (%)',
            labels={'Difference (%)': 'Price Difference (%)'},
            color='Difference (%)',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.markdown("#### 💡 Key Takeaways from Amenities vs. Price")
        
        positive = data['amenities'][data['amenities']['Difference (%)'] > 0].sort_values('Difference (%)', ascending=False)
        if len(positive) > 0:
            st.write("**Most Valuable Amenities:**")
            for idx, row in positive.head(3).iterrows():
                st.write(f"➡️ **{row['Amenity']}**: +{row['Difference (%)']:.1f}% price increase")
        
        negative = data['amenities'][data['amenities']['Difference (%)'] < 0].sort_values('Difference (%)')
        if len(negative) > 0:
            st.write("The findings are quite counterintuitive, as some amenities are associated with lower prices. This could be due to various factors such as the quality of the amenity, its maintenance, or its relevance to buyers. For example, an amenity that is poorly maintained or not highly valued by buyers may actually detract from the property's appeal, leading to a price decrease.")
            st.write("\n**Amenities Associated with Lower Prices:**")
            for idx, row in negative.head(3).iterrows():
                st.write(f"➡️ **{row['Amenity']}**: {row['Difference (%)']:.1f}% price decrease")
            
            # more takeaways
            st.write("This suggests proximity to public amenities is NOT a universal value-add in this market. Instead, this appears to be a car-centric, suburban-preferring market where buyers are willing to pay premium for:")
            st.write("➡️ Distance from urban congestionation and noise")
            st.write("➡️ Larger property size and more internal amenities (bedrooms, bathrooms)")
            st.write("➡️ Newer developments with modern design and facilities in less-dense area")
            st.write("➡️ Privacy and exclusivity (fewer total units, freehold tenure)")
            st.write("***Which is the case in most urban-centers in Malaysia, such as KL, Johor Bahru and Penang.***")
    
    st.markdown("---")
    
    # Facilities Analysis
    st.markdown("### 🏗️ Analysis on Number of Facilities vs. Price")
    
    if data['facilities_summary'] is not None:
        # Summary
        summary = data['facilities_summary'].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correlation", f"{summary['Correlation']:.3f}")
        with col2:
            st.metric("Strength", summary['Strength'])
        with col3:
            st.metric("Avg Price Increase", f"RM {summary['Avg_Price_Increase_Per_Facility']:,.0f}")
        
        st.warning(f"**Interpretation**: {summary['Interpretation']}")
    
    if data['facilities_detailed'] is not None:
        st.markdown("#### Detailed Facilities Breakdown")
        st.write("The table below provides a detailed breakdown of how the number of facilities in a housing development impacts its price compared to the overall market median. It shows the average price increase associated with each additional facility, as well as the percentage difference from the overall median price.")
        st.dataframe(data['facilities_detailed'], use_container_width=True)
    
    # Facilities visualization
    if os.path.exists('reports/figures/05b_facilities_vs_price.png'):
        st.markdown("#### 📊 Number of Facilities vs Price Visualization")
        image = Image.open('reports/figures/05b_facilities_vs_price.png')
        st.image(image, use_container_width=True)

        st.write("**Interpretation of Facilities vs Price Visualization:**")
        st.write("The visualization illustrates a positive relationship between the number of facilities and housing prices, indicating that properties with more facilities tend to command higher prices. However, the relationship is not perfectly linear, suggesting that while additional facilities generally add value, the specific types of facilities and their quality may also play a significant role in determining price. The presence of outliers with high prices despite fewer facilities suggests that other factors such as location, property size, and building quality can also significantly influence price, sometimes outweighing the impact of facilities alone.")

# ============================================================================
# PAGE 6: FULL REPORTS
# ============================================================================

elif page == "📋 Full Reports":
    st.markdown('<h1 class="main-header">📋 Comprehensive Analysis Reports</h1>', unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("### 📊 Available Reports")
        
    # Report selector
    report_options = {
        "Price Detailed Analysis": 'price_detailed',
        "Numerical Features Analysis": 'numerical_detailed',
        "Feature Quality Assessment": 'ml_quality',
        "Correlation Analysis": 'correlations',
        "Categorical Features Analysis": 'categorical',
        "Amenity Impact Analysis": 'amenities',
        "Facilities Detailed Analysis": 'facilities_detailed',
        "Facilities Summary": 'facilities_summary',
        "Multicollinearity Detection": 'multicollinearity',
        "Grouped Analysis (Property × Tenure)": 'grouped_analysis',
        "ML Feature Summary": 'ml_summary',
        "Analysis Metadata": 'metadata'
    }
    
    selected_report = st.selectbox(
        "Select a Report to View and Download",
        list(report_options.keys())
    )
    
    # Display selected report
    report_key = report_options[selected_report]

    
    if data[report_key] is not None:
        st.markdown(f"### 📋 {selected_report}")
        st.dataframe(data[report_key], use_container_width=True)
        
        # Download button
        csv = data[report_key].to_csv(index=False)
        st.download_button(
            label=f"📥 Download {selected_report}",
            data=csv,
            file_name=f"{report_key}.csv",
            mime='text/csv'
        )
    else:
        st.warning(f"Report not available: {selected_report}")

    st.write("**Note:** The reports available for download are generated from the analysis and may contain detailed statistics, interpretations, and insights that can be used for further research, presentations, or decision-making related to the Malaysian housing market.")
    

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>Developed and Analyzed By <a href="https://github.com/piqim" target="_blank" style="font-weight: bold; text-decoration: none; color: inherit;">Mustaqim Burhanuddin</a></p>
        <p>Built with Python Streamlit & Pandas</p>
    </div>
    """, unsafe_allow_html=True)