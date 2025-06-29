# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load Data

df = pd.read_csv("data/inventory.csv", parse_dates=['Date'])
df['Inventory'] = df['Stock_Received'] + df['Stock_on_Hand']
# Data Preprocessing
df['Product']=df['Product'].str.strip().str.lower()
df['Product']=df['Product'].replace({
    '???':'soap',
})
df['Product']=df['Product'].fillna('soap')

df['Stock_Received']=df['Stock_Received'].fillna(df['Stock_Received'].mean())
df['Stock_Sold']=df['Stock_Sold'].fillna(df['Stock_Sold'].mean())
df['Stock_on_Hand']=df['Stock_on_Hand'].fillna(df['Stock_on_Hand'].mean())
df['Reorder_Level']=df['Reorder_Level'].fillna(df['Reorder_Level'].mean())
df['Season']=df['Season'].fillna('Rainy')
df['Date']=pd.to_datetime(df['Date'], errors='coerce')
df=df.dropna(subset=['Date'])
df.drop(columns=['Category'], inplace=True)
df=df[df['Stock_Received'].between(20,43)]
df=df[df['Stock_Sold'].between(15,37)]
df=df[df['Stock_on_Hand'].between(0,20)]

# Page Config
st.set_page_config(page_title="Smart Inventory Dashboard", layout="wide")

# --- Sidebar ---
st.sidebar.image("assets/logo.png", width=200)
st.sidebar.title("Smart Inventory Management")


# --- Header ---
st.title("📦 Smart Inventory Dashboard")
st.markdown("Get real-time visibility into stock levels, reorder alerts, and product trends.")

# --- KPI Cards ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📦 Total Stock Received", int(df['Stock_Received'].sum()))

with col2:
    st.metric("🛒 Total Stock Sold", int(df['Stock_Sold'].sum()))

with col3:
    st.metric("📍 Active Products", df['Product'].nunique())


# 📆 Prepare daily data
daily_df = df.groupby('Date').agg({
    'Inventory': 'sum',
    'Stock_Sold': 'sum'
}).reset_index()

# 🔄 Melt for dual-line Plot
melted_df = daily_df.melt(id_vars='Date',
                          value_vars=['Inventory', 'Stock_Sold'],
                          var_name='Metric',
                          value_name='Units')

# 📈 Static Line Chart (No Animation)
fig_static = px.line(
    melted_df,
    x="Date",
    y="Units",
    color="Metric",
    title="📊 Daily Inventory Level vs Sales Over Time",
    markers=True,
    template="plotly_dark",
    line_shape="spline",
    color_discrete_map={
        "Inventory": "#00CC96",     # Teal
        "Stock_Sold": "#EF553B"     # Red-orange
    }
)

fig_static.update_layout(
    xaxis_title="Date",
    yaxis_title="Units",
    hovermode="x unified",
    legend=dict(orientation="h", x=0.5, xanchor="center"),
    margin=dict(t=50, b=40, l=10, r=10)
)

st.plotly_chart(fig_static, use_container_width=True)



# 📆 Create Month column and group by
df['Month'] = df['Date'].dt.to_period('M')
monthly_df = df.groupby('Month').agg({
    'Inventory': 'sum',      # Assuming this is Inventory level
    'Stock_Sold': 'sum'
}).reset_index()

monthly_df['Month'] = monthly_df['Month'].dt.to_timestamp()

# 📈 Melt for multi-line Plotly chart
monthly_melted = monthly_df.melt(id_vars='Month', 
                                 var_name='Metric', value_name='Units')

# 🎨 Plot with Plotly
fig = px.line(monthly_melted, x='Month', y='Units', color='Metric',
              markers=True, line_shape="spline",  # Smooth lines
              color_discrete_map={
                  'Stock_Sold': '#EF553B'      # Red
              },
              title='📈 Monthly Inventory Level vs Sales')

fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Units',
    hovermode='x unified',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)


top_products = df.groupby("Product")["Stock_Sold"].sum().sort_values(ascending=False).head(10)
fig2 = px.bar(top_products, x=top_products.index, y=top_products.values,
              title="Top 10 Selling Products",
              color=top_products.values,
              color_continuous_scale='Blues')
st.plotly_chart(fig2, use_container_width=True)

# 📌 Filter by Product (including "All Products" option)
# 👇 Dropdown filter
product_list = ["All Products"] + sorted(df["Product"].dropna().unique().tolist())
selected_product = st.selectbox("🔍 Select a Product to Compare Seasonal Inventory vs Sales", product_list)

# 🔄 Filter based on product
if selected_product != "All Products":
    filtered_df = df[df["Product"] == selected_product]
else:
    filtered_df = df.copy()

# 📊 Group by Season with both Inventory and Sales
seasonal_df = filtered_df.groupby("Season").agg({
    'Inventory': 'sum',  # Assuming 'Inventory' = Inventory
    'Stock_Sold': 'sum'
}).reset_index()

# 🧹 Melt for grouped bar chart
seasonal_melted = seasonal_df.melt(id_vars="Season",
                                   value_vars=["Inventory", "Stock_Sold"],
                                   var_name="Metric", value_name="Units")

# 📈 Plotly grouped bar chart
fig4 = px.bar(seasonal_melted, x="Season", y="Units", color="Metric", barmode="group",
              title=f"📊 Inventory vs Sales by Season for {'All Products' if selected_product == 'All Products' else selected_product}",
              color_discrete_map={
                  'Inventory': '#00BFFF',  # Light blue
                  'Stock_Sold': '#FF6347'      # Tomato red
              },
              text_auto=True)

fig4.update_layout(xaxis_title="Season", yaxis_title="Total Units", template="plotly_white")

st.plotly_chart(fig4, use_container_width=True)



animated = df.groupby(["Date", "Product"])["Stock_Sold"].sum().reset_index()
fig_anim = px.bar(animated, x="Product", y="Stock_Sold", animation_frame="Date",
                  title="Animated Sales Over Time",
                  color="Stock_Sold", color_continuous_scale="Turbo")
st.plotly_chart(fig_anim, use_container_width=True)


# 📌 Step 1: Aggregate monthly sales by product and season
monthly_agg = df.groupby(["Product", "Season", "Month"])["Stock_Sold"].sum().reset_index()

# 📌 Step 2: Encode categorical variables
le_product = LabelEncoder()
le_season = LabelEncoder()

monthly_agg["Product_encoded"] = le_product.fit_transform(monthly_agg["Product"])
monthly_agg["Season_encoded"] = le_season.fit_transform(monthly_agg["Season"])

# 📌 Step 3: Define features (X) and target (y)
X = monthly_agg[["Product_encoded", "Season_encoded"]]
y = monthly_agg["Stock_Sold"]

# 📌 Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 5: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Step 6: Evaluate model performance
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Monthly MAE:", mae)

# 📌 Step 7: Prediction function using encoded inputs
def predict_stock(product_name, season):
    product_code = le_product.transform([product_name])[0]
    season_code = le_season.transform([season])[0]
    return model.predict([[product_code, season_code]])[0]

st.markdown("## 🔮 Predict Stock to Order")

# 📋 Dropdowns for user input
selected_product = st.selectbox("Select Product", sorted(df["Product"].dropna().unique()))
selected_season = st.selectbox("Select Season", sorted(df["Season"].dropna().unique()))

# 🧮 Predict when button is clicked
if st.button("Predict Stock to Order"):
    try:
        predicted_stock = predict_stock(selected_product, selected_season)
        st.success(f"✅ Predicted stock to order for **{selected_product}** in **{selected_season}** season is: **{int(predicted_stock)} units**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
