# Real-world Health Data Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("row_dataset.csv")

# Rename columns
df.columns = [
    "Country","Code","Year","Schizophrenia","Depression",
    "Anxiety","Bipolar","Eating_Disorder"
]

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Cleaning
df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

# Correlation
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Depression trend
yearly=df.groupby("Year")["Depression"].mean()
plt.figure(figsize=(12,5))
plt.plot(yearly.index, yearly.values)
plt.title("Average Depression Rate Over Years")
plt.grid(True)
plt.show()

# Anxiety trend
yearly=df.groupby("Year")["Anxiety"].mean()
plt.figure(figsize=(12,5))
plt.plot(yearly.index, yearly.values)
plt.title("Average Anxiety Rate Over Years")
plt.grid(True)
plt.show()

# Top 10 countries
latest=df[df["Year"]==df["Year"].max()]
top=latest.sort_values("Depression",ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(data=top,x="Depression",y="Country")
plt.title("Top 10 Countries with Highest Depression")
plt.show()

# Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Depression"],bins=30,kde=True)
plt.title("Distribution of Depression")
plt.show()

# Scatter
plt.figure(figsize=(8,6))
sns.scatterplot(data=df,x="Anxiety",y="Depression")
plt.title("Anxiety vs Depression")
plt.show()

# Country comparison
countries=["India","United States","China","Brazil"]
subset=df[df["Country"].isin(countries)]
plt.figure(figsize=(12,6))
sns.lineplot(data=subset,x="Year",y="Depression",hue="Country")
plt.title("Depression Trends")
plt.show()

# ML
X=df[["Year","Schizophrenia","Anxiety","Bipolar","Eating_Disorder"]]
y=df["Depression"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
pred=model.predict(X_test)

print("MAE:",mean_absolute_error(y_test,pred))
print("MSE:",mean_squared_error(y_test,pred))
print("RMSE:",np.sqrt(mean_squared_error(y_test,pred)))
print("R2 Score:",r2_score(y_test,pred))

plt.figure(figsize=(8,6))
plt.scatter(y_test,pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Depression")
plt.show()

print("""
Key Findings:
1. Depression and Anxiety are positively correlated.
2. Mental health prevalence varies across countries.
3. Depression trends change over time.
4. Linear Regression can predict Depression using other disorder indicators.
5. The project demonstrates complete EDA and predictive analytics.
""")
