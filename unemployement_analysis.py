import pandas as pd
import matplotlib.pyplot as plt

# load the data set

data = pd.read_csv("India.csv")
data.columns = data.columns.str.strip()
print(data.columns)

# verify

print(data.head())

#check coloumns

print(data.columns)

#check data size

print(data.shape)

#data info

print(data.info())

#checking mising values

print(data.isnull().sum())

print(data.describe())

#finding average unemployement rate

print("Average Unemployment Rate:")
print(data["Estimated Unemployment Rate (%)"].mean())

#finding highest unemployement rate

print("Highest Unemployment Rate:")
print(data["Estimated Unemployment Rate (%)"].max())

#finding lowest unemployement rate

print("Lowest Unemployment Rate:")
print(data["Estimated Unemployment Rate (%)"].min())

print(data.columns.to_list())

#check columns real name

for i, col in enumerate(data.columns):
    print(i, repr(col))

# Average unemployment rate by region

rate_column = data.columns[3]

region_data = data.groupby("Region")[rate_column].mean()

print(region_data)

region_data.plot(kind = "bar", figsize=(12,6))
plt.title("Average Unemployement Rate by Region")
plt.xlabel("Region")
plt.ylabel("Unemployement Rate (%)")

plt.xticks(rotation=90)
plt.tight_layout
plt.show()

top5 = region_data.sort_values(ascending = False).head(5)
print("Top5 region with highest unemployement:")
print(top5)

bottom5 = region_data.sort_values().head(5)
print("Top5 region with lowest unemployement:")
print(bottom5)