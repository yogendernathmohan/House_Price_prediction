import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("House_Price_Prediction.csv")

# Display the first few rows of the data
print(data.head())
print(data.describe())

# Check for outliers
plt.figure(figsize=(10, 6))
plt.scatter(data['Area'], data['Price'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs Price')
plt.show()
