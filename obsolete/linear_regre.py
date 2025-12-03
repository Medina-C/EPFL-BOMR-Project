import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- Parameters ---
csv_file = 'outputs/speed_to_mtr_input.csv'        # Path to your CSV file
x_min = 75                   # Minimum x-value for filtering
x_max = 450                   # Maximum x-value for filtering

# --- Step 1: Load CSV ---
df = pd.read_csv(csv_file)

# --- Step 2: Filter rows based on custom x range ---
df_filtered = df[(df['speed'] >= x_min) & (df['speed'] <= x_max)]

# --- Step 3: Prepare data for regression ---
X = df_filtered['speed'].values.reshape(-1, 1)   # x values must be 2D for sklearn
y = df_filtered['velocity'].values              # y values

# --- Step 4: Perform linear regression ---
model = LinearRegression()
model.fit(X, y)

# --- Step 5: Print results ---
print(f"Slope (m): {model.coef_[0]}")
print(f"Intercept (b): {model.intercept_}")

# --- Step 6: Optional: Plot the data and regression line ---
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Speed')
plt.ylabel('Velocity')
plt.title(f'Linear Regression for speed between {x_min} and {x_max}')
plt.legend()
plt.show()
