import pandas as pd
import matplotlib.pyplot as plt

# --- User Configuration ---
csv_file = 'outputs\motor_rl_constants2.csv'        # Path to your CSV file
x_column = 'speed'         # Name of the column for x-axis
y_column = 'theta'         # Name of the column for y-axis
title = 'Scatter Plot'       # Plot title
x_label = x_column           # Label for x-axis
y_label = y_column           # Label for y-axis

# --- Load CSV Data ---
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: The file '{csv_file}' was not found.")
    exit()


# --- Check if columns exist ---
if x_column not in df.columns or y_column not in df.columns:
    print(f"Error: One or both columns '{x_column}' and '{y_column}' not found in CSV.")
    print("Available columns:", df.columns.tolist())
    exit()

# --- Create Scatter Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(df[x_column], df[y_column], color='blue', alpha=0.7, edgecolors='w', s=100)
plt.title(title)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.grid(True)
plt.tight_layout()

# --- Show Plot ---
plt.show()
