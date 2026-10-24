import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load your dataset (Make sure the filename matches exactly)
# Note: If your file is strictly an Excel file, use pd.read_excel('dataset.xlsx')
df = pd.read_excel('dataset.xlsx')

# 2. Calculate the correlation matrix
# We use 'spearman' here because behavioral/brain data often has non-linear relationships and outliers. 
# You can change it to 'pearson' if you strictly want linear correlations.
corr_matrix = df.corr(method='spearman')

# 3. Extract and sort correlations specifically for your target variable
# This ranks the features from most impactful to least impactful (ignoring whether it's positive or negative)
target_col = 'GE_Ratio'
if target_col in corr_matrix.columns:
    print(f"--- Correlation of all features with {target_col} ---")
    target_correlations = corr_matrix[target_col].drop(target_col) # Remove self-correlation
    target_correlations_sorted = target_correlations.sort_values(key=abs, ascending=False)
    print(target_correlations_sorted)
    print("\n")
else:
    print(f"Error: Target column '{target_col}' not found in the dataset.")

# 4. Plot the full Correlation Heatmap
plt.figure(figsize=(12, 10))
# cmap='coolwarm' makes negative correlations blue and positive correlations red
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap (Spearman)')
plt.tight_layout()

# Save and show the plot
plt.savefig('correlation_heatmap.png')
print("Heatmap saved as 'correlation_heatmap.png'")
plt.show()