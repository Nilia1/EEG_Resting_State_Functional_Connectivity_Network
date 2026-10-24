import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RANSACRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Load data
# Update the filename to match your new file (use pd.read_csv if it is a .csv file)
df = pd.read_excel('dataset.xlsx')

# 2. Separate target label (y) and features (X)
# This assumes your target is 'GE_Ratio' and the remaining columns are BIS/BAS
y = df['GE_Ratio'].dropna()
X = df.loc[y.index].drop(columns=['GE_Ratio'])

# 3. Handle missing values (median imputation)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 4. Feature Scaling (Robust Scaler to handle behavioral outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_final = pd.DataFrame(X_scaled, columns=X.columns)

# 5. RANSAC Regression (Robust Outlier Rejection)
# We use Ridge as the base estimator. 
# min_samples=0.7 forces the model to find a pattern fitting at least 70% of subjects.
base_estimator = Ridge(alpha=10.0) 
ransac = RANSACRegressor(estimator=base_estimator, 
                         min_samples=0.7, 
                         random_state=42, 
                         max_trials=1000)

# 6. Evaluate the model using 5-Fold Cross Validation
y_pred_cv = cross_val_predict(ransac, X_final, y, cv=5)
mse = mean_squared_error(y, y_pred_cv)
mae = mean_absolute_error(y, y_pred_cv)
r2 = r2_score(y, y_pred_cv)

print("--- RANSAC + Robust Scaler (CV) Results ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}\n")

# 7. Fit on full data to identify the Outliers vs Inliers
ransac.fit(X_final, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

print(f"Total Subjects: {len(y)}")
print(f"Inliers Found: {inlier_mask.sum()}")
print(f"Outliers Ignored: {outlier_mask.sum()}\n")

# Extract and display feature coefficients
coefficients = ransac.estimator_.coef_
feature_importance = pd.DataFrame({'Feature': X_final.columns, 'Coefficient': coefficients})
feature_importance = feature_importance.sort_values(by='Coefficient', key=abs, ascending=False)

print("--- Feature Coefficients ---")
print(feature_importance.to_string(index=False))

# 8. Plotting Results
plt.figure(figsize=(14, 6))

# Plot 1: Actual vs Predicted (Inliers vs Outliers)
plt.subplot(1, 2, 1)
plt.scatter(y[inlier_mask], y_pred_cv[inlier_mask], color='mediumseagreen', label='Inliers', alpha=0.7, edgecolors='k')
plt.scatter(y[outlier_mask], y_pred_cv[outlier_mask], color='salmon', label='Outliers (Ignored)', alpha=0.7, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual GE_Ratio')
plt.ylabel('Predicted GE_Ratio')
plt.title('RANSAC: Actual vs. Predicted GE_Ratio')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Plot 2: Feature Coefficients
plt.subplot(1, 2, 2)
colors = ['salmon' if c < 0 else 'skyblue' for c in feature_importance['Coefficient'][::-1]]
plt.barh(feature_importance['Feature'][::-1], feature_importance['Coefficient'][::-1], color=colors, edgecolor='k')
plt.xlabel('Coefficient Value (Impact on Prediction)')
plt.title('Predictors of DMN Global Efficiency')
plt.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('bisbas_ransac_results.png')
print("\nPlot saved successfully as 'bisbas_ransac_results.png'.")