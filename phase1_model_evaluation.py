# ============================================================
# PHASE 1.1 + 1.2 + 1.3 — MODEL EVALUATION, BASELINE, EXPLAINABILITY
# ------------------------------------------------------------
# Paste these as notebook cells AFTER your model training cell
# and BEFORE the Gradio UI cell. They assume these already exist
# from your training code:
#
#   trained_models   -> dict, e.g. {"Linear Regression": lr_model,
#                                    "Random Forest": rf_model,
#                                    "Gradient Boosting": gb_model}
#   X_train, X_test, y_train, y_test  -> your existing split
#   clean_df         -> your cleaned DataFrame (has Source,
#                        Destination, Travel_Class, Price)
#
# If your variable names differ, just rename them at the top —
# nothing else needs to change.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# 1.1 — MODEL PERFORMANCE TABLE
# ------------------------------------------------------------

def evaluate_models(trained_models, X_test, y_test):
    rows = []
    for name, model in trained_models.items():
        preds = model.predict(X_test)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
        })
    results = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return results


results_df = evaluate_models(trained_models, X_test, y_test)
results_df.to_csv("ai_travel_model_comparison.csv", index=False)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("Model Performance")
print("─" * 40)
print(results_df.to_string(index=False))
print()
print(f"'{best_model_name}' was selected as the final model because it achieved "
      f"the best validation R² ({results_df.iloc[0]['R2']:.3f}) and lowest MAE "
      f"(₹{results_df.iloc[0]['MAE']:,.2f}) among the evaluated models.")

# --- charts (matches your existing filenames) ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(results_df["Model"], results_df["MAE"], color="#F5B942")
ax.set_ylabel("MAE (₹)")
ax.set_title("Model Comparison — Mean Absolute Error (lower is better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison_mae.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(results_df["Model"], results_df["R2"], color="#3DDC97")
ax.set_ylabel("R²")
ax.set_title("Model Comparison — R² (higher is better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison_r2.png", dpi=150)
plt.close()


# ------------------------------------------------------------
# 1.2 — BASELINE vs ML MODEL
# ------------------------------------------------------------
# Baseline = historical median fare for the same route + class.
# This answers "why did you even need machine learning?"

def baseline_predict(row, lookup):
    key = (row["Source"], row["Destination"], row["Travel_Class"])
    return lookup.get(key, clean_df["Price"].median())


baseline_lookup = (
    clean_df.groupby(["Source", "Destination", "Travel_Class"])["Price"]
    .median()
    .to_dict()
)

X_test_ctx = clean_df.loc[X_test.index] if hasattr(X_test, "index") else None

if X_test_ctx is not None:
    baseline_preds = X_test_ctx.apply(lambda r: baseline_predict(r, baseline_lookup), axis=1)
    ml_preds = best_model.predict(X_test)

    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    ml_mae = mean_absolute_error(y_test, ml_preds)
    improvement_pct = (baseline_mae - ml_mae) / baseline_mae * 100

    print("\nBaseline vs. ML Model")
    print("─" * 40)
    print(f"Historical Median Baseline  → MAE ₹{baseline_mae:,.2f}")
    print(f"{best_model_name} (ML)      → MAE ₹{ml_mae:,.2f}")
    print(f"Improvement over baseline: {improvement_pct:.1f}%")
else:
    print("\n[Skip] Could not align X_test back to clean_df — "
          "make sure X_test is a DataFrame slice with the original index, "
          "or rebuild this cell to join on your own row IDs.")


# ------------------------------------------------------------
# 1.3 — EXPLAINABILITY: PRICE DRIVERS
# ------------------------------------------------------------
# Works directly for tree-based models (Random Forest, Gradient
# Boosting) via .feature_importances_. If your final model is
# linear, use abs(coef_) instead — swap the line marked below.

feature_names = list(X_train.columns)

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_)  # linear model fallback
else:
    importances = None

if importances is not None:
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv("ai_travel_feature_importance.csv", index=False)

    print("\nPRICE DRIVERS")
    print("─" * 40)
    max_imp = importance_df["Importance"].max()
    for _, r in importance_df.head(10).iterrows():
        bar_len = int(r["Importance"] / max_imp * 20)
        print(f"{r['Feature']:<28} {'█' * bar_len}")

    fig, ax = plt.subplots(figsize=(7, 5))
    top = importance_df.head(10).iloc[::-1]
    ax.barh(top["Feature"], top["Importance"], color="#F5B942")
    ax.set_title("What Drives the Predicted Price")
    plt.tight_layout()
    plt.savefig("model_feature_importance.png", dpi=150)
    plt.close()
else:
    print("\n[Skip] best_model has neither feature_importances_ nor coef_ — "
          "tell me what model type it is and I'll add the right explainability method.")
