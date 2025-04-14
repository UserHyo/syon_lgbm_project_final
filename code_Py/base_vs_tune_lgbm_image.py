import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 설정
groups = ["BrunchSalad", "Noodles", "RiceDishes", "SideDish", "SoupStew", "StirFryGrill"]
base_model_path = "/content/drive/MyDrive/syon/Models_LGBM_Streamlit"
tune_model_path = "/content/drive/MyDrive/syon/Models_LGBM_Streamlit_Tuned"
data_path = "/content/drive/MyDrive/syon/Data"
save_dir = "/content/drive/MyDrive/syon/result_data_image"
os.makedirs(save_dir, exist_ok=True)

# 1️⃣ 성능 비교 - 4개 메트릭 하나로
perf_base = pd.read_csv(f"{base_model_path}/performance_Models_LGBM_Streamlit.csv", index_col=0)
perf_tune = pd.read_csv(f"{tune_model_path}/performance_Models_LGBM_Streamlit_Tuned.csv", index_col=0)
metrics = ["R2", "MAE", "RMSE", "MAPE"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, metric in enumerate(metrics):
    ax = axes[i // 2][i % 2]
    x = np.arange(len(groups))
    # ax.bar(x - 0.2, perf_base.loc[groups][metric], width=0.4, label="Base", color="skyblue")
    ax.bar(x + 0.2, perf_tune.loc[groups][metric], width=0.4, label="Tuned", color="orange")
    ax.set_title(f"{metric} Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45)
    ax.legend()
plt.tight_layout()
plt.savefig(f"{save_dir}/performance_all_metrics.png")
plt.close()

# 2️⃣ 예측 결과 (히스토그램 + 산점도)
fig, axes = plt.subplots(len(groups), 2, figsize=(12, len(groups)*3))
for i, group in enumerate(groups):
    try:
        with open(f"{data_path}/LGBM_X_test_streamlit_{group}.pkl", "rb") as f:
            X_test = pickle.load(f)
        with open(f"{data_path}/LGBM_y_test_streamlit_{group}.pkl", "rb") as f:
            y_test = pickle.load(f)
        model_base = joblib.load(os.path.join(base_model_path, f"{group}.pkl"))
        model_tuned = joblib.load(os.path.join(tune_model_path, f"{group}_tune.pkl"))
    except:
        print(f"🚫 Skipping group: {group}")
        continue

    # y_pred_base = model_base.predict(X_test)
    y_pred_tuned = model_tuned.predict(X_test)

    # 히스토그램
    axes[i][0].hist(y_test, bins=30, alpha=0.5, label="True", color="gray")
    # axes[i][0].hist(y_pred_base, bins=30, alpha=0.5, label="Base", color="skyblue")
    axes[i][0].hist(y_pred_tuned, bins=30, alpha=0.5, label="Tuned", color="orange")
    axes[i][0].set_title(f"{group} - Prediction Distribution")
    axes[i][0].legend()

    # 산점도
    # axes[i][1].scatter(y_test, y_pred_base, alpha=0.4, label="Base", color="blue")
    axes[i][1].scatter(y_test, y_pred_tuned, alpha=0.4, label="Tuned", color="darkorange")
    axes[i][1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[i][1].set_title(f"{group} - Scatter Plot")
    axes[i][1].legend()
plt.tight_layout()
plt.savefig(f"{save_dir}/prediction_all_groups.png")
plt.close()

# 3️⃣ 중요도 비교
fig, axes = plt.subplots(len(groups), 1, figsize=(10, len(groups)*3))
for i, group in enumerate(groups):
    fi_base_path = os.path.join(base_model_path, f"feature_importance_{group}.csv")
    fi_tune_path = os.path.join(tune_model_path, f"feature_importance_{group}_tune.csv")
    if not os.path.exists(fi_base_path) or not os.path.exists(fi_tune_path):
        continue

    fi_base = pd.read_csv(fi_base_path).set_index("feature")
    fi_tune = pd.read_csv(fi_tune_path).set_index("feature")
    top_features = fi_base.head(5).index.union(fi_tune.head(5).index)

    # base_vals = fi_base.loc[top_features]["importance"].fillna(0)
    tune_vals = fi_tune.loc[top_features]["importance"].fillna(0)

    x = np.arange(len(top_features))
    # axes[i].bar(x - 0.2, base_vals, width=0.4, label="Base", color="skyblue")
    axes[i].bar(x + 0.2, tune_vals, width=0.4, label="Tuned", color="orange")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(top_features, rotation=45, ha='right')
    axes[i].set_title(f"{group} - Feature Importance")
    axes[i].legend()
plt.tight_layout()
plt.savefig(f"{save_dir}/importance_all_groups.png")
plt.close()
