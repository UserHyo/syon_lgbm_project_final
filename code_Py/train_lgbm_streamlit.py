# syon_train_streamlit_best.py (Part 1)

import os
import glob
import pickle
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def apply_transform(y, strategy):
    if strategy == "log":
        return np.log1p(y)
    elif strategy == "power":
        return np.power(y, 1.5)
    return y  # none

def inverse_transform(y_pred, strategy):
    if strategy == "log":
        return np.expm1(y_pred)
    elif strategy == "power":
        return np.power(y_pred, 1 / 1.5)
    return y_pred

def train_model(X_train, X_test, y_train, y_test, group, strategy):
    model = LGBMRegressor(random_state=42)

    # 변환
    y_train_trans = apply_transform(y_train, strategy)
    y_test_trans = apply_transform(y_test, strategy)

    # 학습
    model.fit(
        X_train, y_train_trans,
        eval_set=[(X_test, y_test_trans)],
        eval_metric="rmse",
        callbacks=[early_stopping(30)]
    )

    # 예측 및 역변환
    y_pred_trans = model.predict(X_test)
    y_pred = inverse_transform(y_pred_trans, strategy)
    y_test_eval = inverse_transform(y_test_trans, strategy)

    # 평가
    r2 = r2_score(y_test_eval, y_pred)
    mae = mean_absolute_error(y_test_eval, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred))
    mape = np.mean(np.abs((y_test_eval - y_pred) / (y_test_eval + 1e-5)))

    result = {
        "R2": round(r2, 2),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "strategy": strategy,
        "model": model
    }
    return result

# syon_train_streamlit_best.py (Part 2)

if __name__ == "__main__":
    folder_name = "syon"
    data_path = f"/content/drive/MyDrive/{folder_name}/Data"
    save_dir = f"/content/drive/MyDrive/{folder_name}/Models_LGBM_Streamlit"
    os.makedirs(save_dir, exist_ok=True)

    strategies = ["none"]
    performance_records = {}
    best_models = {}

    # ✅ 데이터 로딩
    for x_path in sorted(glob.glob(os.path.join(data_path, "LGBM_X_train_streamlit_*.pkl"))):
        group = os.path.basename(x_path).replace("LGBM_X_train_streamlit_", "").replace(".pkl", "")
        try:
            with open(x_path, "rb") as f1, \
                 open(x_path.replace("X_train", "X_test"), "rb") as f2, \
                 open(x_path.replace("X_train", "y_train"), "rb") as f3, \
                 open(x_path.replace("X_train", "y_test"), "rb") as f4:
                X_train = pickle.load(f1)
                X_test = pickle.load(f2)
                y_train = pickle.load(f3)
                y_test = pickle.load(f4)

                best_r2 = -np.inf
                best_result = None

                for strategy in strategies:
                    print(f"📦 학습 중: {group} / 전략: {strategy}")
                    result = train_model(X_train, X_test, y_train, y_test, group, strategy)

                    # 전략별 기록
                    performance_records[(group, strategy)] = result

                    if result["R2"] > best_r2:
                        best_r2 = result["R2"]
                        best_result = result

                # 베스트 모델 저장
                best_models[group] = best_result
                joblib.dump(best_result["model"], os.path.join(save_dir, f"{group}.pkl"))
                print(f"✅ {group} → best 전략: {best_result['strategy']} 모델 저장 완료")
                
                # ✅ Feature Importance 저장
                importance = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": best_result["model"].feature_importances_
                }).sort_values(by="importance", ascending=False)

                importance_path = os.path.join(save_dir, f"feature_importance_{group}.csv")
                importance.to_csv(importance_path, index=False)

                print(f"📌 {group} → 중요도 저장 완료: feature_importance_{group}.csv")

        except Exception as e:
            print(f"❌ {group} 로딩 실패: {e}")

    # ✅ 성능 결과 저장
    best_summary = {
        group: {
            "Strategy": result["strategy"],
            "R2": result["R2"],
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
            "MAPE": result["MAPE"]
        }
        for group, result in best_models.items()
    }

    best_df = pd.DataFrame(best_summary).T
    best_df.to_csv(os.path.join(save_dir, "performance_Models_LGBM_Streamlit.csv"))
    print("🎯 전체 best 성능 요약 저장 완료 → performance_Models_LGBM_Streamlit.csv")
