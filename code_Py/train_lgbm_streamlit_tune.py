import os
import glob
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler

# 🔧 타겟 변환 함수
def apply_transform(y, strategy):
    if strategy == "log":
        return np.log1p(y)  # 로그 변환
    elif strategy == "power":
        return np.power(y, 1.5)  # power 변환
    elif strategy == "sqrt":
        return np.sqrt(y)  # 제곱근 변환
    return y  # 변환 없음

def inverse_transform(y_pred, strategy):
    if strategy == "log":
        return np.expm1(y_pred)  # 로그 변환을 되돌리기
    elif strategy == "power":
        return np.power(y_pred, 1 / 1.5)  # power 변환을 되돌리기
    elif strategy == "sqrt":
        return np.power(y_pred, 2)  # 제곱근 변환을 되돌리기 (제곱)
    return y_pred  # 변환 없음


# 모델 학습 함수
def train_model_with_tuning(X_train, X_test, y_train, y_test, group, strategy, n_iter=12):
    y_train_trans = apply_transform(y_train, strategy)
    y_test_trans = apply_transform(y_test, strategy)

    # param_grid = {
    #     "n_estimators": [100, 200, 300, 400],
    #     "max_depth": [4, 5, 6, 7, 8],
    #     "learning_rate": [0.01, 0.02, 0.05],
    #     "num_leaves": [15, 31, 63],
    #     "subsample": [0.6, 0.8, 1.0],
    #     "colsample_bytree": [0.6, 0.8, 1.0],
    #     "min_child_samples": [5, 10, 20],
    #     "min_child_weight": [1e-3, 1e-2, 1e-1],
    #     "min_gain_to_split": [0.0, 1e-5, 1e-2],
    # }
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.01, 0.02],
        "num_leaves": [15, 31],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_samples": [10],
        "min_child_weight": [1e-3],
        "min_gain_to_split": [1e-5],
    }


    sampled_params = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))

    best_metrics = {
        "R2": -np.inf,
        "MAE": np.inf,
        "RMSE": np.inf,
        "MAPE": np.inf,
    }
    best_model = None
    best_params = {}

    for params in sampled_params:
        model = LGBMRegressor(random_state=42, **params)
        model.fit(
            X_train, y_train_trans,
            eval_set=[(X_test, y_test_trans)],
            eval_metric=["rmse", "mae", "mape"],
            callbacks=[early_stopping(50)],  # Early stopping 적용
        )

        y_pred = inverse_transform(model.predict(X_test), strategy)
        y_true = inverse_transform(y_test_trans, strategy)

        # 성능 지표 계산
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5)))

        # 모델 성능 비교
        if (r2 > best_metrics["R2"]) and (mae < best_metrics["MAE"]) and (rmse < best_metrics["RMSE"]) and (mape < best_metrics["MAPE"]):
            best_metrics["R2"] = r2
            best_metrics["MAE"] = mae
            best_metrics["RMSE"] = rmse
            best_metrics["MAPE"] = mape
            best_model = model
            best_params = params

    return {
        "R2": round(best_metrics["R2"], 2),
        "MAE": round(best_metrics["MAE"], 2),
        "RMSE": round(best_metrics["RMSE"], 2),
        "MAPE": round(best_metrics["MAPE"], 2),
        "strategy": strategy,
        "params": best_params,
        "model": best_model
    }

# 메인 함수
if __name__ == "__main__":
    folder_name = "syon"
    data_path = f"/content/drive/MyDrive/{folder_name}/Data"
    save_dir = f"/content/drive/MyDrive/{folder_name}/Models_LGBM_Streamlit_Tuned"
    os.makedirs(save_dir, exist_ok=True)

    strategies = {
        "BrunchSalad": "power",
        "Noodles": "none",
        "RiceDishes": "power",
        "SideDish": "none",
        "SoupStew": "power",
        "StirFryGrill": "none"
    }

    performance_records = {}
    best_models = {}

    # 각 그룹별로 처리
    for x_path in sorted(glob.glob(os.path.join(data_path, "LGBM_X_train_streamlit_*.pkl"))):
        group = os.path.basename(x_path).replace("LGBM_X_train_streamlit_", "").replace(".pkl", "")
        strategy = strategies.get(group, "none")  # 전략을 그룹별로 다르게 지정

        try:
            with open(x_path, "rb") as f1, \
                 open(x_path.replace("X_train", "X_test"), "rb") as f2, \
                 open(x_path.replace("X_train", "y_train"), "rb") as f3, \
                 open(x_path.replace("X_train", "y_test"), "rb") as f4:
                X_train = pickle.load(f1)
                X_test = pickle.load(f2)
                y_train = pickle.load(f3)
                y_test = pickle.load(f4)

                best_result = None

                print(f"📦 학습 중: {group} / 전략: {strategy}")
                result = train_model_with_tuning(X_train, X_test, y_train, y_test, group, strategy)

                performance_records[(group, strategy)] = result

                best_result = result

                best_models[group] = best_result
                joblib.dump(best_result["model"], os.path.join(save_dir, f"{group}_tune.pkl"))
                print(f"✅ {group} → best 전략: {best_result['strategy']} / R2: {best_result['R2']} 저장 완료")
               
                # ✅ Feature Importance 저장
                importance = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": best_result["model"].feature_importances_
                }).sort_values(by="importance", ascending=False)

                importance_path = os.path.join(save_dir, f"feature_importance_{group}_tune.csv")
                importance.to_csv(importance_path, index=False)

                print(f"📌 {group} → 중요도 저장 완료: feature_importance_{group}_tune.csv")

        except Exception as e:
            print(f"❌ {group} 로딩 실패: {e}")

    # 📊 best 성능 요약 저장
    best_summary = {
        group: {
            "Strategy": result["strategy"],
            "R2": result["R2"],
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
            "MAPE": result["MAPE"],
            "BestParams": result["params"]
        }
        for group, result in best_models.items()
    }

    best_df = pd.DataFrame(best_summary).T
    best_df.to_csv(os.path.join(save_dir, "performance_Models_LGBM_Streamlit_Tuned.csv"))
    print("🎯 전체 best 성능 요약 저장 완료 → performance_Models_LGBM_Streamlit_Tuned.csv")
