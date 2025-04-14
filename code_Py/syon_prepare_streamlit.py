# syon_prepare_streamlit.py (Part 1)

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ✅ 한글 → 영문 매핑
groupname_map = {
    "면요리": "Noodles",
    "밥/죽/덮밥": "RiceDishes",
    "볶음/구이": "StirFryGrill",
    "브런치/샐러드": "BrunchSalad",
    "안주/보양식": "SideDish",
    "찌개/국/탕": "SoupStew"
}

def add_streamlit_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    df['is_weekend'] = df['Weekday'] >= 5
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)

    # ✅ RN_DAY → 강수량 처리
    if 'RN_DAY' in df.columns:
        # 강수량은 -9을 0으로 처리
        df['RN_DAY'] = df['RN_DAY'].replace(-9, 0)  # 강수량의 -9을 0으로 대체
        df['RN_DAY'] = df['RN_DAY'].replace("강수없음", 0).replace("-", 0)
        df['RN_DAY'] = pd.to_numeric(df['RN_DAY'], errors='coerce').fillna(0)

    # 기상 변수에서 -9 값을 NaN으로 처리
    weather_columns = ['TA_AVG', 'HM_AVG', 'WS_AVG', 'POP', 'Pressure', 'Sunshine']
    for col in weather_columns:
        if col in df.columns:
            df[col] = df[col].replace(-9, np.nan)  # -9 값을 NaN으로 처리

    # 결측치 처리 (NaN -> 중앙값 대체)
    if 'TA_AVG' in df.columns:
        df['TA_AVG'] = df['TA_AVG'].fillna(df['TA_AVG'].median())  # 기온 결측값 중앙값으로 대체

    if 'HM_AVG' in df.columns:
        df['HM_AVG'] = df['HM_AVG'].fillna(df['HM_AVG'].median())  # 습도 결측값 중앙값으로 대체

    if 'WS_AVG' in df.columns:
        df['WS_AVG'] = df['WS_AVG'].fillna(df['WS_AVG'].median())  # 풍속 결측값 중앙값으로 대체

    if 'POP' in df.columns:
        df['POP'] = df['POP'].fillna(df['POP'].median())  # 강수확률 결측값 중앙값으로 대체

    # 선형 보간법 (기타 기상 데이터에 대해)
    for col in ['Pressure', 'Sunshine']:  # 예시로 기압, 일조시간
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')  # 선형 보간법

    return df



# 1. 데이터 로딩
def load_data(folder_name):
    file_path = f"/content/drive/MyDrive/{folder_name}/Data/merge_trends_weather_long_dl.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 파일이 존재하지 않습니다: {file_path}")
    print(f"✅ 데이터 로드 성공: {file_path}")
    
    df = pd.read_csv(file_path)

    # ✅ GroupName 영어 치환
    if 'GroupName' in df.columns:
        df['GroupName'] = df['GroupName'].map(groupname_map).fillna(df['GroupName'])

    return df

# syon_prepare_streamlit.py (Part 3)

def encode_and_scale_features(df):
    df = df.copy()
    encoders = {}
    scalers = {}

    # 🎯 범주형: Gender, Age, Region(STN)
    le_gender = LabelEncoder()
    df["Gender"] = le_gender.fit_transform(df["Gender"])
    encoders["gender"] = le_gender

    le_age = LabelEncoder()
    df["Age_Group"] = le_age.fit_transform(df["Age_Group"])
    encoders["age_group"] = le_age

    stn_to_region = {
        108: "서울", 119: "수원", 105: "강릉",
        131: "청주", 133: "대전", 156: "광주",
        143: "대구", 159: "부산", 184: "제주"
    }
    df["Region"] = df["STN"].map(stn_to_region)

    le_region = LabelEncoder()
    df["Region"] = le_region.fit_transform(df["Region"])
    encoders["region"] = le_region

    df.drop(columns=["STN"], inplace=True)

    # 🎯 수치형 스케일링
    standard_cols = ["TA_AVG", "HM_AVG", "WS_AVG"]  # 기온, 습도, 풍속
    minmax_cols = ["RN_DAY"]  # 강수량

    if 'POP' in df.columns:  # 강수확률 있을 경우만
        minmax_cols.append("POP")

    for col in standard_cols:
        if col in df.columns:
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler

    for col in minmax_cols:
        if col in df.columns:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler

    return df, encoders, scalers

# syon_prepare_streamlit.py (Part 4)

def preprocess_groupwise(df, target_col="Search_Count", group_col="GroupName"):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    cutoff = df["Date"].max() - pd.Timedelta(days=365)

    SERVICE_FEATURES = [
        "Gender", "Age_Group", "Region", "TA_AVG", "HM_AVG", "WS_AVG", "RN_DAY",
        "Month_sin", "Month_cos", "Day_sin", "Day_cos", "is_weekend"
    ]
    if "POP" in df.columns:
        SERVICE_FEATURES.append("POP")

    data_by_group = {}
    for group in df[group_col].unique():
        df_g = df[df[group_col] == group]
        train = df_g[df_g["Date"] < cutoff]
        test = df_g[df_g["Date"] >= cutoff]

        feature_cols = [col for col in SERVICE_FEATURES if col in df.columns]
        X_train, y_train = train[feature_cols], train[target_col]
        X_test, y_test = test[feature_cols], test[target_col]

        data_by_group[group] = (X_train, X_test, y_train, y_test)

    return data_by_group


# syon_prepare_streamlit.py (Part 5)

def save_artifacts(data_by_group, encoders, scalers, folder_name):
    output_path = f"/content/drive/MyDrive/{folder_name}/Data"
    os.makedirs(output_path, exist_ok=True)

    for group, (X_train, X_test, y_train, y_test) in data_by_group.items():
        safe_name = group.replace("/", "_")
        with open(os.path.join(output_path, f"LGBM_X_train_streamlit_{safe_name}.pkl"), "wb") as f:
            pickle.dump(X_train, f)
        with open(os.path.join(output_path, f"LGBM_X_test_streamlit_{safe_name}.pkl"), "wb") as f:
            pickle.dump(X_test, f)
        with open(os.path.join(output_path, f"LGBM_y_train_streamlit_{safe_name}.pkl"), "wb") as f:
            pickle.dump(y_train, f)
        with open(os.path.join(output_path, f"LGBM_y_test_streamlit_{safe_name}.pkl"), "wb") as f:
            pickle.dump(y_test, f)

    # 전처리기 저장
    with open(os.path.join(output_path, "streamlit_label_encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(output_path, "streamlit_scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    print("✅ 서비스용 전처리 데이터 저장 완료!")

# syon_prepare_streamlit.py (main)

if __name__ == "__main__":
    folder = "syon"
    df = load_data(folder)
    df = add_streamlit_features(df)
    df, encoders, scalers = encode_and_scale_features(df)
    data_by_group = preprocess_groupwise(df)
    save_artifacts(data_by_group, encoders, scalers, folder)