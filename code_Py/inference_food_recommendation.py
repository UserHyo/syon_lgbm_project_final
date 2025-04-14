# inference_food_recommendation.py

import pickle
import pandas as pd
import os

# 🛣️ 모델 파일들이 들어 있는 폴더 경로
MODEL_DIR = "/content/drive/MyDrive/syon/Models_LGBM_Streamlit_Tuned"

# 1. 음식군 매핑
food_categories = {
    "면요리": "Noodles",
    "밥/죽/덮밥": "RiceDishes",
    "볶음/구이": "StirFryGrill",
    "브런치/샐러드": "BrunchSalad",
    "안주/보양식": "SideDish",
    "찌개/국/탕": "SoupStew"
}

# 2. Feature 순서 정의
feature_order = [
    'avg_temp', 'min_temp', 'max_temp', 'rainfall', 'humidity',
    'wind_speed', 'day_of_week', 'is_weekend',
    'month', 'search_count', 'prev_day_score', 'holiday'
]

# 3. 사용자 입력값
user_input = {
    'avg_temp': 17.3,
    'min_temp': 12.0,
    'max_temp': 21.0,
    'rainfall': 0.0,
    'humidity': 50.0,
    'wind_speed': 3.0,
    'day_of_week': 4,
    'is_weekend': 0,
    'month': 4,
    'search_count': 75.0,
    'prev_day_score': 0.5,
    'holiday': 0
}

# 4. 입력값 준비
X_input = pd.DataFrame([user_input])[feature_order]

# 5. 예측 수행
results = {}
for kor_name, eng_name in food_categories.items():
    model_filename = f'{eng_name}_tune.pkl'
    model_path = os.path.join(MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        print(f"[경고] 모델 파일 없음: {model_path}")
        continue

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    pred = model.predict(X_input)[0]
    results[kor_name] = pred

# 6. 출력
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("\n🍽️ 음식군별 예측 결과:")
for food, score in sorted_results:
    print(f"{food}: {score:.4f}")
