"""
syon_functions_extended.py

이 파일은 프로젝트의 데이터 병합 및 수집 파이프라인에 관련된 핵심 함수들을 포함합니다.

📌 포함된 기능:
    1. merge_recipes: TB_RECIPE_SEARCH CSV 파일을 병합하여 `food_database.csv`로 저장
    2. collect_naver_trend: `food_database.csv`을 기반으로 네이버 검색 트렌드 데이터를 수집
    3. merge_trends_weather: 네이버 트렌드 데이터와 기상 데이터를 날짜 및 지점(STN) 기준으로 병합

작성자: [이영웅]
작성일: [2025-03-12]
"""

import os
import glob
import json
import time
import random
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import sys


# =============================================================================
# 1. TB_RECIPE_SEARCH 파일 병합 및 레시피 데이터 생성 (merge_recipes)
# =============================================================================

def merge_recipes(folder_name):
    """
    📌 TB_RECIPE_SEARCH CSV 파일을 병합하여 음식군 데이터를 생성하는 함수

    🔹 주요 기능:
      - `TB_RECIPE_SEARCH*.csv` 파일 검색 및 병합
      - 인코딩 감지 후 UTF-8 변환
      - 불필요한 카테고리 제거
      - 음식 텍스트에서 음식군 자동 추출 (복수 + 대표)
      - 최종 결과를 CSV로 저장

    Args:
        folder_name (str): Google Drive 내 데이터 폴더 이름
    """
    print(f"\n🚀 [1] TB_RECIPE_SEARCH 파일을 병합합니다...")

    import os, glob
    import pandas as pd
    from datetime import datetime

    PROJECT_FOLDER = f"/content/drive/MyDrive/{folder_name}"
    OUTPUT_PATH = os.path.join(PROJECT_FOLDER, "Data")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    file_pattern = os.path.join(PROJECT_FOLDER, "**", "TB_RECIPE_SEARCH*.csv")
    file_paths = glob.glob(file_pattern, recursive=True)

    if not file_paths:
        raise FileNotFoundError("❌ TB_RECIPE_SEARCH 파일을 찾을 수 없습니다.")

    df_list = []
    encoding_priority = ["utf-8", "euc-kr", "cp949"]

    for file in file_paths:
        detected_encoding = None
        for enc in encoding_priority:
            try:
                with open(file, 'r', encoding=enc) as f:
                    f.readline()
                detected_encoding = enc
                break
            except Exception:
                continue

        detected_encoding = detected_encoding or "utf-8"

        try:
            temp_txt_file = file.replace(".csv", ".txt")
            with open(file, 'r', encoding=detected_encoding, errors='replace') as f, \
                 open(temp_txt_file, 'w', encoding='utf-8') as out_f:
                out_f.write(f.read())

            df = pd.read_csv(temp_txt_file, encoding='utf-8', dtype=str)
        finally:
            if os.path.exists(temp_txt_file):
                os.remove(temp_txt_file)

        # 필요 없는 카테고리 제거
        drop_categories = ["과자", "김치/젓갈/장류", "디저트", "밑반찬", "양념/소스/잼", "차/음료/술", "이유식"]
        df = df[~df["CKG_KND_ACTO_NM"].isin(drop_categories)]

        df_list.append(df)

    # 데이터 병합 및 정제
    df_food = pd.concat(df_list, ignore_index=True)[[
        "CKG_NM",              # 음식 이름
        "CKG_MTH_ACTO_NM",     # 조리법
        "CKG_STA_ACTO_NM",     # 주요 재료
        "CKG_MTRL_ACTO_NM",    # 부재료
        "CKG_KND_ACTO_NM",     # 음식 종류
        "CKG_IPDC",            # 인기도
        "FIRST_REG_DT",        # 최초 등록일
        "CKG_MTRL_CN",         # 레시피
        "RCP_TTL"              # 간략설명
    ]].copy()

    df_food["FIRST_REG_DT"] = pd.to_datetime(df_food["FIRST_REG_DT"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_food = df_food.sort_values(by=["CKG_NM", "FIRST_REG_DT"], ascending=[True, False]) \
                     .drop_duplicates(subset=["CKG_NM"], keep="first")

    # 🔹 복수 음식군 매핑 함수
    def extract_groups_from_row_v4(row):
        text = " ".join([str(row[col]) for col in ["CKG_MTH_ACTO_NM", "CKG_STA_ACTO_NM", "CKG_MTRL_ACTO_NM", "CKG_KND_ACTO_NM", "CKG_IPDC"] if pd.notna(row[col])])
        group_keywords = {
            "찌개/국/탕": ["김치찌개", "된장찌개", "감자탕", "순두부찌개", "부대찌개", "갈비탕", "해장국", "북어국", "콩나물국", "뼈해장국", "삼계탕", "추어탕", "미역국", "육개장", "곰탕", "설렁탕", "청국장", "매운탕", "재첩국", "찌개", "국", "탕"],
            "면요리": ["냉면", "라면", "우동", "쫄면", "잔치국수", "비빔국수", "칼국수", "막국수", "파스타", "크림파스타", "짜장면", "짬뽕", "쌀국수", "메밀국수", "로제파스타", "라구파스타", "라볶이", "냉모밀", "해물우동", "국수", "메밀", "스파게티"],
            "밥/죽/덮밥": ["김치볶음밥", "오므라이스", "전복죽", "참치덮밥", "규동", "스팸덮밥", "계란덮밥", "새우볶음밥", "차슈덮밥", "카레라이스", "쇠고기덮밥", "유부초밥", "장조림덮밥", "곱창덮밥", "닭갈비덮밥", "부타동", "낙지덮밥", "연어덮밥", "장어덮밥", "덮밥", "볶음밥", "죽", "비빔밥", "카레", "초밥"],
            "볶음/구이": ["제육볶음", "오징어볶음", "닭갈비", "소불고기", "돼지불고기", "곱창볶음", "김치볶음", "두루치기", "감자조림", "닭볶음탕", "고등어구이", "갈치구이", "꽁치구이", "닭봉구이", "연어스테이크", "버섯볶음", "낙지볶음", "가지볶음", "간장불고기", "볶음", "구이", "조림", "스테이크", "불고기", "전골"],
            "브런치/샐러드": ["샐러드", "연어샐러드", "에그마요샐러드", "닭가슴살샐러드", "고구마샐러드", "브런치", "오픈샌드위치", "오믈렛", "바나나팬케이크", "에그인헬", "토마토샐러드", "두부샐러드", "연두부샐러드", "리코타샐러드", "단호박샐러드", "크로플", "요거트볼", "그릭요거트", "시저샐러드", "샌드위치", "요거트", "팬케이크", "에그마요", "리코타"],
            "간편식": ["컵밥", "도시락", "냉동볶음밥", "즉석밥", "편의점도시락", "냉동피자", "핫도그", "햄버거", "컵누들", "라면사리", "치즈볼", "피자빵", "만두", "군만두", "핫바", "떡갈비", "볶음김치", "통조림햄", "계란말이", "즉석", "냉동", "편의점", "간편", "피자"],
            "안주/보양식": ["감바스", "삼계탕", "장어구이", "곱창전골", "닭발", "주꾸미볶음", "골뱅이무침", "닭똥집", "문어숙회", "계란찜", "매운오징어", "육회", "해물파전", "감자전", "소라무침", "전복구이", "황태구이", "돼지껍데기", "번데기탕", "장어", "전골", "곱창", "파전", "안주", "숙회", "찜", "소라", "번데기", "문어"]
        }
        matched_groups = [group for group, keywords in group_keywords.items() if any(kw in text for kw in keywords)]
        return ", ".join(sorted(set(matched_groups))) if matched_groups else None

    # 🔹 복수 음식군 생성
    df_food["CKG_GROUP_MULTI"] = df_food.apply(extract_groups_from_row_v4, axis=1)

    # 🔹 대표 음식군 추출 (복수 중 첫 번째)
    df_food["CKG_GROUP"] = df_food["CKG_GROUP_MULTI"].apply(
        lambda x: x.split(",")[0].strip() if pd.notna(x) and "," in x else x
    )
    # ✅ 간편식이 복수에 포함됐더라도 대표가 간편식이면 밥/죽/덮밥으로 재정의
    # 간편식이 실제 식품 구성상 밥/죽/덮밥 부분의 하위개념이 아주 강하기때문에 통일함(노이즈감소)
    df_food.loc[df_food["CKG_GROUP"] == "간편식", "CKG_GROUP"] = "밥/죽/덮밥"

    # 🔹 필요한 컬럼만 저장
    df_food_reduced = df_food[["CKG_NM", "CKG_GROUP", "CKG_GROUP_MULTI", "CKG_IPDC", "CKG_MTRL_CN", "RCP_TTL"]]
    merged_data_path_csv = os.path.join(OUTPUT_PATH, "food_database.csv")
    df_food_reduced.to_csv(merged_data_path_csv, index=False)

    print(f"\n🎉 음식 데이터 csv 저장 완료: {merged_data_path_csv}")



# =============================================================================
# 2. 네이버 검색 트렌드 데이터 수집 (collect_naver_trend)
# =============================================================================

import pandas as pd
import numpy as np
import os
import time
import random
from datetime import datetime, timedelta

# 그룹별 평균 미리 계산 (lookup 테이블 생성)
def collect_naver_trend(folder_name):
    """
    📌 `food_database.csv`을 기반으로 네이버 검색 트렌드 데이터를 수집하여 저장하는 함수
    """
    
    print(f"\n🚀 [2] 네이버 검색 트렌드 데이터 수집을 시작합니다...")

    # ✅ 기본 경로 설정
    PROJECT_FOLDER = f"/content/drive/MyDrive/{folder_name}"
    OUTPUT_PATH = os.path.join(PROJECT_FOLDER, "Data")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ✅ 음식 카테고리 파일 로드
    file_path = os.path.join(OUTPUT_PATH, "food_database.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ food_database.csv 파일이 없습니다.")

    df_food = pd.read_csv(file_path, encoding="utf-8-sig")
    base_food_categories = df_food["CKG_GROUP"].dropna().astype(str).unique().tolist()
    print(f"✅ {len(base_food_categories)}개의 음식 카테고리 추출 완료!")

    # ✅ 키워드 매핑 테이블
    keyword_groups_master = [
        {"groupName": "찌개/국/탕", "keywords": ["김치찌개", "된장찌개", "감자탕", "순두부찌개", "부대찌개", "갈비탕", "해장국", "북어국", "콩나물국", "뼈해장국", "삼계탕", "추어탕", "미역국", "육개장", "곰탕", "설렁탕", "청국장", "매운탕", "재첩국"]},
        {"groupName": "면요리", "keywords": ["냉면", "라면", "우동", "쫄면", "잔치국수", "비빔국수", "칼국수", "막국수", "파스타", "크림파스타", "짜장면", "짬뽕", "쌀국수", "메밀국수", "로제파스타", "라구파스타", "라볶이", "냉모밀", "해물우동"]},
        {"groupName": "밥/죽/덮밥", "keywords": ["김치볶음밥", "오므라이스", "전복죽", "참치덮밥", "규동", "스팸덮밥", "계란덮밥", "새우볶음밥", "차슈덮밥", "카레라이스", "쇠고기덮밥", "유부초밥", "장조림덮밥", "곱창덮밥", "닭갈비덮밥", "부타동", "낙지덮밥", "연어덮밥", "장어덮밥"]},
        {"groupName": "볶음/구이", "keywords": ["제육볶음", "오징어볶음", "닭갈비", "소불고기", "돼지불고기", "곱창볶음", "김치볶음", "두루치기", "감자조림", "닭볶음탕", "고등어구이", "갈치구이", "꽁치구이", "닭봉구이", "연어스테이크", "버섯볶음", "낙지볶음", "가지볶음", "간장불고기"]},
        {"groupName": "브런치/샐러드", "keywords": ["샐러드", "연어샐러드", "에그마요샐러드", "닭가슴살샐러드", "고구마샐러드", "브런치", "오픈샌드위치", "오믈렛", "바나나팬케이크", "에그인헬", "토마토샐러드", "두부샐러드", "연두부샐러드", "리코타샐러드", "단호박샐러드", "크로플", "요거트볼", "그릭요거트", "시저샐러드"]},
        {"groupName": "간편식", "keywords": ["컵밥", "도시락", "냉동볶음밥", "즉석밥", "편의점도시락", "냉동피자", "핫도그", "햄버거", "컵누들", "라면사리", "치즈볼", "피자빵", "만두", "군만두", "핫바", "떡갈비", "볶음김치", "통조림햄", "계란말이"]},
        {"groupName": "안주/보양식", "keywords": ["감바스", "삼계탕", "장어구이", "곱창전골", "닭발", "주꾸미볶음", "골뱅이무침", "닭똥집", "문어숙회", "계란찜", "매운오징어", "육회", "해물파전", "감자전", "소라무침", "전복구이", "황태구이", "돼지껍데기", "번데기탕"]}]

    # ✅ 누락된 GroupName 체크
    defined_groups = {entry["groupName"] for entry in keyword_groups_master}
    missing_groups = [group for group in base_food_categories if group not in defined_groups]

    if missing_groups:
        print("❌ 다음 그룹들은 키워드 매핑 테이블에 없습니다. keyword_groups_master를 확인하세요:")
        for g in missing_groups:
            print(f"   - {g}")
        raise ValueError("🚫 매핑되지 않은 그룹이 있어 수집을 중단합니다.")

    keyword_groups = [entry for entry in keyword_groups_master if entry["groupName"] in base_food_categories]

    # ✅ 네이버 API 인증 정보 및 설정
    NAVER_CLIENT_ID = "bT7UYFByAXsxF_zS9oXr"
    NAVER_CLIENT_SECRET = "zO4PA4vxYW"
    max_retries = 3

    # 날짜 범위: 최근 2년치
    today = datetime.today()
    current_year = today.year

    START_DATE = datetime(current_year - 4, 1, 1)
    END_DATE = datetime(current_year - 1, 12, 31) 

    # 누락 조합 생성용 변수
    start_dt = START_DATE.strftime('%Y-%m-%d')
    end_dt = END_DATE.strftime('%Y-%m-%d')
    base_groupnames = [entry["groupName"] for entry in keyword_groups]

    age_groups = {
        "청년층": ["2", "3", "4"],
        "중년층": ["5", "6", "7", "8"],
        "장년층": ["9", "10", "11"]
    }
    genders = ["m", "f"]
    long_format_path = os.path.join(OUTPUT_PATH, "syon_naver_trend_long.csv")

    # ✅ 네이버 API 호출 함수
    def get_naver_trend_data(keyword_batch, gender, age_list, age_name, start_date, end_date):
        url = "https://openapi.naver.com/v1/datalab/search"
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
            "Content-Type": "application/json"
        }
        request_body = {
            "startDate": start_date.strftime('%Y-%m-%d'),
            "endDate": end_date.strftime('%Y-%m-%d'),
            "timeUnit": "date",
            "keywordGroups": keyword_batch,
            "ages": age_list,
            "gender": gender
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=request_body)
                if response.status_code == 200:
                    data = response.json().get("results", [])
                    if not data:
                        return pd.DataFrame()
                    records = []
                    for keyword_data in data:
                        group_name = keyword_data["title"]
                        for entry in keyword_data["data"]:
                            records.append({
                                "Date": entry["period"],
                                "GroupName": group_name,
                                "Search_Count": entry["ratio"],
                                "Gender": gender,
                                "Age_Group": age_name
                            })
                    return pd.DataFrame(records)
            except requests.exceptions.RequestException as e:
                print(f"❌ 요청 실패 ({e}) - 재시도 {attempt + 1}/{max_retries}")
            time.sleep(2 * (attempt + 1))
        print("❌ 최대 재시도 초과")
        return pd.DataFrame()

    # ✅ 수집 루프
    final_data = []
    current_start = START_DATE

    # ✅ 1년 단위로 API 요청
    while current_start <= END_DATE:
        current_end = min(current_start + timedelta(days=365), END_DATE)
        for group in keyword_groups:
            for gender in genders:
                for age_name, age_values in age_groups.items():
                    print(f"📡 API 요청: {group['groupName']} | Gender: {gender}, Age: {age_name} | {current_start.date()} ~ {current_end.date()}")
                    result = get_naver_trend_data([group], gender, age_values, age_name, current_start, current_end)
                    if not result.empty:
                        final_data.append(result)
                    time.sleep(random.uniform(2, 4))
        current_start = current_end + timedelta(days=1)

    if final_data:
        df_final = pd.concat(final_data, ignore_index=True)

        # ✅ 모든 날짜 × GroupName × Gender × Age_Group 조합 생성
        print("\n🧩 누락된 날짜-조합 보간 처리 중...")
        all_dates = pd.date_range(start=start_dt, end=end_dt).strftime("%Y-%m-%d")
        full_index = pd.MultiIndex.from_product(
            [all_dates, base_groupnames, genders, age_groups.keys()],
            names=["Date", "GroupName", "Gender", "Age_Group"]
        )
        df_final = df_final.set_index(["Date", "GroupName", "Gender", "Age_Group"]).reindex(full_index).reset_index()

        # ✅ 전체 평균 (최후 fallback용)
        global_avg = df_final[df_final["Search_Count"] > 0]["Search_Count"].mean()
        global_avg = max(global_avg, 0.1)

        # 그룹별 평균 미리 계산 (lookup 테이블 생성)
        group_avg_lookup = df_final[df_final["Search_Count"] > 0].groupby(
            ["GroupName", "Gender", "Age_Group"]
        )["Search_Count"].mean().to_dict()

        # 보간 함수 수정 (그룹별 평균을 미리 계산하여 lookup 방식으로 보간)
        def fill_missing_v3(row):
            val = row["Search_Count"]
            if pd.notna(val) and val > 0:
                return val

            group_key = (row["GroupName"], row["Gender"], row["Age_Group"])
            return group_avg_lookup.get(group_key, global_avg)

        # 보간 적용
        df_final["Search_Count"] = df_final.apply(fill_missing_v3, axis=1)

        # ✅ 최소값 1 보정
        df_final["Search_Count"] = df_final["Search_Count"].apply(lambda x: max(x, 1))

        # ✅ 저장
        df_final.to_csv(long_format_path, index=False)
        print(f"\n✅ Long Format 저장 완료: {long_format_path}")

    else:
        print("⚠️ 수집된 데이터가 없습니다.")



# =============================================================================
# 3. 네이버 검색 트렌드 + 기상 데이터 병합 (merge_trends_weather)
# =============================================================================

def merge_trends_weather(folder_name):
    """
    📌 네이버 검색 트렌드 데이터와 기상 데이터를 병합하는 함수
    """
    print(f"\n🚀 [3] 네이버 검색 트렌드 + 기상 데이터 병합을 시작합니다...")

    PROJECT_FOLDER = f"/content/drive/MyDrive/{folder_name}"
    OUTPUT_PATH = os.path.join(PROJECT_FOLDER, "Data")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    long_format_path = os.path.join(OUTPUT_PATH, "syon_naver_trend_long.csv")
    if not os.path.exists(long_format_path):
        raise FileNotFoundError("❌ 트렌드 데이터 파일이 존재하지 않습니다.")

    print(f"✅ [3-1] 네이버 검색 트렌드 데이터 로드 중...")
    df_trends_long = pd.read_csv(long_format_path, encoding="utf-8-sig")
    df_trends_long["Date"] = pd.to_datetime(df_trends_long["Date"]).dt.strftime("%Y%m%d")
    print("✅ [3-2] 네이버 검색 트렌드 데이터 로드 완료!")

    STATION_LIST = [108, 119, 105, 131, 133, 156, 143, 159, 184]
    df_stations = pd.DataFrame({"STN": STATION_LIST, "key": 1})
    df_trends_long_expanded = df_trends_long.assign(key=1).merge(df_stations, on="key").drop(columns=["key"])

    print("✅ [3-3] 기상 데이터 관측소 정보 추가 완료!")

    API_KEY = "ll52-PwbRaCedvj8G9WgxA"
    BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"

    def get_weather_data(start_dt, end_dt, retries=3):
        params = {
            "stn": ",".join(map(str, STATION_LIST)),
            "tm1": start_dt.strftime("%Y%m%d"),
            "tm2": end_dt.strftime("%Y%m%d"),
            "authKey": API_KEY
        }
        for attempt in range(retries):
            try:
                response = requests.get(BASE_URL, params=params, timeout=60)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                print(f"❌ API 요청 오류: {e}, 재시도 {attempt + 1}/{retries} 후 10초 대기...")
                time.sleep(10)
        print(f"❌ 최대 재시도 횟수 초과 - 요청 실패")
        return ""

    start_date = df_trends_long["Date"].min()
    end_date = df_trends_long["Date"].max()
    print(f"📅 [3-4] 기상 데이터 수집 기간: {start_date} ~ {end_date}")

    current_start = datetime.strptime(start_date, "%Y%m%d")
    end_date_dt = datetime.strptime(end_date, "%Y%m%d")
    df_weather_list = []
    # 고정 컬럼 리스트
    weather_columns = [
        "Date", "STN",
        "WS_AVG", "WR_DAY", "WD_MAX", "WS_MAX", "WS_MAX_TM",
        "WD_INS", "WS_INS", "WS_INS_TM",
        "TA_AVG", "TA_MAX", "TA_MAX_TM", "TA_MIN", "TA_MIN_TM",
        "TD_AVG", "TS_AVG", "TG_MIN",
        "HM_AVG", "HM_MIN", "HM_MIN_TM",
        "PV_AVG", "EV_S", "EV_L",
        "FG_DUR",
        "PA_AVG", "PS_AVG", "PS_MAX", "PS_MAX_TM", "PS_MIN", "PS_MIN_TM",
        "CA_TOT", "SS_DAY", "SS_DUR", "SS_CMB",
        "SI_DAY", "SI_60M_MAX", "SI_60M_MAX_TM",
        "RN_DAY", "RN_D99", "RN_DUR",
        "RN_60M_MAX", "RN_60M_MAX_TM",
        "RN_10M_MAX", "RN_10M_MAX_TM",
        "RN_POW_MAX", "RN_POW_MAX_TM",
        "SD_NEW", "SD_NEW_TM", "SD_MAX", "SD_MAX_TM",
        "TE_05", "TE_10", "TE_15", "TE_30", "TE_50"
    ]

    while current_start <= end_date_dt:
        current_end = min(current_start + timedelta(days=365), end_date_dt)
        print(f"📡 [3-5] 기상 데이터 요청: {current_start.strftime('%Y-%m-%d')} ~ {current_end.strftime('%Y-%m-%d')}")
        raw_data = get_weather_data(current_start, current_end)

        extracted_data = []
        for line in raw_data.split("\n"):
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) != len(weather_columns):
                continue

            record = {}
            for col, val in zip(weather_columns, parts):
                if val in ["NULL", "-9", "-9.0", "-9.00"]:
                    record[col] = None
                elif col == "Date":
                    record[col] = val
                elif col == "STN":
                    record[col] = int(val)
                else:
                    try:
                        record[col] = float(val)
                    except ValueError:
                        record[col] = None
            extracted_data.append(record)

        if extracted_data:
            df_weather = pd.DataFrame(extracted_data)
            df_weather_list.append(df_weather)

        current_start = current_end + timedelta(days=1)

    print("✅ [3-6] 기상 데이터 수집 완료!")

    if df_weather_list:
        df_weather_new = pd.concat(df_weather_list, ignore_index=True)
        df_merged_long = pd.merge(df_trends_long_expanded, df_weather_new, on=["Date", "STN"], how="left")
        df_merged_long["Date"] = pd.to_datetime(df_merged_long["Date"], format="%Y%m%d")
        start_year = df_merged_long["Date"].dt.year.min()
        end_year = df_merged_long["Date"].dt.year.max()
        df_merged_long["Date"] = df_merged_long["Date"].dt.strftime("%Y-%m-%d")

        merged_long_dl_path = os.path.join(OUTPUT_PATH, "merge_trends_weather_long_dl.csv")
        df_merged_long.to_csv(merged_long_dl_path, index=False)
        print(f"\n✅ [3-7] 데이터 저장 완료 ({start_year}~{end_year}년): {merged_long_dl_path}")

# =============================================================================
# 🚀 전체 실행 블록 (__main__)
# =============================================================================
if __name__ == "__main__":
    import argparse

    # 📌 argparse를 사용하여 실행 옵션 추가
    parser = argparse.ArgumentParser(description="네이버 트렌드 & 기상 데이터 수집 및 병합 자동화")
    parser.add_argument("--folder", type=str, default="syon", help="Google Drive 프로젝트 폴더 이름 (기본값: 'syon')")
    parser.add_argument("--merge_recipes", action="store_true", help="TB_RECIPE_SEARCH CSV 파일을 병합하여 food_database.csv 저장")
    parser.add_argument("--collect_naver_trend", action="store_true", help="네이버 검색 트렌드 데이터 수집 및 저장")
    parser.add_argument("--merge_trends_weather", action="store_true", help="네이버 트렌드 + 기상 데이터 병합 및 저장")

    args = parser.parse_args()

    print("\n📌 === AI 데이터 수집 & 병합 자동화 시작 ===\n")

    # 📌 인자가 하나도 없으면 전체 실행
    if not (args.merge_recipes or args.collect_naver_trend or args.merge_trends_weather):
        print("⚠️ 실행 옵션이 지정되지 않았습니다. 모든 단계를 실행합니다.\n")
        args.merge_recipes = True
        args.collect_naver_trend = True
        args.merge_trends_weather = True

    # 📌 Step 1: 음식 데이터 병합 및 저장
    if args.merge_recipes:
        print("\n🚀 Step 1: TB_RECIPE_SEARCH 파일 병합 및 food_database.csv 저장 시작")
        merge_recipes(args.folder)
        print("✅ Step 1 완료!")

    # 📌 Step 2: 네이버 검색 트렌드 데이터 수집
    if args.collect_naver_trend:
        print("\n🚀 Step 2: 네이버 검색 트렌드 데이터 수집 시작")
        collect_naver_trend(args.folder)
        print("✅ Step 2 완료!")

    # 📌 Step 3: 네이버 트렌드 + 기상 데이터 병합
    if args.merge_trends_weather:
        print("\n🚀 Step 3: 네이버 검색 트렌드 + 기상 데이터 병합 시작")
        merge_trends_weather(args.folder)
        print("✅ Step 3 완료!")

    print("\n🎉 모든 과정이 성공적으로 완료되었습니다! ✅")

