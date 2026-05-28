import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def parse_label(value):
    # 숫자 형태면 int로, 문자가 섞여있으면 str로 반환하는 헬퍼 함수
    try:
        return int(value)
    except ValueError:
        return value

def generate_metadata_csv(data_dir: str, output_csv_path: str, random_seed: int = 42): # val_ratio: float = 0.2
    #지정된 디렉토리의 파일들을 분석하여 PyTorch 학습을 위한 메타데이터 CSV를 생성
    data_path = Path(data_dir)
    
    # 1. 특정 확장자(.csv) 파일들의 전체 경로 수집
    # data_path.rglob()을 사용하면 하위 디렉토리의 파일까지 모두 재귀적으로 찾을 수 있습니다.
    all_files = list(data_path.rglob('*.csv'))
    
    if not all_files:
        raise FileNotFoundError(f"Error: No files found in {data_dir}")

    parsed_data = []
    
    # 2. 파일 경로 파싱 및 라벨(character) 추출
    for file_path in all_files:
        # file_path.name -> '2021-07-28-181258_256x256_0_sungjae_RD_20ms.csv'
        filename = file_path.name
        # '_'를 기준으로 분할하여 '0' (인덱스 2) 값을 가져옴
        parts = filename.split('_')
        
        try:
            # 1. 명시적으로 int()를 씌워 정수형으로 변환합니다.
            # 만약 숫자가 아닌 문자열이 들어오면 ValueError를 발생시켜 예외처리로 넘어갑니다.
            character_label = parse_label(parts[2])
            
            # 저장할 상대 경로나 절대 경로 형태 지정 (예시와 동일하게 부모 폴더명 포함)
            # data_kine_digit_1000/... 형태의 문자열
            csv_path_str = f"{file_path.parent.name}/{file_path.name}"
            
            parsed_data.append({
                'csv_path': csv_path_str,
                'character': character_label
            })
        except (IndexError, ValueError) as e:
            # 파일명 규칙이 다르거나, 해당 위치의 값이 숫자가 아닐 경우 경고를 출력합니다.
            print(f"경고: 파싱 실패 (사유: {e}) -> 파일명: {filename}")
            continue

    # 3. DataFrame 변환
    df = pd.DataFrame(parsed_data)

    # 1. 파일명(csv_path) 기준으로 미리 오름차순 정렬
    # 이렇게 하면 CSV 파일을 열었을 때 0번부터 순서대로 보입니다.
    # df = df.sort_values(by='csv_path').reset_index(drop=True)
    
    # 4. Stratified Split을 통한 Train / Validation 분할
    # stratify=df['character'] 설정으로 각 클래스의 비율을 유지하며 분할
    # [수정 포인트] stratify 부분에 .astype(str) 추가
    # ---------------------------------------------------------
    try:
        # 데이터 분할 시에만 character 컬럼을 str로 형변환하여 전달 (자료형 충돌 방지)
        # 1단계: Train(60%) vs Remainder(40%) 분리
        df_train, df_rem = train_test_split(
            df, 
            test_size=0.4, 
            random_state=random_seed, 
            stratify=df['character'].astype(str) # 이 부분이 핵심!
        )
        df_val, df_test = train_test_split(
            df_rem, 
            test_size=0.5, 
            random_state=random_seed, 
            stratify=df_rem['character'].astype(str)
        )
    except ValueError as e:
        print(f"Stratified split failed: {e}. Falling back to random split.") # 데이터 부족으로 일반 무작위 분할을 수행
        df_train, df_rem = train_test_split(df, test_size=0.4, random_state=random_seed)
        df_val, df_test = train_test_split(df_rem, test_size=0.5, random_state=random_seed)
        
    # 2. Train / Val 분할
    # stratify를 적용하려면 모든 클래스의 샘플이 최소 2개 이상이어야 합니다.
    # 만약 클래스별 데이터가 너무 적으면 stratify=None으로 설정해야 합니다.
    # try:
    #     df_train, df_val = train_test_split(
    #         df, 
    #         test_size=val_ratio, 
    #         random_state=random_seed, 
    #         stratify=df['character']
    #     )
    # except ValueError:
    #     print("주의: 일부 클래스의 데이터가 너무 적어 일반 무작위 분할을 수행합니다.")
    #     df_train, df_val = train_test_split(
    #         df, 
    #         test_size=val_ratio, 
    #         random_state=random_seed
    #     )

    # 'split' 레이블 할당
    df_train = df_train.copy()
    df_train['split'] = 'train'
    
    df_val = df_val.copy()
    df_val['split'] = 'val'

    df_test = df_test.copy()
    df_test['split'] = 'test'
    
    # 다시 하나의 DataFrame으로 병합
    # final_df = pd.concat([df_train, df_val]).sort_values(by='csv_path').reset_index(drop=True)
    final_df = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)
    
    # 원하는 컬럼 순서대로 재배치
    final_df = final_df[['csv_path', 'character', 'split']]

    # 3. CSV 저장 전, csv_path (파일명) 기준으로 오름차순 정렬
    final_df = final_df.sort_values(by='csv_path').reset_index(drop=True)
    
    # 5. CSV 파일로 저장
    final_df.to_csv(output_csv_path, index=False, encoding='utf-8') # -sig
    print(f"총 {len(final_df)}개의 데이터 처리 완료. (Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)})")
    print(f"결과 파일 저장 완료: {output_csv_path}")

# ==========================================
# 실행 예시
# ==========================================
if __name__ == "__main__":
    # 타겟 데이터가 들어있는 폴더명을 지정하세요.
    TARGET_DIR = "/home/yunjinna/DynamicMNIST/Data/data_vector_62characters_100" 
    OUTPUT_CSV = "/home/yunjinna/DynamicMNIST/Data/data_split/data_split_62characters_100.csv"
    
    # 코드가 있는 위치에 임시 테스트용 폴더가 없으면 에러가 나므로, 실제 환경의 경로를 입력해 주시면 됩니다.
    generate_metadata_csv(data_dir=TARGET_DIR, output_csv_path=OUTPUT_CSV)
    # 한 문자당 100개