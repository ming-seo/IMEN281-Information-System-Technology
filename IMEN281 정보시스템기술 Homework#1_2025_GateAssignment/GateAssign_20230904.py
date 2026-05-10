import sys
import time
import pandas as pd
import traceback

class GateAssign:
    # 클래스 생성자
    def __init__(self, input_files, interval1, interval2, algo_type, obj_type):
        self.flight_input_file = input_files[0]
        self.eligibility_files = input_files[1:6]
        self.preference_file = input_files[6]
        self.min_interval = int(interval1)
        self.preferred_interval = int(interval2)
        self.algo_type = int(algo_type)
        self.obj_type = int(obj_type)
        self.flights_df = None
        self.gate_schedule = {}
        self.compute_time = 0
        self.result_df = None
    
    # 파일 read 함수
    def read_csv(self, file_path):
        return pd.read_csv(file_path, encoding = 'cp949')
    
    # input 파일 읽고 dataframe으로 병합
    def load_flights(self):
        flights_input_df = self.read_csv(self.flight_input_file)
        # 불필요 columns 제거
        flights_input_df.drop(columns=['번호', '운항일자', '항공사', '여객/화물'], inplace=True, errors = 'ignore')
        
        # 공백 제거
        if 'Fleet' in flights_input_df.columns:
            flights_input_df['Fleet'] = flights_input_df['Fleet'].astype(str).str.strip()
        if '터미널' in flights_input_df.columns:
            flights_input_df['터미널'] = flights_input_df['터미널'].astype(str).str.strip()
        
        # 도착편과 출발편 연결
        self.flights_df = self.match_flights(flights_input_df)
        
        # 선호 게이트 정보 dataframe에 추가
        preference_df = self.load_preferences()
        self.merge_preferences(preference_df)
        
        # 사용 가능한 게이트 정보 dataframe에 추가
        eligibility_df = self.load_eligibility()
        self.merge_eligibility(eligibility_df)
        
    def match_flights(self, flights_df):
        # 출발편과 도착편 연결
        # 항공편을 출발/도착으로 각각 나누고 합쳐서 도착편명, 도착시간, 출발편명, 출발시간, Fleet, 터미널 열 만들기
        
        # 항공편을 도착편과 출발편으로 분리
        arrivals = flights_df[flights_df['I/O'] == '도착'].copy()
        departures = flights_df[flights_df['I/O'] == '출발'].copy()
        
        # 연결편 기준으로 병합
        merged_df = pd.merge(
            arrivals,
            departures,
            left_on = '연결편명',
            right_on = '편명',
            suffixes = ('_도착', '_출발')
        )
        
        # dataframe 생성
        matched_df = pd.DataFrame({
            '도착편명': merged_df['편명_도착'],
            '도착시간': merged_df['예정시간_도착'],
            '출발편명': merged_df['편명_출발'],
            '출발시간': merged_df['예정시간_출발'],
            'Fleet': merged_df['Fleet_도착'],
            '터미널': merged_df['터미널_도착']
        })
        
        return matched_df
    
    def load_eligibility(self):
        # 사용 가능한 게이트 정보 dataframe 생성
        
        eligibility_list = []
        
        # eligibility 파일 읽기
        for file in self.eligibility_files:
            temp_df = self.read_csv(file)
            
            # 공백 제거
            if 'Fleet' in temp_df.columns:
                temp_df['Fleet'] = temp_df['Fleet'].astype(str).str.strip()
            temp_df.columns = [str(col).strip() for col in temp_df.columns]
            
            # 파일명에서 터미널명 추출
            terminal = file.split('(')[1].split(')')[0]
            temp_df['터미널'] = terminal
            
            # 게이트 목록
            gate_columns = [col for col in temp_df.columns if col not in ['Fleet']]

            # Fleet별 사용 가능한 게이트 리스트
            eligibility_temp = []
            for _, row in temp_df.iterrows():
                gates = [gate for gate in gate_columns if str(row[gate]).strip() == 'O']
                eligibility_temp.append(gates)
            temp_df['eligible_gates'] = eligibility_temp
            
            # 리스트로 저장
            eligibility_list.append(temp_df[['Fleet', '터미널', 'eligible_gates']])
        
        # 모든 파일 정보 저장한 리스트 병합
        return pd.concat(eligibility_list, ignore_index = True)
            
    def merge_eligibility(self, eligibility_df):
        # 사용 가능한 게이트 dataframe과 원본 dataframe 병합
        # Fleet, 터미널 기준으로 병합
        elig_merged = pd.merge(
            self.flights_df,
            eligibility_df,
            left_on = ['Fleet', '터미널'],
            right_on = ['Fleet', '터미널'],
            how = 'left'
        )  
        self.flights_df = elig_merged
    
    def load_preferences(self):
        # 선호 게이트 dataframe 생성
        preference_df = self.read_csv(self.preference_file)
        preference_df.rename(columns={'flight':'편명'}, inplace=True)
        return preference_df
    
    def merge_preferences(self, preference_df):
        # 선호 게이트 dataframe과 원본 dataframe 병합
        
        # 도착편명 기준으로 1차 병합
        pref_merged_on_arr = pd.merge(
            self.flights_df,
            preference_df,
            left_on = '도착편명',
            right_on = '편명',
            how = 'left'
        )
        pref_merged_on_arr.rename(columns={'Gate':'Gate_pref_arr'}, inplace = True)
        pref_merged_on_arr.drop(columns=['편명'], inplace = True)
        
        # 출발편명 기준으로 2차 병합
        pref_merged = pd.merge(
            pref_merged_on_arr,
            preference_df,
            left_on = '출발편명',
            right_on = '편명',
            how = 'left'
        )
        pref_merged.rename(columns={'Gate':'Gate_pref_dep'}, inplace = True)
        pref_merged.drop(columns=['편명'], inplace = True)
        
        # 최종 병합
        pref_merged['Gate_preference'] = pref_merged['Gate_pref_dep'].fillna(pref_merged['Gate_pref_arr'])
        
        # 불필요 columns 제거
        pref_merged.drop(columns=['Gate_pref_arr', 'Gate_pref_dep'], inplace = True)
        self.flights_df = pref_merged

    # FFD 알고리즘 구현
    def FFD(self):
        # 도착시간과 출발시간을 분 단위로 변환한 column 추가
        self.flights_df['도착시간_분'] = self.flights_df['도착시간'].apply(time_to_minutes)
        self.flights_df['출발시간_분'] = self.flights_df['출발시간'].apply(time_to_minutes)

        # 항공편별 게이트 점유 시간 계산 후 내림차순 정렬
        self.flights_df['점유시간'] = self.flights_df['출발시간_분'] - self.flights_df['도착시간_분']
        sorted_flights_df = self.flights_df.sort_values(by='점유시간', ascending=False).copy()
        
        # 배정된 게이트 정보를 저장할 column 생성
        sorted_flights_df['Assigned_Gate'] = None

        # 사용 가능한 게이트 목록 생성
        all_gates = set()
        for gate_list in self.flights_df['eligible_gates'].dropna():
            if isinstance(gate_list, list):
                all_gates.update(gate_list)
        # 게이트 정렬
        sorted_gates = sorted(list(all_gates))
        
        # 점유시간이 긴 항공편부터 순차적으로 게이트 배정
        for index, flight in sorted_flights_df.iterrows():
            arrival_min = flight['도착시간_분']
            departure_min = flight['출발시간_분']
            eligible_gates = flight['eligible_gates']
            
            # 배정 가능한 게이트가 없는 경우 배정하지 않음
            if not isinstance(eligible_gates, list):
                continue

            # 게이트 목록을 순회하며 배정 가능한 첫 번째 게이트에 배정
            for gate in sorted_gates:
                # 해당 항공편이 사용할 수 없는 게이트인 경우 건너뜀
                if gate not in eligible_gates:
                    continue

                # 항공편 간 간격 확인하는 변수 (충돌 여부)
                is_conflict = False
                # 해당 게이트에 이미 배정된 항공편이 있는 경우
                if gate in self.gate_schedule:
                    for scheduled_arrival, scheduled_departure in self.gate_schedule[gate]:
                        # 현재 항공편과 이전에 배정된 항공편 간 최소 간격 만족하지 않으면 충돌
                        if not ((arrival_min >= scheduled_departure + self.min_interval) or (departure_min + self.min_interval <= scheduled_arrival)):
                            is_conflict = True
                            break
                
                # 충돌하지 않는 경우 게이트에 항공편 배정
                if not is_conflict:
                    sorted_flights_df.at[index, 'Assigned_Gate'] = gate
                    # 게이트 스케줄에 현재 항공편 추가
                    if gate not in self.gate_schedule:
                        self.gate_schedule[gate] = []
                    self.gate_schedule[gate].append((arrival_min, departure_min))
                    break

        # 임시로 사용한 columns 제거
        self.flights_df = sorted_flights_df.drop(columns=['도착시간_분', '출발시간_분', '점유시간'])
        
    def your_algorithm(self):
        # 분 단위로 변환한 columns 추가
        self.flights_df['도착시간_분'] = self.flights_df['도착시간'].apply(time_to_minutes)
        self.flights_df['출발시간_분'] = self.flights_df['출발시간'].apply(time_to_minutes)
        self.flights_df['점유시간'] = self.flights_df['출발시간_분'] - self.flights_df['도착시간_분']
        # 사용 가능한 게이트 수 계산하여 column 추가
        self.flights_df['eligible_gate_count'] = self.flights_df['eligible_gates'].apply(len)
        self.flights_df['Assigned_Gate'] = None
        
        # 전체 게이트 목록 정렬
        all_gates = sorted(list(set(g for gates in self.flights_df['eligible_gates'] if isinstance(gates, list) for g in gates)))

        # 선호 게이트 우선 배정
        # 사용 가능한 게이트 목록에 선호 게이트가 있는 항공편 필터링
        pref_flights = self.flights_df[self.flights_df.apply(lambda row: str(row['Gate_preference']).split('.')[0] in row['eligible_gates'], axis=1)].copy()
        pref_flights['Gate_preference'] = pref_flights['Gate_preference'].astype(str).str.split('.').str[0]
        # 선호 게이트별 그룹화
        grouped_by_pref = pref_flights.groupby('Gate_preference')
        assigned_indices_pass1 = []

        # 선호 게이트 그룹 내 게이트 배정
        for gate, group in grouped_by_pref:
            # 도착 시간 기준 항공편 정렬
            flight_times = sorted([(idx, row['도착시간_분'], row['출발시간_분']) for idx, row in group.iterrows()], key=lambda x: x[1])
            # 그룹 내 항공편 간 충돌 여부 확인하는 변수
            is_conflicted_group = False
            # 그룹 내 항공편 간 충돌 여부 검사
            if len(flight_times) > 1:
                for i in range(len(flight_times) - 1):
                    if flight_times[i][2] + self.min_interval > flight_times[i+1][1]:
                        is_conflicted_group = True; break
            # 그룹 내 충돌 없는 경우 선호 게이트에 항공편 모두 배정
            if not is_conflicted_group:
                for index, arr, dep in flight_times:
                    self.flights_df.at[index, 'Assigned_Gate'] = gate
                    assigned_indices_pass1.append(index)
                self.gate_schedule[gate] = [(arr, dep) for _, arr, dep in flight_times]

        # 잔여 항공편 배정
        # 게이트가 배정되지 않은 항공편 필터링
        remaining_flights_df = self.flights_df[~self.flights_df.index.isin(assigned_indices_pass1)]
        # 점유시간 기준 내림차순, 배정 가능 게이트 수 기준 오름차순 정렬
        sorted_remaining_df = remaining_flights_df.sort_values(by=['점유시간', 'eligible_gate_count'], ascending=[False, True])

        # 정렬 순서대로 게이트 배정
        for index, flight in sorted_remaining_df.iterrows():
            arrival_min, departure_min = flight['도착시간_분'], flight['출발시간_분']
            eligible_gates = flight['eligible_gates']
            preferred_gate = str(flight['Gate_preference']).split('.')[0]
            was_assigned = False

            # 선호 게이트가 있는 경우
            if preferred_gate != 'nan' and preferred_gate in eligible_gates:
                is_conflict = False
                if preferred_gate in self.gate_schedule:
                    for sched_arr, sched_dep in self.gate_schedule[preferred_gate]:
                        if not (arrival_min >= sched_dep + self.min_interval or departure_min + self.min_interval <= sched_arr):
                            is_conflict = True; break
                if not is_conflict:
                    self.flights_df.at[index, 'Assigned_Gate'] = preferred_gate
                    self.gate_schedule.setdefault(preferred_gate, []).append((arrival_min, departure_min))
                    was_assigned = True
            if was_assigned: continue

            # object_type 1인 경우 - 사용 게이트 수 최소화
            if self.obj_type == 1:
                # 항공편 스케줄링이 되어있는 게이트에 우선 배정
                active_gates = sorted(list(self.gate_schedule.keys()))
                for gate in active_gates:
                    if gate in eligible_gates:
                        is_conflict = False
                        for sched_arr, sched_dep in self.gate_schedule[gate]:
                            if not (arrival_min >= sched_dep + self.min_interval or departure_min + self.min_interval <= sched_arr):
                                is_conflict = True; break
                        if not is_conflict:
                            self.flights_df.at[index, 'Assigned_Gate'] = gate
                            self.gate_schedule[gate].append((arrival_min, departure_min))
                            was_assigned = True
                            break
                if was_assigned: continue

                # 항공편 스케줄링이 되어있는 게이트 중 배정 가능한 게이트가 없는 경우 새로운 게이트에 배정
                for gate in all_gates:
                    if gate in eligible_gates and gate not in self.gate_schedule:
                        self.flights_df.at[index, 'Assigned_Gate'] = gate
                        self.gate_schedule.setdefault(gate, []).append((arrival_min, departure_min))
                        was_assigned = True
                        break
                if was_assigned: continue

            # object_type 2인 경우 - 항공편 간 간격 불만족 최소화
            elif self.obj_type == 2:
                first_available_gate, ideal_gate = None, None
                # 사용 가능한 모든 게이트 검사
                for gate in all_gates:
                    if gate in eligible_gates:
                        is_conflict = False
                        if gate in self.gate_schedule:
                            for sched_arr, sched_dep in self.gate_schedule[gate]:
                                if not (arrival_min >= sched_dep + self.min_interval or departure_min + self.min_interval <= sched_arr):
                                    is_conflict = True
                                    break
                        
                        # 충돌이 없는 경우
                        if not is_conflict:
                            # 배정 가능한 첫 번째 게이트를 우선 저장
                            if first_available_gate is None: first_available_gate = gate

                            # 시간 간격 만족하는지 확인하는 변수
                            is_available = False
                            
                            # 게이트에 할당된 항공편 스케줄이 없는 경우
                            if gate not in self.gate_schedule: is_available = True
                            # 게이트에 다음 항공편이 없거나 시간 간격이 충분한 경우
                            else:
                                # 현재 항공편 이후에 출발하는 다음 항공편 목록
                                next_arrivals = [s_arr for s_arr, s_dep in self.gate_schedule[gate] if s_arr >= departure_min]
                                # 다음 항공편이 없는 경우
                                if not next_arrivals: is_available = True
                                # 다음 항공편이 있는 경우
                                else:
                                    gap = min(next_arrivals) - departure_min
                                    if gap >= self.preferred_interval: is_available = True
                            
                            # 시간 간격 만족하는 게이트가 있는 경우 게이트 탐색 종료
                            if is_available:
                                ideal_gate = gate
                                break

                # 사용 가능한 게이트가 있는 경우 해당 게이트로 배정, 아닌 경우 배정 가능한 첫 번째 게이트에 배정
                final_gate = ideal_gate if ideal_gate is not None else first_available_gate
                if final_gate:
                    self.flights_df.at[index, 'Assigned_Gate'] = final_gate
                    self.gate_schedule.setdefault(final_gate, []).append((arrival_min, departure_min))

        # 임시로 사용한 불필요 columns 제거
        self.flights_df.drop(columns=['도착시간_분', '출발시간_분', '점유시간', 'eligible_gate_count'], inplace=True, errors='ignore')
    
    def assign_gates(self):
        # 시작 시간 기록
        start_time = time.time()
        
        # 입력 옵션에 따라 알고리즘 실행
        if self.algo_type == 1: self.FFD()
        elif self.algo_type == 2: self.your_algorithm()
        
        # 종료 시간 기록
        end_time = time.time()
        
        # 실행 시간 계산
        self.compute_time = (end_time - start_time) * 1000
        
        # 결과 저장
        self.result_df = self.flights_df.copy()
    
    # 결과 출력 csv 파일 작성
    def write_output(self, output_file):
        # 알고리즘 타입, 실행 시간 및 각종 수치 저장
        algo_name = "First Fit Decreasing" if self.algo_type == 1 else "Your algorithm"
        comp_time_str = f"{self.compute_time:.1f} ms"
        assigned_flights = self.result_df.dropna(subset=['Assigned_Gate']).copy()
        num_used_gates = assigned_flights['Assigned_Gate'].nunique()
        num_unassigned = self.result_df['Assigned_Gate'].isnull().sum()
        assigned_flights['Gate_preference_str'] = assigned_flights['Gate_preference'].astype(str)
        assigned_flights['Assigned_Gate_str'] = assigned_flights['Assigned_Gate'].astype(str)
        num_preferred = (assigned_flights['Assigned_Gate_str'] == assigned_flights['Gate_preference_str']).sum()
        
        # 권장 간격 불만족 항공편 수 계산
        assigned_flights['도착시간_분'] = assigned_flights['도착시간'].apply(time_to_minutes)
        assigned_flights['출발시간_분'] = assigned_flights['출발시간'].apply(time_to_minutes)
        assigned_flights.sort_values(by=['Assigned_Gate', '도착시간_분'], inplace=True)
        assigned_flights['다음_항공편_도착시간_분'] = assigned_flights.groupby('Assigned_Gate')['도착시간_분'].shift(-1)
        assigned_flights['간격'] = assigned_flights['다음_항공편_도착시간_분'] - assigned_flights['출발시간_분']
        num_under_interval = assigned_flights[(assigned_flights['간격'] > 0) & (assigned_flights['간격'] < self.preferred_interval)].shape[0]

        # 파일 작성
        with open(output_file, 'w', encoding='cp949', newline='') as f:
            # 결과 요약
            f.write(f"Name = 조민서\n")
            f.write(f"Algorithm Type = {algo_name}\n")
            f.write(f"Computation time = {comp_time_str}\n")
            f.write(f"Number of used gates = {num_used_gates}\n")
            f.write(f"Number of unassigned flights = {num_unassigned}\n")
            f.write(f"Number of assigned flights to their preferred gates = {num_preferred}\n")
            f.write(f"Number of flights under the recommended interval time ({self.preferred_interval} min) = {num_under_interval}\n")

            # 출력 컬럼 지정
            output_columns = ['도착편명', '도착시간', '출발편명', '출발시간', 'Fleet', '터미널', 'Assigned_Gate']
            output_df = assigned_flights[output_columns].rename(columns={'Assigned_Gate': 'Gate'})
            
            # dataframe csv 파일에 추가
            output_df.to_csv(f, index=False)

# 시간 HH:MM 형식을 분 단위로 변환 (0시 기준으로 경과한 시간)
def time_to_minutes(time_str):
    try:
        h, m = map(int, time_str.split(':'))
        return h * 60 + m
    except (ValueError, AttributeError):
        return -1
    
# 입력 오류 시 출력되는 프로그램 사용법
def show_usage():
    usage_text = """
    [실행 명령어] 
    python.exe GateAssign_20230904.py <input file names> <interval1> 
    <interval2> <algorithm type> <objective type> <output file name>
    
    [참고] 프로그램 개발시 사용하는 launch.json 파일 argument
    args: “Flights_input.csv”, “Eligibility(P01).csv”, “Eligibility(P02).csv”,
    “Eligibility(P03).csv”, “Eligibility(C01).csv”, “Eligibility(C02).csv”, 
    “Gate_preference.csv”, “30”, “60”, “2”, “1”, “GateAssignOut.csv”"""
    print(usage_text)

def main():
    # argument parsing, 입력 예외처리, 알고리즘 실행
    origin_arg = sys.argv[1:]
    
    # argument 개수가 잘못된 경우
    if len(origin_arg) != 12:
        show_usage()
        sys.exit(1)

    # argument에 포함된 공백 및 쉼표 제거
    args = [arg.strip().rstrip(',') for arg in origin_arg]
    
    # argument 변수에 할당
    input_files = args[:7]
    interval1, interval2, algo_type, obj_type = args[7], args[8], args[9], args[10]
    output_file = args[11]
    
    # 변수 유효성 검사
    try:
        int_interval1 = int(interval1)
        int_interval2 = int(interval2)
        int_algo = int(algo_type)
        int_obj = int(obj_type)
        
        if int_interval1 < 0 or int_interval2 < 0:
            raise ValueError("interval1, interval2: 0 이상 정수 입력")
        if int_algo not in (1, 2):
            raise ValueError("algorithm type: 1 또는 2 입력")
        if int_obj not in (1, 2):
            raise ValueError("objective type: 1 또는 2 입력")
    
    except ValueError as ve:
        print(f"{ve}")
        show_usage()
        sys.exit(1)
    
    # 게이트 배정 알고리즘 실행
    try:
        gate_assign = GateAssign(
            input_files = input_files,
            interval1 = interval1,
            interval2 = interval2,
            algo_type = algo_type,
            obj_type = obj_type
        )
        gate_assign.load_flights()
        gate_assign.assign_gates()
        gate_assign.write_output(output_file)
        
    except Exception as ex:
        print("오류가 발생했습니다.")
        traceback.print_exc()
        sys.exit(1)
    
if __name__ == "__main__":
    main()