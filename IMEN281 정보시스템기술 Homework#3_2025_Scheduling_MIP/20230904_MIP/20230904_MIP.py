import pandas as pd
import time
import os
import sys
import csv
import random
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from docplex.mp.model import Model

# input file 처리 및 데이터 저장, MSP 최적해 계산
class MSP:
    # 초기화
    def __init__(self):
        self.parameters = {}
        self.file_paths = {}
        self.lots = None
        self.machines = None
        self.process_data = {}
        self.solution = []
        self.model = None
        self.BigM = 1000000
        self.vars = {'x': {}, 'y': {}, 't': {}, 'z': {}}

    # 파일로부터 데이터 읽고 저장하는 함수
    def read_data(self, parameter_file):
        try:
            # parameter file 읽기, 디렉토리 경로 설정
            dir_path = os.path.dirname(os.path.abspath(parameter_file))
            parameter_df = pd.read_csv(parameter_file)
            for index, row in parameter_df.iterrows():
                self.parameters[row['Item']] = row['value']
            
            # parameter file 기반으로 파일 경로 설정
            self.file_paths = {
                'lot': os.path.join(dir_path, self.parameters['Lot_file']),
                'machine': os.path.join(dir_path, self.parameters['Machine_file']),
                'setup': os.path.join(dir_path, self.parameters['Setup_processing_file']),
                'output': os.path.join(dir_path, self.parameters['Out_schedule_file']),
                'output_image': os.path.join(dir_path, self.parameters['Output_image_file']),
            }

            # 각 데이터 파일 읽어서 저장
            self.lots = pd.read_csv(self.file_paths['lot'])
            self.machines = pd.read_csv(self.file_paths['machine'])
            setup_df = pd.read_csv(self.file_paths['setup'])
            for index, row in setup_df.iterrows():
                pid = row['product_id']
                mid = row['machine_id']
                self.process_data[(pid, mid)] = {
                    'processing_time': row['unit_processing_time(sec)'],
                    'setup_time': row['setup_time(sec)']
                }

        # 예외 처리
        except Exception as e:
            print("Failed to read data", e)
            sys.exit(1)

    # 제품 id, 설비 id에 따른 setup time 반환
    def get_setup_time(self, product_from, product_to, machine_id):
        if product_from == product_to:
            return 0
        
        key = (product_to, machine_id)
        if key in self.process_data:
            return self.process_data[key]['setup_time']
        
        return self.BigM
    
    # MIP 모델 생성, 변수 및 제약조건 설정, 목적함수 정의 및 수리모델링 문제 풀이
    def solve_MSP(self):
        # MIP 모델 생성
        self.model = Model(name="MIP")

        # 변수, 제약조건, 목적함수 설정
        ##### Sets #####
        # product(lot) set
        set_P = self.lots['lot_id'].tolist()
        
        # machine set
        set_F = self.machines['machine_id'].tolist()
        
        # Helper
        P_pid = dict(zip(self.lots['lot_id'], self.lots['product_id']))
        P_qty = dict(zip(self.lots['lot_id'], self.lots['qty']))
        
        # P_m : 설비 m에서 생산 가능한 제품 집합
        # F_p : 제품 p를 생산할 수 있는 설비 집합
        P_m = {m: [] for m in set_F}
        F_p = {p: [] for p in set_P}
        for p in set_P:
            product_p = P_pid[p]
            for m in set_F:
                if (product_p, m) in self.process_data:
                    P_m[m].append(p)
                    F_p[p].append(m)

        ##### Parameters #####
        # due date of product p
        base = dt.datetime.strptime(str(self.parameters['schedule_date']), "%Y%m%d")
        D_p = {}
        for index, row in self.lots.iterrows():
            due = dt.datetime.strptime(str(row['due_date']), "%Y%m%d")
            D_p[row['lot_id']] = (due - base).total_seconds()
        
        # prioirity of product p
        U_p = dict(zip(self.lots['lot_id'], self.lots['priority']))
        
        # ready time of machine m
        B_m = dict(zip(self.machines['machine_id'], self.machines['start_time(sec)']))
        
        # current product setting at machine m
        A_m = dict(zip(self.machines['machine_id'], self.machines['current_product_id']))
        
        # enougn large number M
        BigM = self.BigM

        ##### Decision Variables #####    
        # start time of product p
        t_p = self.model.continuous_var_dict(set_P)
        self.vars['t'] = t_p

        # lot p가 machine m에서 생산되는 경우 1, 아니면 0
        valid_x_pm = [(p, m) for p in set_P for m in F_p[p]]
        x_pm = self.model.binary_var_dict(valid_x_pm)
        self.vars['x'] = x_pm

        # 설비 m에서 p -> q 순서로 생산되는 경우 1, 아니면 0
        valid_y_pqm = []
        for m in set_F:
            Pm = P_m[m]
            for p in Pm:
                for q in Pm:
                    if p == q: continue
                    else: valid_y_pqm.append((p, q, m))
        y_pqm = self.model.binary_var_dict(valid_y_pqm)
        self.vars['y'] = y_pqm

        # delayed time of product p
        z_p = self.model.continuous_var_dict(set_P)
        self.vars['z'] = z_p

        ##### Constraints #####
        # (1) product p must be assigned
        for p in set_P:
            self.model.add_constraint(self.model.sum(x_pm[p, m] for m in F_p[p]) == 1)
        
        # (2) the first job start time
        for m in set_F:
            Am = A_m[m]
            Bm = B_m[m]
            for p in P_m[m]:
                product_p = P_pid[p]
                S_Am_P_m = self.get_setup_time(Am, product_p, m)
                self.model.add_constraint(t_p[p] >= Bm + S_Am_P_m - BigM * (1 - x_pm[p, m]))

        # (3) - (6)
        for (p, q, m) in valid_y_pqm:
            # (3) machine precedence
            self.model.add_constraint(y_pqm[p, q, m] <= x_pm[p, m])
            # (4) machine precedence
            self.model.add_constraint(y_pqm[p, q, m] <= x_pm[q, m])
            # (5) machine precedence
            if (q, p, m) in y_pqm:
                self.model.add_constraint(y_pqm[p, q, m] + y_pqm[q, p, m] >= x_pm[p, m] + x_pm[q, m] - 1)
            # (6) machine time
            product_p = P_pid[p]
            product_q = P_pid[q]
            T_pm = self.process_data[(product_p, m)]['processing_time'] * P_qty[p]
            S_pqm = self.get_setup_time(product_p, product_q, m)
            self.model.add_constraint(t_p[q] >= t_p[p] + T_pm + S_pqm - BigM * (1 - y_pqm[p, q, m]))

        # (7) delayed time
        for p in set_P:
            Dp = D_p[p]
            product_p = P_pid[p]
            for m in F_p[p]:
                T_pm = self.process_data[(product_p, m)]['processing_time'] * P_qty[p]
                self.model.add_constraint(z_p[p] >= t_p[p] + T_pm - Dp - BigM * (1 - x_pm[p, m]))
        
        # (8) variables domain
        ''' x_pm, y_pqm은 binary variable로 선언하여 이미 처리됨 '''

        ##### Objective Function #####
        self.model.minimize(self.model.sum(U_p[p] * z_p[p] for p in set_P))

        # MIP 문제 풀이
        # 시간 제한 설정
        time_limit = int(self.parameters.get('Time_limit(sec)', 600))
        self.model.parameters.timelimit = time_limit

        solution = self.model.solve(log_output=True)

        # 해를 찾은 경우 solution 저장
        if solution:
            print("Solution found")
            lot_info = self.lots.set_index('lot_id').to_dict('index')
            self.get_solution(lot_info)
            return True
        # 해를 찾지 못한 경우 예외 처리
        else:
            print("No solution found within time limit.")
            return False

    # solution 결과를 딕셔너리 형태로 저장       
    def get_solution(self, lot_info):
        self.solution = []

        # 모든 x_pm (할당 여부) 변수에 대해 탐색
        for (p, m), x_var in self.vars['x'].items():
            # 할당된 경우 해당 lot의 시작 시간, 제품 id, 작업 시간, 용량, 우선순위 저장
            if self.model.solution.get_value(x_var) > 0.5:
                start_var = self.model.solution.get_value(self.vars['t'][p])
                product_p = lot_info[p]['product_id']
                process_unit = self.process_data[(product_p, m)]['processing_time']
                qty = lot_info[p]['qty']
                process_time = process_unit * qty

                # 딕셔너리 형태로 리스트에 저장
                self.solution.append({
                    'lot_id': p,
                    'product_id': product_p,
                    'machine_id': m,
                    'start_time': start_var,
                    'end_time': start_var + process_time,
                    'qty': qty,
                    'priority': lot_info[p]['priority'],
                })
    
    # solution 결과를 파일로 저장
    def save_solution(self, calc_time):
        if not self.solution:
            return
        
        # 결과 파일에 출력할 내용
        sol_rows = [
            ['calculation time', calc_time],
            ['objective value', self.model.solution.objective_value],
            ['number of lots', len(self.solution)]
        ]

        # solution을 dataframe 형태로 변환 후 column 재배치
        sol_df = pd.DataFrame(self.solution)
        cols = ['lot_id', 'product_id', 'qty', 'priority', 'machine_id', 'start_time', 'end_time']
        sol_df = sol_df[cols]

        # csv 파일로 작성
        with open(self.file_paths['output'], 'w', newline='', encoding = 'cp949') as f:
            writer = csv.writer(f)
            for row in sol_rows:
                writer.writerow(row)
            writer.writerow([])
            sol_df.to_csv(f, index=False)
        
        return self.solution, self.file_paths['output']

# 최적해 도출 결과 Gantt Chart로 시각화
class Gantt:
    def __init__(self, solution):
        self.solution = solution

    # 시각화
    def plot_gantt(self, output_image):
        # 도출된 해가 없는 경우 예외 처리
        if not self.solution:
            print("No solution to visualize.")
            return
        
        # solution dataframe 저장 후 duration column 미리 계산
        df = pd.DataFrame(self.solution)
        df['Duration'] = df['end_time'] - df['start_time']

        # product id에 따라 색상 랜덤 지정
        products = df['product_id'].unique()
        color_map = {}
        for p in products:
            r = random.uniform(0.1, 0.9)
            g = random.uniform(0.1, 0.9)
            b = random.uniform(0.1, 0.9)
            color_map[p] = (r, g, b)

        # machine id에 따라 정렬 및 그래프 설정
        machines = sorted(df['machine_id'].unique())
        fig, ax = plt.subplots(figsize=(12, 6 + len(machines) * 0.5))
        y_pos = {machine: i for i, machine in enumerate(machines)}

        # dataframe의 각 row 순회하며 gantt chart 막대 생성
        for _, row in df.iterrows():
            mid = row['machine_id']
            pid = row['product_id']
            start = row['start_time']
            duration = row['Duration']

            bar_color = color_map[pid]

            ax.barh(y_pos[mid], duration, left=start, height=0.6, color=bar_color, edgecolor='black', alpha=0.8)
            ax.text(start + duration/2, y_pos[mid], str(row['lot_id']), va='center', ha='center', color='white', fontsize=8, fontweight='bold')
        
        # x축, y축, Gantt chart 제목 설정
        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels(list(y_pos.keys()))
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Machine")
        ax.set_title("MIP Schedule Gantt Chart")
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)

        # jpg 파일로 저장
        plt.tight_layout()
        plt.savefig(output_image, format='jpg', dpi=300)
        plt.close()

if __name__ == "__main__":
    # 입력 인자가 2개 미만인 경우 사용법 출력
    if len(sys.argv) < 2:
        print("Command: python 20230904_MIP.py <directory>/<parameter_file>")
        print("ex: python 20230904_MIP.py dataset_small_2_5_30/Parameter.csv")
        sys.exit(1)

    # parameter file 저장
    parameter_file = sys.argv[1]

    # MSP 클래스 인스턴스 생성 및 데이터 저장
    msp = MSP()
    msp.read_data(parameter_file)

    # MSP 최적해 도출 및 소요 시간 계산
    start_time = time.time()
    is_solved = msp.solve_MSP()
    end_time = time.time()
    calc_time = end_time - start_time

    # 결과 파일 및 시각화 이미지 생성
    if is_solved:
        sol_data, output_file = msp.save_solution(calc_time)
        output_image = msp.file_paths['output_image']
        gantt = Gantt(sol_data)
        gantt.plot_gantt(output_image)