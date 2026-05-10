import math
import time
import sys
import csv
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QObject, QThread, pyqtSignal

# TSP 해를 계산하고 시각화하는 GUI 클래스
class TSP_GUI(QMainWindow):
    
    # 생성자 정의
    def __init__(self):
        super().__init__()
        
        # TSP 해 계산을 위한 변수
        self.data = None
        self.list = []
        self.myds = None
        self.node_id = []
        
        # GUI
        self.initUI()
        self.setGeometry(200, 200, 800, 600)
        
    # GUI 창 설정 함수
    def initUI(self):
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.addWidget(self.canvas)
        self.layout = layout

        self.statusBar()
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        
        # File 메뉴 추가
        filemenu = menubar.addMenu("&File")
        
        # File Open 메뉴 추가
        fileopenAction = QAction('File Open', self)
        fileopenAction.setStatusTip('Open new file')
        fileopenAction.triggered.connect(self.showDialog)
        
        # Exit 메뉴 추가
        exitAction = QAction('Exit', self)
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)
        
        filemenu.addAction(fileopenAction)
        filemenu.addAction(exitAction)
        
        # TSP 메뉴 추가
        TSPmenu = menubar.addMenu("&TSP")
        
        ## Nearest Addition + List 메뉴 추가
        NAlistAction = QAction('Nearest Addition + List', self)
        NAlistAction.setStatusTip('Nearest addition algorithm with python list')
        NAlistAction.triggered.connect(lambda: self.run_algo("NearestAddition_List"))
        
        ## Nearest Addition + My data structure (NumPy Array) 메뉴 추가
        NAmydsAction = QAction('Nearest Addition + NumPy Array', self)
        NAmydsAction.setStatusTip('Nearest addition algorithm with my data structure')
        NAmydsAction.triggered.connect(lambda: self.run_algo("NearestAddition_MyDS"))
        
        ## Greedy 2-OPT + List 메뉴 추가
        G2listAction = QAction('Greedy 2-OPT + List', self)
        G2listAction.setStatusTip('Greedy 2-OPT algorithm with python list')
        G2listAction.triggered.connect(lambda: self.run_algo("Greedy2OPT_List"))
        
        ## Greedy 2-OPT + My data structure (NumPy Array) 메뉴 추가
        G2mydsAction = QAction('Greedy 2-OPT + NumPy Array', self)
        G2mydsAction.setStatusTip('Greedy 2-OPT algorithm with my data structure')
        G2mydsAction.triggered.connect(lambda: self.run_algo("Greedy2OPT_MyDS"))
        
        ## Full 2-OPT + List 메뉴 추가
        F2listAction = QAction('Full 2-OPT + List', self)
        F2listAction.setStatusTip('Full 2-OPT algorithm with python list')
        F2listAction.triggered.connect(lambda: self.run_algo("Full2OPT_List"))
        
        ## Full 2-OPT + My data structure (NumPy Array) 메뉴 추가
        F2mydsAction = QAction('Full 2-OPT + NumPy Array', self)
        F2mydsAction.setStatusTip('Full 2-OPT algorithm with my data structure')
        F2mydsAction.triggered.connect(lambda: self.run_algo("Full2OPT_MyDS"))

        TSPmenu.addAction(NAlistAction)
        TSPmenu.addAction(NAmydsAction)
        TSPmenu.addAction(G2listAction)
        TSPmenu.addAction(G2mydsAction)
        TSPmenu.addAction(F2listAction)
        TSPmenu.addAction(F2mydsAction)
        
    # File open을 위한 showDialog 함수 추가
    def showDialog(self):
        self.data = pd.DataFrame()
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        
        if fname[0]:
            f = open(fname[0], 'r')
            with f:
                df = pd.read_csv(fname[0], encoding = 'cp949')
                self.data = df    
    
    # File에서 데이터를 불러오는 함수
    def load_data(self):
        
        # List를 이용하는 알고리즘에서 사용할 리스트 생성
        self.list = self.data[['latitude', 'longitude']].values.tolist()
        self.node_id = self.data['node_id'].tolist()
        
        # NumPy 배열을 이용하는 알고리즘에서 사용할 matrix 생성
        numpy_array = self.data[['latitude', 'longitude']].to_numpy()
        # calDistance_mtx 함수를 사용하여 좌표 간 거리를 모두 계산하고 저장
        self.myds = self.calDistance_mtx(numpy_array)
        
        return True
    
    # Nearest Addition 알고리즘 (파이썬 리스트 사용)
    def NearestAddition_List(self, ds):
        
        # 전체 노드 수, 방문하지 않은 노드 집합, 경로 초기화
        num_nodes = len(ds)
        path = [0]
        
        # 방문하지 않은 노드의 리스트와 거리 리스트 초기화
        unvisited_list = list(range(1, num_nodes))
        dists = []
        
        # 시작 노드
        lat0, lon0 = ds[0]
        
        # 방문하지 않은 노드와 시작 노드 간의 거리 계산
        for k in unvisited_list:
            lat_k, lon_k = ds[k]
            dist = calDistance(lat_k, lon_k, lat0, lon0)
            dists.append(dist)
        
        # 최소 거리에 있는 노드를 경로에 추가
        while unvisited_list:
            min_dist = float('inf')
            next_node = None
            min_idx = -1
            
            # 거리 리스트 순회하며 최소 거리 업데이트
            for i in range(len(dists)):
                if dists[i] < min_dist:
                    min_dist = dists[i]
                    min_idx = i 
            
            # 최소 거리에 있는 노드 선택
            next_node = unvisited_list[min_idx]
            
            # 해당 노드를 삽입할 위치 탐색
            min_cost = float('inf')
            insert_pos = None
            
            # 삽입할 노드의 좌표
            lat_k, lon_k = ds[next_node]
            
            # 경로를 순회하며 삽입되는 위치별 거리 증가량 계산
            for i in range(len(path)):
                
                # 다음 노드의 인덱스
                j = (i + 1) % len(path)
                
                # 현재 노드와 다음 노드의 좌표
                node_i = path[i]
                node_j = path[j]
                lat_i, lon_i = ds[node_i]
                lat_j, lon_j = ds[node_j]
                
                # 현재 노드 i, 다음 노드 j, 삽입할 노드 k 사이의 거리 계산
                dist_ik = calDistance(lat_i, lon_i, lat_k, lon_k)
                dist_kj = calDistance(lat_k, lon_k, lat_j, lon_j)
                dist_ij = calDistance(lat_i, lon_i, lat_j, lon_j)
                
                # 삽입 시 늘어나는 거리 계산
                cost = dist_ik + dist_kj - dist_ij
                
                # 현재까지의 최소 거리와 비교해 삽입 위치 업데이트
                if cost < min_cost:
                    min_cost = cost
                    insert_pos = i + 1
            
            # 삽입할 노드 k를 탐색한 위치에 삽입하고 방문하지 않은 노드에서 삭제
            path.insert(insert_pos, next_node)
            unvisited_list.remove(next_node)
            dists.remove(min_dist)
            
            # 방문하지 않은 노드가 없으면 종료
            if not unvisited_list:
                break
            
            # 새로 삽입된 노드와 방문하지 않은 노드 간의 거리 업데이트
            for i in range(len(unvisited_list)):
                k = unvisited_list[i]
                dist_lat_k, dist_lon_k = ds[k]
                dist = calDistance(lat_k, lon_k, dist_lat_k, dist_lon_k)
                
                if dist < dists[i]:
                    dists[i] = dist
        
        # 최종 경로 반환     
        return path
            
    # Greedy 2-OPT 알고리즘 (파이썬 리스트 사용)
    def Greedy2OPT_List(self, ds, init_path):
        # 초기 경로 복사 및 노드 수 초기화
        path = init_path.copy()
        num_nodes = len(path)
        
        # 개선 여부 확인 변수
        is_improved = True
        
        # 더 이상 개선점이 없을 때까지 반복 탐색        
        while is_improved:
            # 개선 여부 초기화
            is_improved = False
            
            # 모든 엣지 쌍에 대해 2-OPT 알고리즘 수행
            for i in range(num_nodes - 1):
                for j in range(i + 1, num_nodes):
                    
                    # 인접한 엣지는 건너뜀
                    if j == (i + 1) % num_nodes:
                        continue
                    
                    # 엣지 쌍의 노드 추출 및 좌표 가져와서 각 노드 간의 거리 계산
                    node_a = path[i]
                    node_b = path[(i + 1) % num_nodes]
                    node_c = path[j]
                    node_d = path[(j + 1) % num_nodes]
                
                    lat_a, lon_a = ds[node_a]
                    lat_b, lon_b = ds[node_b]
                    lat_c, lon_c = ds[node_c]
                    lat_d, lon_d = ds[node_d]
                    
                    dist_ab = calDistance(lat_a, lon_a, lat_b, lon_b)
                    dist_cd = calDistance(lat_c, lon_c, lat_d, lon_d)
                    dist_ac = calDistance(lat_a, lon_a, lat_c, lon_c)
                    dist_bd = calDistance(lat_b, lon_b, lat_d, lon_d)
                    
                    # 엣지 교환으로 거리 감소 시 경로 갱신
                    if dist_ac + dist_bd < dist_ab + dist_cd:
                        path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                        is_improved = True
                        break
                
                # 첫번째 개선점 발견 시 다음 엣지 탐색 (Greedy)
                if is_improved:
                    break
            
            # 더 이상 개선점이 없으면 종료
            if not is_improved:
                break
        
        # 최종 경로 반환
        return path
                    
    # Full 2-OPT 알고리즘 (파이썬 리스트 사용)
    def Full2OPT_List(self, ds, init_path):
        # 초기 경로 복사 및 노드 수 초기화
        path = init_path.copy()
        num_nodes = len(path)
        
        # 가능한 모든 개선점 탐색
        while True:
            # 최선의 개선점과 해당하는 엣지 인덱스
            best_imp = 0
            best_i, best_j = -1, -1
            
            # 모든 엣지 쌍에 대해 2-OPT 알고리즘 수행
            for i in range(num_nodes - 1):
                for j in range(i + 1, num_nodes):
                    
                    # 인접한 엣지는 건너뜀
                    if j == (i + 1) % num_nodes:
                        continue
                    
                    # 엣지 쌍의 노드 추출 및 좌표 가져와서 각 노드 간의 거리 계산
                    node_a = path[i]
                    node_b = path[(i + 1) % num_nodes]
                    node_c = path[j]
                    node_d = path[(j + 1) % num_nodes]
                    
                    lat_a, lon_a = ds[node_a]
                    lat_b, lon_b = ds[node_b]
                    lat_c, lon_c = ds[node_c]
                    lat_d, lon_d = ds[node_d]
                    
                    dist_ab = calDistance(lat_a, lon_a, lat_b, lon_b)
                    dist_cd = calDistance(lat_c, lon_c, lat_d, lon_d)
                    dist_ac = calDistance(lat_a, lon_a, lat_c, lon_c)
                    dist_bd = calDistance(lat_b, lon_b, lat_d, lon_d)
                    
                    # 현재 엣지 교환으로 인한 거리 개선량 계산
                    curr_imp = (dist_ab + dist_cd) - (dist_ac + dist_bd)
                    
                    # 이전까지의 개선점과 비교하여 최선의 개선점 업데이트
                    if curr_imp > best_imp:
                        best_imp = curr_imp
                        best_i, best_j = i, j  
            
            # 개선점이 존재하는 경우 경로 갱신 후 다음 탐색
            if best_imp > 0:
                path[best_i + 1:best_j + 1] = path[best_i + 1:best_j + 1][::-1]
            
            # 더 이상 개선점이 없으면 종료
            else:
                break
        
        # 최종 경로 반환
        return path
            
    # Nearest Addition 알고리즘 (NumPy 배열 사용)
    def NearestAddition_MyDS(self, ds):
        # 전체 노드 수, 방문 여부 배열 초기화
        num_nodes = ds.shape[0]
        visited = np.zeros(num_nodes, dtype=bool)
        
        # 시작 경로 설정
        visited[0] = True
        dist0 = ds[0].copy() # 노드 0에서 모든 노드까지의 거리
        dist0[0] = np.inf # 자기자신과의 거리는 무한대로 설정
        second = np.argmin(dist0) # 가장 가까운 노드를 삽입할 노드로 선택
        visited[second] = True
        path = np.array([0, second])
        
        # 방문하지 않은 노드와 시작 경로 사이 거리
        dists = np.minimum(ds[0], ds[second])
        dists[visited] = np.inf
        
        # 모든 노드에 대해 Nearest Addition 알고리즘 수행
        for i in range(num_nodes - 2):
            j = np.roll(path, -1)
            
            # 거리가 가장 작은 노드를 삽입할 노드로 선택 
            k = np.argmin(dists)
            
            # 삽입 시 늘어나는 거리 계산 후 증가량이 가장 작은 위치 탐색
            cost = ds[path, k] + ds[k, j] - ds[path, j]
            insert_pos = np.argmin(cost) + 1
            
            # 경로에 삽입
            path = np.insert(path, insert_pos, k)
            visited[k] = True
            
            # 새로 추가된 노드와 방문하지 않은 노드 사이의 거리 업데이트
            dists = np.minimum(dists, ds[k])
            dists[visited] = np.inf
        
        # 최종 경로 반환 (리스트로 변환 후 반환)
        return path.tolist()

    # Greedy 2-OPT 알고리즘 (NumPy 배열 사용)
    def Greedy2OPT_MyDS(self, ds, init_path):
        
        # 초기 경로, 노드 수, 개선 여부 변수 초기화
        path = np.array(init_path)
        num_nodes = len(path)
        is_improved = True
        
        # 더 이상 개선점이 없을 때까지 반복
        while is_improved:
            is_improved = False
            
            # 모든 엣지 쌍에 대해 2-OPT 알고리즘 수행
            for i in range(num_nodes - 2):
                
                # 현재 엣지 a-b
                node_a = path[i]
                node_b = path[(i + 1) % num_nodes]
                
                # 교환 가능한 엣지 후보
                j_range = np.arange(i + 2, num_nodes)
                node_c = path[j_range]
                node_d = path[(j_range + 1) % num_nodes]
                
                # 엣지 교환 전후 거리 차이 계산
                cost = (ds[node_a, node_b] + ds[node_c, node_d]) - (ds[node_a, node_c] + ds[node_b, node_d])
                improved_idx = np.where(cost > 0)[0]
                
                # 첫번째 개선점 발견시 엣지 교환 및 다음 개선점 탐색
                if improved_idx.size > 0:
                    j = j_range[improved_idx[0]]
                    path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                    is_improved = True
                    break
        
        # 최종 경로 반환 (리스트 변환)
        return path.tolist()
    
    # Full 2-OPT 알고리즘 (NumPy 배열 사용)
    def Full2OPT_MyDS(self, ds, init_path):
        # 초기 경로, 노드 수, 개선 여부 변수 초기화
        path = np.array(init_path)
        num_nodes = len(path)
        is_improved = True
        
        # 더 이상 개선점이 없을 때까지 탐색
        while is_improved:
            is_improved = False
            
            # 가능한 모든 엣지 쌍 인덱스 생성
            i, j = np.triu_indices(num_nodes, 2)
            
            # 각 엣지 쌍의 노드 추출
            node_a = path[i]
            node_b = path[(i + 1) % num_nodes]
            node_c = path[j]
            node_d = path[(j + 1) % num_nodes]
            
            # 엣지 교환 전후 거리 차이 계산
            cost = (ds[node_a, node_b] + ds[node_c, node_d]) - (ds[node_a, node_c] + ds[node_b, node_d])
            
            # 최선의 개선점 선택
            best_idx = np.argmax(cost)
            
            # 개선점이 있는 경우 경로 업데이트 후 다음 탐색
            if cost[best_idx] > 0:
                i_best = i[best_idx]
                j_best = j[best_idx]
                path[i_best + 1:j_best + 1] = path[i_best + 1:j_best + 1][::-1]
                is_improved = True
            
        # 최종 경로 반환
        return path.tolist()
    
    # 알고리즘 실행 함수
    def run_algo(self, algo):
        if not self.load_data():
            return
        
        na_func = None
        opt_func = None
        ds = None
        algo_name = ""
        
        # 각 알고리즘 종류별 실행 함수와 자료구조, 알고리즘 이름 설정
        if algo == "NearestAddition_List":
            na_func = self.NearestAddition_List
            ds = self.list
            algo_name = "Nearest Addition + List"
            
        elif algo == "NearestAddition_MyDS":
            na_func = self.NearestAddition_MyDS
            ds = self.myds
            algo_name = "Nearest Addition + NumPy Array"
            
        elif algo == "Greedy2OPT_List":
            na_func = self.NearestAddition_List
            opt_func = self.Greedy2OPT_List
            ds = self.list
            algo_name = "Greedy 2-OPT + List"
        
        elif algo == "Greedy2OPT_MyDS":
            na_func = self.NearestAddition_MyDS
            opt_func = self.Greedy2OPT_MyDS
            ds = self.myds
            algo_name = "Greedy 2-OPT + NumPy Array"
            
        elif algo == "Full2OPT_List":
            na_func = self.NearestAddition_List
            opt_func = self.Full2OPT_List
            ds = self.list
            algo_name = "Full 2-OPT + List"
            
        elif algo == "Full2OPT_MyDS":
            na_func = self.NearestAddition_MyDS
            opt_func = self.Full2OPT_MyDS
            ds = self.myds
            algo_name = "Full 2-OPT + NumPy Array"
        
        # GUI 응답없음 방지를 위해 별도 스레드로 분리하여 각 알고리즘 함수 실행
        self.thread = QThread()
        self.worker = Worker(na_func, opt_func, ds, algo_name, coord=self.list)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_algo_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()
    
    # 알고리즘 함수 실행 종료 시 호출하여 경로 시각화 및 결과 파일 생성
    def on_algo_finished(self, path, algo, total_dist, runtime):
        self.draw_path(path, algo, total_dist, runtime) 
        self.write_output(path, algo, total_dist, runtime)      

    # 경로 시각화 함수
    def draw_path(self, path, algo, total_dist, runtime):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        x_coords = [coord[1] for coord in self.list]
        y_coords = [coord[0] for coord in self.list]
        ax.scatter(x_coords, y_coords, c='green', marker='o')
        
        path_x = [self.list[i][1] for i in path]
        path_y = [self.list[i][0] for i in path]
        
        path_x.append(self.list[path[0]][1])
        path_y.append(self.list[path[0]][0])
        
        ax.plot(path_x, path_y, color = 'skyblue')
        
        # x, y축 라벨 설정
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # 제목에 알고리즘 이름, 전체거리, 실행 시간 표시
        text = (f"Algorithm: {algo}\n\
                       Total Distance: {total_dist:.2f} km\n\
                       Running Time: {runtime:.2f} ms")
        
        ax.set_title(text)
        
        self.canvas.draw()
    
    # 결과 파일 작성
    def write_output(self, path, algo, total_dist, runtime):
        num_nodes = len(self.list)
        filename = (f"TSP_output{num_nodes}.csv")
        
        output_path = path + [path[0]]
        path_node_ids = [self.node_id[i] for i in output_path]
        path_str = ", ".join(map(str, path_node_ids))
        
        rows = [
            ["알고리즘", algo],
            ["계산시간", f"{runtime:.2f} ms"],
            ["전체길이", f"{total_dist:.2f} km"],
            ["Tour", path_str]
        ]
        
        with open(filename, 'w', newline = '', encoding = 'cp949') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    
    # 각 노드간의 거리를 km 단위로 사전에 행렬로 계산해두기 위한 함수
    def calDistance_mtx(self, coord):
        radius = 6371.0
        
        rlat = np.radians(coord[:, 0])
        rlon = np.radians(coord[:, 1])
        
        dlat = rlat[:, None] - rlat
        dlon = rlon[:, None] - rlon
        
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(rlat)[:, None] * np.cos(rlat) *
             np.sin(dlon / 2) ** 2)
        
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance_mtx = radius * c
        return distance_mtx

# GUI 응답없음 방지를 위해 TSP 계산 함수를 별도로 실행하기 위한 클래스
class Worker(QObject):
    finished = pyqtSignal(list, str, float, float)

    # 실행 함수, 자료구조, 알고리즘 이름, 좌표 초기화
    def __init__(self, na_func, opt_func, ds, algo_name, coord):
        super().__init__()
        self.na_func = na_func
        self.opt_func = opt_func
        self.ds = ds
        self.algo_name = algo_name
        self.coord = coord
    
    # 실행 및 실행 시간 계산   
    def run(self):
        start_time = time.time()
        
        # Nearest addition 알고리즘인 경우
        path = self.na_func(self.ds)
        
        # OPT 알고리즘인 경우
        if self.opt_func:
            path = self.opt_func(self.ds, path)
        
        end_time = time.time()
        
        # 실행시간, 전체 거리 계산
        runtime = (end_time - start_time) * 1000
        total_dist = total_distance(path, self.coord)
        self.finished.emit(path, self.algo_name, total_dist, runtime)

# 거리를 km단위로 변환하는 함수
def calDistance(lat1, lon1, lat2, lon2):
    radius = 6371.0 

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)

    a = (math.sin(dLat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    dDistance = radius * c

    return dDistance

# 전체 거리를 누적 계산하는 함수
def total_distance(path, coord):
        total_dist = 0
        
        for i in range(len(path)):
            start_node = path[i]
            end_node = path[(i + 1) % len(path)]
            
            lat1, lon1 = coord[start_node]
            lat2, lon2 = coord[end_node]
            
            dist = calDistance(lat1, lon1, lat2, lon2)
            total_dist += dist
        
        return total_dist

# 메인 실행 함수
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = TSP_GUI()
    ex.show()
    sys.exit(app.exec_())