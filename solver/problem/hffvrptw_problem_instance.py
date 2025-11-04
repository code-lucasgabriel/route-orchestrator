from np_solver.core import BaseProblemInstance
from utils import load_instance, save_solution_json, calculate_adjacency_matrix
from solver.problem.hffvrptw_solution import HFFVRPTWSolution
import os
from settings import RESULTS_PATH

class HFFVRPTWProblem(BaseProblemInstance):
    def read_instance(self, filename: str=None):
        """
        Lê os dados da instância, processa e armazena todas as variáveis 
        do problema conforme definido na imagem de referência.
        """
        self.instance = filename
        self.raw_customers, self.raw_fleet = load_instance(self.instance)

        self.n = len(self.raw_customers) 
        
        # V: conjunto de todos os índices de vértices {0, 1, ..., n-1}
        # Onde 0 é o depósito
        self.V = list(range(self.n))
        
        # C: conjunto de índices de clientes {1, ..., n-1}
        self.C = list(range(1, self.n)) 

        coords = [(c['XCOORD.'], c['YCOORD.']) for c in self.raw_customers]
        
        # d_ij: matriz de distâncias euclidianas (calculada)
        self.d = calculate_adjacency_matrix(coords) 

        # A: conjunto de todos os arcos (i, j) onde i, j in V e i != j
        self.A = [(i, j) for i in self.V for j in self.V if i != j]
        
        # t_ij: tempo de viagem (=distância d_ij).
        self.t = self.d 

        # q_i: Lista de demandas para cada vértice i in V (q_0 = 0)
        self.q = [c['DEMAND'] for c in self.raw_customers]
        
        # e_i: lista de inícios de janela (Ready Time) para i in V
        self.e = [c['READY TIME'] for c in self.raw_customers]
        
        # l_i: lista de fins de janela (Due Date) para i in V
        self.l = [c['DUE DATE'] for c in self.raw_customers]
        
        # s_i: lista de tempos de serviço para i in V
        self.s = [c['SERVICE TIME'] for c in self.raw_customers]

        # P: conjunto (lista) de todos os tipos de veículos
        self.P = [f['type'] for f in self.raw_fleet]
        
        # K^p: dicionário {tipo_veiculo -> contagem}
        self.K_p = {f['type']: f['count'] for f in self.raw_fleet}
        
        # Q^p: dicionário {tipo_veiculo -> capacidade}
        self.Q_p = {f['type']: f['capacity'] for f in self.raw_fleet}
        
        # F^p: dicionário {tipo_veiculo -> custo fixo}
        self.F_p = {f['type']: f['fixed_cost'] for f in self.raw_fleet}
        
        # c^p: dicionário {tipo_veiculo -> custo variável}
        self.c_p = {f['type']: f['variable_cost'] for f in self.raw_fleet}

        # K: conjunto (lista) de todos os veículos individuais
        self.K = []
        for p in self.P:
            # Para cada tipo 'p', adiciona 'K_p[p]' veículos
            for i in range(self.K_p[p]):
                self.K.append((p, i)) # Ex: [('A', 0), ('A', 1), ..., ('B', 0), ...]

        # M: Big-M
        self.M = 1e6 

        # print("--- Verificação das Variáveis Geradas ---")
        # print(f"--- V (Vértices) ---: {self.V}\n")
        # print(f"--- C (Clientes) ---: {self.C}\n")
        # print(f"--- P (Tipos de Frota) ---: {self.P}\n")
        # print(f"--- K (Veículos Individuais, ex) ---: {self.K[:3]}... (Total: {len(self.K)})\n")
        # print(f"--- K^p (Contagem por tipo) ---: {self.K_p}\n")
        # print(f"--- Q^p (Capacidade por tipo) ---: {self.Q_p}\n")
        # print(f"--- F^p (Custo Fixo por tipo) ---: {self.F_p}\n")
        # print(f"--- c^p (Custo Variável por tipo) ---: {self.c_p}\n")
        # print(f"--- q (Demandas) ---: {self.q}\n")
        # print(f"--- e (Janela Início) ---: {self.e}\n")
        # print(f"--- l (Janela Fim) ---: {self.l}\n")
        # print(f"--- s (Tempo Serviço) ---: {self.s}\n")
        # print(f"--- d_0j (Distâncias do Depósito) ---: {self.d[0]}\n")
        # print(f"--- M (Big-M) ---: {self.M}\n")
        # print("\n\n\n\n")
    
    def get_instance_name(self):
        return os.path.basename(self.instance)
    
    def get_domain_size(self):
        # TODO define what is the domain size
        return self.fleet
    
    def report_experiment(self, filename: str, sol: HFFVRPTWSolution):
        # TODO verify if it is correct and working
        try:
            save_solution_json(RESULTS_PATH, filename, sol)
        except Exception as e:
            print(f"An error occurred during insertion in the experiment log file for instance {self.get_instance_name()}: {e}")