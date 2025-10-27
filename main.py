from solver.hffvrptw import HFFVRPTWProblem, HFFVRPTWEvaluator
from np_solver.core import BaseSolution
# from solver.metaheuristics import SolverTS

def run_solver():
    # create and load the problem instance data
    problem = HFFVRPTWProblem()
    problem.read_instance("100_customers/R2_1_04.csv")    
    # create the evaluator and pass the problem instance data
    evaluator = HFFVRPTWEvaluator(problem)

    # ! TEST AND VERIFICATON OF CORRECTEDNESS
    sol = BaseSolution()
    sol.extend([
        [0, 1, 2, 0],
        [0, 3, 0],     
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ])

    print(evaluator.constraints(sol))
    print(evaluator.evaluate(sol))

    # create the metaheuristic solver, passing the evaluator
    # solver = SolverTS(
    #     evaluator=evaluator,
    #     generations=10000,
    #     pop_size=100,
    #     mutation_rate=0.01
    # )

    # run the solver
    # print(f"Running the solver for problem {problem.get_instance_name()}")
    # best_solution = solver.solver()

    # write results
    # if best_solution:
        # print(f"Best solution found!")
        # print(f"Cost: {best_solution.cost}")
        # problem.write_results("results/{instance}", best_solution)

if __name__=="__main__":
    run_solver()
