from solver.hffvrptw import HFFVRPTWSolution, HFFVRPTWProblem, HFFVRPTWEvaluator, HFFVRPTWConstructiveHeuristic
from solver.metaheuristics.ts import ts_tenure5

def run_solver():
    # create and load the problem instance data
    problem = HFFVRPTWProblem()
    problem.read_instance("100_customers/R2_1_04.csv")    
    # create the evaluator and pass the problem instance data
    evaluator = HFFVRPTWEvaluator(problem)

    # ! TEST AND VERIFICATON OF CORRECTEDNESS
    constructor = HFFVRPTWConstructiveHeuristic(problem, evaluator)
    initial_sol = constructor.build()

    print(initial_sol)

    print(evaluator.constraints(initial_sol))
    print(evaluator.evaluate(initial_sol))

    # create the metaheuristic solver, passing the evaluator
    solver = ts_tenure5    

    # run the solver
    print(f"Running the solver for problem {problem.get_instance_name()}")
    best_solution = solver.solve(problem, evaluator, initial_sol)

    # write results
    if best_solution:
        print(f"Best solution found!")
        print(f"Cost: {best_solution.cost}")

if __name__=="__main__":
    run_solver()
