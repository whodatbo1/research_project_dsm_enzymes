# Research Project Smart Scheduling for DSM


### Smart scheduling for DSM enzyme production line
---

#### Research description

Responsible professor: Dr. Mathijs de Weerdt
Supervisor: Kim van den Houten

#### Background and motivation
The scheduling department of a DSM plant repeatedly solves a complex scheduling question for their enzyme production line. In their plants batches of enzymes are produced according to a product-specific recipe that consists of different unit operations. A batch that is processed first undergoes a set of upstream processes, after which fermentation takes place in large fermenters. After the fermentation, downstream processing unit operations start from which the last 3 steps are visualized in Figure 1.  This is exactly the part of the production process that involves a lot of scheduling rules that require advanced optimisation techniques for improvement. 

In the plant there are different production lines with slightly different operations. There are several criteria to consider when analyzing the quality of a schedule. A common objective in scheduling tasks is to minimize the make span of the schedule. In practice there are multiple objective criteria, e.g. the management of DSM aims to meet customer demand restricted to due dates and an important objective for the plant schedulers is to minimize idle time of the machines. 

![width=80%;height=40%;production line](https://projectforum.tudelft.nl/system/images/files/000/000/604/original/Batch.png?1643732437)

The complexity of the problem raises by the fact that specialized cleaning is required if one machine successively processes two different products. This is a special form of the flexible job shop problem. As an immediate result the complexity of these problems requires creativity in optimization methods. It is extremely important to choose a solution approach that has a good trade-off between optimality and run-time and to describe this in your report. It may be the case that exact methods are not suitable for the complexity size of the problem instances.

#### Activities 
The mixed integer programming formulation of this problem is provided in documents /milp_formulation.pdf. The instances to evaluate on can be found in the folder instances. The gurobi implementation of the MILP can be found in classes/MILP.py.

#### Research questions
Each student should consider the following subquestions, however the projects differ from each other by choosing alternative methods for question 2.

1.	What is the performance of exact solvers for the Mixed Integer Linear Programming formulation of this problem? 
  * Run the MILP. It could be that the solver cannot reach the optimum in a reasonable amount of time.
  * If necessary, adjust the solver settings such that you can find at least feasible solutions for all the problem instances. 
  * Report your findings. Please visualize and describe your results. 

2. What algorithm choice other than MILP is suitable for tackling this optimization problem?
  * Each student chooses an alternative optimisation approach X to tackle this problem. Here you can think of heuristics, local search algorithms, metaheuristics, constraint programming techniques and so on.   
  * Try out different versions of your algorithm by changing hyperparameter settings and report your results
  * Compare with MILP?

3. How can we deal with conflicting objectives in the optimization problem? 
  * Define an objective function that considers different criteria. Be creative, it is not the case that there is only one right answer here. E.g., use the due dates of the given instances to evaluate on tardiness.

4. Are there any bottlenecks in the plant?
  * For example, you could advise to buy one additional machine to achieve better performance.

#### Prerequisites 
This project requires basic knowledge of optimisation techniques and programming skills in Python. Experience in MILP is preferred, but not required.

#### Q&A sessions
To be announced.

#### References
Fattahi, P.,·Saidi, M., Jolai, M.F., 2007. Mathematical modeling and heuristic approaches to flexible jobshop scheduling problems

Özgüven, C., Özbakırb, L., Yavuza, Y., 2010. Mathematical models for job-shop scheduling problems with routing and process plan flexibility. Applied Mathematical Modelling 34 (2010) 1539–1548.

Demir, Y. , İşleyen S.K., 2013. Evaluation of mathematical models for flexible job-shop scheduling problems. Applied Mathematical Modelling Volume 37, Issue 3, 1: (2013) 977-988

Gao, J., Sun, L., Gen, M., 2017. A hybrid genetic and variable neighborhood descent algorithm for flexible job shop scheduling problems. Computers & Operations Research 35 (2008) 2892 – 2907.