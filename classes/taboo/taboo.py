from operator import attrgetter
import instance
import graphSchedule

def tabu_solve():
    solution = []
    for i in range(0, 13):
        steps = 1
        steps_improved = 5
        tabu_length = 10
        neighborhood_size = 100
        try:
            s = fjsProblem(i, tabu_length, steps, steps_improved, neighborhood_size)
            m = s.makeSpan
            print((i, m))
            solution.append((i, m))
        except:
            # Currently: store nothing in case no feasible solution is found in the time limit
            pass
    return solution

def fjsProblem(i, tabu_length, steps, steps_improved, neighborhood_size):
    # obatin the initial schedule and makespan from the instance
    inst = instance(i)
    initial = inst.globalSelectionInitialSchedule()

    # set up values for the optimal schedule
    optimal = initial
    optimal_value = initial.makeSpan

    # set up the loop with tracking values and the current schedule
    tabu = []
    current = initial
    s = 0
    s_improved = 0

    # run the loop as long as none of the stopping conditions are met
    while (s < steps and s_improved < steps_improved):
        # get a neighborhood list of schedules and their makespans
        neighborhood = current.getNeighborhoodAssignment(neighborhood_size)

        # find the best move that is not taboo
        move = max(neighborhood, key=attrgetter("makeSpan"))
        while (move in tabu):
            neighborhood.remove(move)
            move = max(neighborhood, key=attrgetter("makeSpan"))

        # run the best found schedule through the scheduling algorithm
        jsp_move = jsProblem(move, tabu_length, steps, steps_improved)

        # put the last move in the tabu list and then make the next one
        tabu.append(current)
        if (len(tabu) > tabu_length):
            tabu.pop(0)

        # update all tracking values if necessary
        current = jsp_move
        s += 1
        s_improved += 1
        if (jsp_move.makeSpan <= optimal_value):
            if (jsp_move.makeSpan < optimal_value):
                s_improved = 0
            optimal_value = jsp_move.makeSpan
            optimal = jsp_move
    
    # return the best obtained schedule
    return optimal
    

def jsProblem(sched: graphSchedule, tabu_length, steps, steps_improved, neighborhood_size):
    # set up values for the optimal schedule
    optimal = sched
    optimal_value = sched.makeSpan

    # set up the loop with tracking values and the current schedule
    tabu = []
    current = sched
    s = 0
    s_improved = 0

    # run the loop as long as none of the stopping conditions are met
    while (s < steps and s_improved < steps_improved):
        # get a neighborhood list of schedules and their makespans
        neighborhood = current.getNeighborhoodSequencing(neighborhood_size)

        # find the best move that is not taboo
        move = max(neighborhood, key=attrgetter("makeSpan"))
        while (move in tabu):
            neighborhood.remove(move)
            move = max(neighborhood, key=attrgetter("makeSpan"))

        # put the last move in the tabu list and then make the next one
        tabu.append(current)
        if (len(tabu) > tabu_length):
            tabu.pop(0)

        # update all tracking values if necessary
        current = move
        s += 1
        s_improved += 1
        if (move.makeSpan <= optimal_value):
            if (move.makeSpan < optimal_value):
                s_improved = 0
            optimal_value = move.makeSpan
            optimal = move

    # return the best obtained schedule
    return optimal