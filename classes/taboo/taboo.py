from operator import attrgetter
from instance import instance
from schedule import schedule
import numpy as np


def fjsProblem(i, tabu_length, steps, steps_improved):
    # obatin the initial schedule and makespan from the instance
    inst = instance(i)
    initial = inst.global_selection_initial_schedule()

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
        neighborhood = current.neigborhoodAssignment()

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
            tabu.pop()

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
    

def jsProblem(sched: schedule, tabu_length, steps, steps_improved):
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
        neighborhood = current.neighborhoodSequencing()

        # find the best move that is not taboo
        move = max(neighborhood, key=attrgetter("makeSpan"))
        while (move in tabu):
            neighborhood.remove(move)
            move = max(neighborhood, key=attrgetter("makeSpan"))

        # put the last move in the tabu list and then make the next one
        tabu.append(current)
        if (len(tabu) > tabu_length):
            tabu.pop()

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