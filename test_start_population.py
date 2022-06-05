from main.main_ga import generate_starting_population_random, generate_starting_population_zhang
from main.ga_utils.utils import get_instance_info

if __name__ == '__main__':
    instance = get_instance_info(4)

    for i in range(10):
        avg_makespan_random = 0
        avg_makespan_zhang = 0
        print('Working...')
        pop_random = generate_starting_population_random(instance, 1000)
        pop_zhang = generate_starting_population_zhang(instance, 1000)
        for _, _, schedule in pop_random:
            avg_makespan_random += schedule.makespan
        for _, _, schedule in pop_zhang:
            avg_makespan_zhang += schedule.makespan
        print('Avg makespan random:', avg_makespan_random / 1000)
        print('Avg makespan zhang:', avg_makespan_zhang / 1000)
