nr_machines = 9
nr_jobs = 30
orders = {0: {'product': 'enzyme0', 'due': 55}, 1: {'product': 'enzyme0', 'due': 36}, 2: {'product': 'enzyme0', 'due': 53}, 3: {'product': 'enzyme0', 'due': 18}, 4: {'product': 'enzyme0', 'due': 39}, 5: {'product': 'enzyme3', 'due': 54}, 6: {'product': 'enzyme3', 'due': 50}, 7: {'product': 'enzyme3', 'due': 59}, 8: {'product': 'enzyme3', 'due': 43}, 9: {'product': 'enzyme3', 'due': 55}, 10: {'product': 'enzyme1', 'due': 13}, 11: {'product': 'enzyme1', 'due': 29}, 12: {'product': 'enzyme1', 'due': 28}, 13: {'product': 'enzyme1', 'due': 22}, 14: {'product': 'enzyme1', 'due': 44}, 15: {'product': 'enzyme2', 'due': 60}, 16: {'product': 'enzyme2', 'due': 46}, 17: {'product': 'enzyme2', 'due': 39}, 18: {'product': 'enzyme2', 'due': 28}, 19: {'product': 'enzyme2', 'due': 31}, 20: {'product': 'enzyme5', 'due': 39}, 21: {'product': 'enzyme5', 'due': 28}, 22: {'product': 'enzyme5', 'due': 42}, 23: {'product': 'enzyme5', 'due': 22}, 24: {'product': 'enzyme5', 'due': 53}, 25: {'product': 'enzyme1', 'due': 15}, 26: {'product': 'enzyme1', 'due': 21}, 27: {'product': 'enzyme1', 'due': 19}, 28: {'product': 'enzyme1', 'due': 55}, 29: {'product': 'enzyme1', 'due': 47}}
machines = [0, 1, 2, 3, 4, 5, 6, 7, 8]
jobs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
operations = {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2], 4: [0, 1, 2], 5: [0, 1, 2], 6: [0, 1, 2], 7: [0, 1, 2], 8: [0, 1, 2], 9: [0, 1, 2], 10: [0, 1], 11: [0, 1], 12: [0, 1], 13: [0, 1], 14: [0, 1], 15: [0, 1], 16: [0, 1], 17: [0, 1], 18: [0, 1], 19: [0, 1], 20: [0, 1], 21: [0, 1], 22: [0, 1], 23: [0, 1], 24: [0, 1], 25: [0, 1], 26: [0, 1], 27: [0, 1], 28: [0, 1], 29: [0, 1]}
machineAlternatives = {(0, 0): [0, 1, 2], (0, 1): [3, 4, 5, 6], (1, 0): [0, 1, 2], (1, 1): [3, 4, 5, 6], (1, 2): [7, 8], (2, 0): [0, 1, 2], (2, 1): [3, 4, 5, 6], (2, 2): [7, 8], (3, 0): [0, 1, 2], (3, 1): [3, 4, 5, 6], (3, 2): [7, 8], (4, 0): [0, 1, 2], (4, 1): [3, 4, 5, 6], (5, 0): [0, 1, 2], (5, 1): [3, 4, 5, 6], (0, 2): [7, 8], (4, 2): [7, 8], (5, 2): [7, 8], (6, 0): [0, 1, 2], (6, 1): [3, 4, 5, 6], (7, 0): [0, 1, 2], (7, 1): [3, 4, 5, 6], (8, 0): [0, 1, 2], (8, 1): [3, 4, 5, 6], (9, 0): [0, 1, 2], (9, 1): [3, 4, 5, 6], (10, 0): [0, 1, 2], (10, 1): [3, 4, 5, 6], (11, 0): [0, 1, 2], (11, 1): [3, 4, 5, 6], (8, 2): [7, 8], (9, 2): [7, 8], (6, 2): [7, 8], (7, 2): [7, 8], (10, 2): [7, 8], (11, 2): [7, 8], (12, 0): [0, 1, 2], (12, 1): [3, 4, 5, 6], (12, 2): [7, 8], (13, 0): [0, 1, 2], (13, 1): [3, 4, 5, 6], (13, 2): [7, 8], (14, 0): [0, 1, 2], (14, 1): [3, 4, 5, 6], (14, 2): [7, 8], (15, 0): [3, 4, 5, 6], (15, 1): [7, 8], (16, 0): [3, 4, 5, 6], (16, 1): [7, 8], (17, 0): [3, 4, 5, 6], (17, 1): [7, 8], (15, 2): [7, 8], (16, 2): [7, 8], (17, 2): [7, 8], (18, 0): [3, 4, 5, 6], (18, 1): [7, 8], (18, 2): [7, 8], (19, 0): [3, 4, 5, 6], (19, 1): [7, 8], (19, 2): [7, 8], (20, 0): [3, 4, 5, 6], (20, 1): [7, 8], (20, 2): [7, 8], (21, 0): [3, 4, 5, 6], (21, 1): [7, 8], (21, 2): [7, 8], (22, 0): [3, 4, 5, 6], (22, 1): [7, 8], (22, 2): [7, 8], (23, 0): [3, 4, 5, 6], (23, 1): [7, 8], (23, 2): [7, 8], (24, 0): [3, 4, 5, 6], (24, 1): [7, 8], (25, 0): [0, 1, 2], (25, 1): [3, 4, 5, 6], (26, 0): [0, 1, 2], (26, 1): [3, 4, 5, 6], (27, 0): [0, 1, 2], (27, 1): [3, 4, 5, 6], (28, 0): [0, 1, 2], (28, 1): [3, 4, 5, 6], (29, 0): [0, 1, 2], (29, 1): [3, 4, 5, 6]}
processingTimes = {(0, 0, 3): 8, (0, 0, 4): 8, (0, 0, 5): 8, (0, 0, 6): 8, (0, 1, 7): 3, (0, 1, 8): 3, (1, 0, 0): 8, (1, 0, 1): 8, (1, 0, 2): 8, (1, 1, 3): 4, (1, 1, 4): 4, (1, 1, 5): 4, (1, 1, 6): 4, (1, 2, 7): 4, (1, 2, 8): 4, (2, 0, 0): 8, (2, 0, 1): 8, (2, 0, 2): 8, (2, 1, 3): 4, (2, 1, 4): 4, (2, 1, 5): 4, (2, 1, 6): 4, (2, 2, 7): 4, (2, 2, 8): 4, (3, 0, 0): 8, (3, 0, 1): 8, (3, 0, 2): 8, (3, 1, 3): 4, (3, 1, 4): 4, (3, 1, 5): 4, (3, 1, 6): 4, (3, 2, 7): 4, (3, 2, 8): 4, (4, 0, 0): 8, (4, 0, 1): 8, (4, 0, 2): 8, (4, 1, 3): 4, (4, 1, 4): 4, (4, 1, 5): 4, (4, 1, 6): 4, (5, 0, 3): 3, (5, 0, 4): 3, (5, 0, 5): 3, (5, 0, 6): 3, (5, 1, 7): 3, (5, 1, 8): 3, (0, 0, 0): 8, (0, 0, 1): 8, (0, 0, 2): 8, (0, 1, 3): 4, (0, 1, 4): 4, (0, 1, 5): 4, (0, 1, 6): 4, (0, 2, 7): 4, (0, 2, 8): 4, (2, 0, 3): 8, (2, 0, 4): 8, (2, 0, 5): 8, (2, 0, 6): 8, (2, 1, 7): 3, (2, 1, 8): 3, (4, 2, 7): 4, (4, 2, 8): 4, (5, 0, 0): 4, (5, 0, 1): 4, (5, 0, 2): 4, (5, 1, 3): 6, (5, 1, 4): 6, (5, 1, 5): 6, (5, 1, 6): 6, (4, 0, 3): 3, (4, 0, 4): 3, (4, 0, 5): 3, (4, 0, 6): 3, (4, 1, 7): 3, (4, 1, 8): 3, (1, 0, 3): 8, (1, 0, 4): 8, (1, 0, 5): 8, (1, 0, 6): 8, (1, 1, 7): 3, (1, 1, 8): 3, (3, 0, 3): 8, (3, 0, 4): 8, (3, 0, 5): 8, (3, 0, 6): 8, (3, 1, 7): 3, (3, 1, 8): 3, (5, 2, 7): 6, (5, 2, 8): 6, (6, 0, 0): 4, (6, 0, 1): 4, (6, 0, 2): 4, (6, 1, 3): 6, (6, 1, 4): 6, (6, 1, 5): 6, (6, 1, 6): 6, (7, 0, 0): 4, (7, 0, 1): 4, (7, 0, 2): 4, (7, 1, 3): 6, (7, 1, 4): 6, (7, 1, 5): 6, (7, 1, 6): 6, (8, 0, 3): 8, (8, 0, 4): 8, (8, 0, 5): 8, (8, 0, 6): 8, (8, 1, 7): 3, (8, 1, 8): 3, (9, 0, 3): 8, (9, 0, 4): 8, (9, 0, 5): 8, (9, 0, 6): 8, (9, 1, 7): 3, (9, 1, 8): 3, (10, 0, 0): 3, (10, 0, 1): 3, (10, 0, 2): 3, (10, 1, 3): 2, (10, 1, 4): 2, (10, 1, 5): 2, (10, 1, 6): 2, (11, 0, 0): 3, (11, 0, 1): 3, (11, 0, 2): 3, (11, 1, 3): 2, (11, 1, 4): 2, (11, 1, 5): 2, (11, 1, 6): 2, (8, 0, 0): 4, (8, 0, 1): 4, (8, 0, 2): 4, (8, 1, 3): 6, (8, 1, 4): 6, (8, 1, 5): 6, (8, 1, 6): 6, (8, 2, 7): 6, (8, 2, 8): 6, (9, 0, 0): 4, (9, 0, 1): 4, (9, 0, 2): 4, (9, 1, 3): 6, (9, 1, 4): 6, (9, 1, 5): 6, (9, 1, 6): 6, (9, 2, 7): 6, (9, 2, 8): 6, (10, 0, 3): 8, (10, 0, 4): 8, (10, 0, 5): 8, (10, 0, 6): 8, (10, 1, 7): 3, (10, 1, 8): 3, (11, 0, 3): 8, (11, 0, 4): 8, (11, 0, 5): 8, (11, 0, 6): 8, (11, 1, 7): 3, (11, 1, 8): 3, (6, 2, 7): 6, (6, 2, 8): 6, (7, 2, 7): 6, (7, 2, 8): 6, (10, 2, 7): 7, (10, 2, 8): 7, (11, 2, 7): 7, (11, 2, 8): 7, (6, 0, 3): 3, (6, 0, 4): 3, (6, 0, 5): 3, (6, 0, 6): 3, (6, 1, 7): 3, (6, 1, 8): 3, (7, 0, 3): 3, (7, 0, 4): 3, (7, 0, 5): 3, (7, 0, 6): 3, (7, 1, 7): 3, (7, 1, 8): 3, (12, 0, 0): 3, (12, 0, 1): 3, (12, 0, 2): 3, (12, 1, 3): 2, (12, 1, 4): 2, (12, 1, 5): 2, (12, 1, 6): 2, (12, 2, 7): 6, (12, 2, 8): 6, (13, 0, 0): 3, (13, 0, 1): 3, (13, 0, 2): 3, (13, 1, 3): 2, (13, 1, 4): 2, (13, 1, 5): 2, (13, 1, 6): 2, (13, 2, 7): 6, (13, 2, 8): 6, (14, 0, 0): 3, (14, 0, 1): 3, (14, 0, 2): 3, (14, 1, 3): 2, (14, 1, 4): 2, (14, 1, 5): 2, (14, 1, 6): 2, (14, 2, 7): 6, (14, 2, 8): 6, (15, 0, 3): 3, (15, 0, 4): 3, (15, 0, 5): 3, (15, 0, 6): 3, (15, 1, 7): 3, (15, 1, 8): 3, (16, 0, 3): 3, (16, 0, 4): 3, (16, 0, 5): 3, (16, 0, 6): 3, (16, 1, 7): 3, (16, 1, 8): 3, (17, 0, 3): 3, (17, 0, 4): 3, (17, 0, 5): 3, (17, 0, 6): 3, (17, 1, 7): 3, (17, 1, 8): 3, (15, 0, 0): 4, (15, 0, 1): 4, (15, 0, 2): 4, (15, 1, 3): 6, (15, 1, 4): 6, (15, 1, 5): 6, (15, 1, 6): 6, (15, 2, 7): 6, (15, 2, 8): 6, (16, 0, 0): 5, (16, 0, 1): 5, (16, 0, 2): 5, (16, 1, 3): 4, (16, 1, 4): 4, (16, 1, 5): 4, (16, 1, 6): 4, (16, 2, 7): 7, (16, 2, 8): 7, (17, 0, 0): 5, (17, 0, 1): 5, (17, 0, 2): 5, (17, 1, 3): 4, (17, 1, 4): 4, (17, 1, 5): 4, (17, 1, 6): 4, (17, 2, 7): 7, (17, 2, 8): 7, (12, 0, 3): 3, (12, 0, 4): 3, (12, 0, 5): 3, (12, 0, 6): 3, (12, 1, 7): 3, (12, 1, 8): 3, (13, 0, 3): 3, (13, 0, 4): 3, (13, 0, 5): 3, (13, 0, 6): 3, (13, 1, 7): 3, (13, 1, 8): 3, (14, 0, 3): 3, (14, 0, 4): 3, (14, 0, 5): 3, (14, 0, 6): 3, (14, 1, 7): 3, (14, 1, 8): 3, (18, 0, 0): 5, (18, 0, 1): 5, (18, 0, 2): 5, (18, 1, 3): 4, (18, 1, 4): 4, (18, 1, 5): 4, (18, 1, 6): 4, (18, 2, 7): 7, (18, 2, 8): 7, (19, 0, 0): 5, (19, 0, 1): 5, (19, 0, 2): 5, (19, 1, 3): 4, (19, 1, 4): 4, (19, 1, 5): 4, (19, 1, 6): 4, (19, 2, 7): 7, (19, 2, 8): 7, (20, 0, 0): 4, (20, 0, 1): 4, (20, 0, 2): 4, (20, 1, 3): 6, (20, 1, 4): 6, (20, 1, 5): 6, (20, 1, 6): 6, (20, 2, 7): 6, (20, 2, 8): 6, (21, 0, 0): 4, (21, 0, 1): 4, (21, 0, 2): 4, (21, 1, 3): 6, (21, 1, 4): 6, (21, 1, 5): 6, (21, 1, 6): 6, (21, 2, 7): 6, (21, 2, 8): 6, (22, 0, 0): 4, (22, 0, 1): 4, (22, 0, 2): 4, (22, 1, 3): 6, (22, 1, 4): 6, (22, 1, 5): 6, (22, 1, 6): 6, (22, 2, 7): 6, (22, 2, 8): 6, (23, 0, 0): 4, (23, 0, 1): 4, (23, 0, 2): 4, (23, 1, 3): 6, (23, 1, 4): 6, (23, 1, 5): 6, (23, 1, 6): 6, (23, 2, 7): 6, (23, 2, 8): 6, (20, 0, 3): 8, (20, 0, 4): 8, (20, 0, 5): 8, (20, 0, 6): 8, (20, 1, 7): 3, (20, 1, 8): 3, (21, 0, 3): 8, (21, 0, 4): 8, (21, 0, 5): 8, (21, 0, 6): 8, (21, 1, 7): 3, (21, 1, 8): 3, (22, 0, 3): 8, (22, 0, 4): 8, (22, 0, 5): 8, (22, 0, 6): 8, (22, 1, 7): 3, (22, 1, 8): 3, (23, 0, 3): 8, (23, 0, 4): 8, (23, 0, 5): 8, (23, 0, 6): 8, (23, 1, 7): 3, (23, 1, 8): 3, (18, 0, 3): 3, (18, 0, 4): 3, (18, 0, 5): 3, (18, 0, 6): 3, (18, 1, 7): 3, (18, 1, 8): 3, (19, 0, 3): 3, (19, 0, 4): 3, (19, 0, 5): 3, (19, 0, 6): 3, (19, 1, 7): 3, (19, 1, 8): 3, (24, 0, 3): 8, (24, 0, 4): 8, (24, 0, 5): 8, (24, 0, 6): 8, (24, 1, 7): 3, (24, 1, 8): 3, (25, 0, 0): 3, (25, 0, 1): 3, (25, 0, 2): 3, (25, 1, 3): 2, (25, 1, 4): 2, (25, 1, 5): 2, (25, 1, 6): 2, (26, 0, 0): 3, (26, 0, 1): 3, (26, 0, 2): 3, (26, 1, 3): 2, (26, 1, 4): 2, (26, 1, 5): 2, (26, 1, 6): 2, (27, 0, 0): 3, (27, 0, 1): 3, (27, 0, 2): 3, (27, 1, 3): 2, (27, 1, 4): 2, (27, 1, 5): 2, (27, 1, 6): 2, (28, 0, 0): 3, (28, 0, 1): 3, (28, 0, 2): 3, (28, 1, 3): 2, (28, 1, 4): 2, (28, 1, 5): 2, (28, 1, 6): 2, (29, 0, 0): 3, (29, 0, 1): 3, (29, 0, 2): 3, (29, 1, 3): 2, (29, 1, 4): 2, (29, 1, 5): 2, (29, 1, 6): 2}
changeOvers = {(0, 'enzyme0', 'enzyme0'): 0, (0, 'enzyme0', 'enzyme1'): 3, (0, 'enzyme0', 'enzyme2'): 1, (0, 'enzyme0', 'enzyme3'): 2, (0, 'enzyme0', 'enzyme4'): 2, (0, 'enzyme0', 'enzyme5'): 3, (0, 'enzyme1', 'enzyme0'): 1, (0, 'enzyme1', 'enzyme1'): 0, (0, 'enzyme1', 'enzyme2'): 1, (0, 'enzyme1', 'enzyme3'): 4, (0, 'enzyme1', 'enzyme4'): 3, (0, 'enzyme1', 'enzyme5'): 1, (0, 'enzyme2', 'enzyme0'): 1, (0, 'enzyme2', 'enzyme1'): 1, (0, 'enzyme2', 'enzyme2'): 0, (0, 'enzyme2', 'enzyme3'): 1, (0, 'enzyme2', 'enzyme4'): 3, (0, 'enzyme2', 'enzyme5'): 2, (0, 'enzyme3', 'enzyme0'): 2, (0, 'enzyme3', 'enzyme1'): 2, (0, 'enzyme3', 'enzyme2'): 3, (0, 'enzyme3', 'enzyme3'): 0, (0, 'enzyme3', 'enzyme4'): 4, (0, 'enzyme3', 'enzyme5'): 2, (0, 'enzyme4', 'enzyme0'): 2, (0, 'enzyme4', 'enzyme1'): 4, (0, 'enzyme4', 'enzyme2'): 3, (0, 'enzyme4', 'enzyme3'): 2, (0, 'enzyme4', 'enzyme4'): 0, (0, 'enzyme4', 'enzyme5'): 4, (0, 'enzyme5', 'enzyme0'): 1, (0, 'enzyme5', 'enzyme1'): 3, (0, 'enzyme5', 'enzyme2'): 1, (0, 'enzyme5', 'enzyme3'): 4, (0, 'enzyme5', 'enzyme4'): 3, (0, 'enzyme5', 'enzyme5'): 0, (1, 'enzyme0', 'enzyme0'): 0, (1, 'enzyme0', 'enzyme1'): 4, (1, 'enzyme0', 'enzyme2'): 1, (1, 'enzyme0', 'enzyme3'): 3, (1, 'enzyme0', 'enzyme4'): 3, (1, 'enzyme0', 'enzyme5'): 1, (1, 'enzyme1', 'enzyme0'): 1, (1, 'enzyme1', 'enzyme1'): 0, (1, 'enzyme1', 'enzyme2'): 2, (1, 'enzyme1', 'enzyme3'): 4, (1, 'enzyme1', 'enzyme4'): 3, (1, 'enzyme1', 'enzyme5'): 4, (1, 'enzyme2', 'enzyme0'): 1, (1, 'enzyme2', 'enzyme1'): 2, (1, 'enzyme2', 'enzyme2'): 0, (1, 'enzyme2', 'enzyme3'): 4, (1, 'enzyme2', 'enzyme4'): 3, (1, 'enzyme2', 'enzyme5'): 3, (1, 'enzyme3', 'enzyme0'): 4, (1, 'enzyme3', 'enzyme1'): 1, (1, 'enzyme3', 'enzyme2'): 1, (1, 'enzyme3', 'enzyme3'): 0, (1, 'enzyme3', 'enzyme4'): 1, (1, 'enzyme3', 'enzyme5'): 2, (1, 'enzyme4', 'enzyme0'): 3, (1, 'enzyme4', 'enzyme1'): 4, (1, 'enzyme4', 'enzyme2'): 4, (1, 'enzyme4', 'enzyme3'): 1, (1, 'enzyme4', 'enzyme4'): 0, (1, 'enzyme4', 'enzyme5'): 4, (1, 'enzyme5', 'enzyme0'): 1, (1, 'enzyme5', 'enzyme1'): 1, (1, 'enzyme5', 'enzyme2'): 1, (1, 'enzyme5', 'enzyme3'): 2, (1, 'enzyme5', 'enzyme4'): 1, (1, 'enzyme5', 'enzyme5'): 0, (2, 'enzyme0', 'enzyme0'): 0, (2, 'enzyme0', 'enzyme1'): 1, (2, 'enzyme0', 'enzyme2'): 3, (2, 'enzyme0', 'enzyme3'): 2, (2, 'enzyme0', 'enzyme4'): 3, (2, 'enzyme0', 'enzyme5'): 2, (2, 'enzyme1', 'enzyme0'): 4, (2, 'enzyme1', 'enzyme1'): 0, (2, 'enzyme1', 'enzyme2'): 1, (2, 'enzyme1', 'enzyme3'): 2, (2, 'enzyme1', 'enzyme4'): 3, (2, 'enzyme1', 'enzyme5'): 3, (2, 'enzyme2', 'enzyme0'): 3, (2, 'enzyme2', 'enzyme1'): 1, (2, 'enzyme2', 'enzyme2'): 0, (2, 'enzyme2', 'enzyme3'): 3, (2, 'enzyme2', 'enzyme4'): 4, (2, 'enzyme2', 'enzyme5'): 4, (2, 'enzyme3', 'enzyme0'): 3, (2, 'enzyme3', 'enzyme1'): 1, (2, 'enzyme3', 'enzyme2'): 2, (2, 'enzyme3', 'enzyme3'): 0, (2, 'enzyme3', 'enzyme4'): 3, (2, 'enzyme3', 'enzyme5'): 3, (2, 'enzyme4', 'enzyme0'): 1, (2, 'enzyme4', 'enzyme1'): 4, (2, 'enzyme4', 'enzyme2'): 1, (2, 'enzyme4', 'enzyme3'): 2, (2, 'enzyme4', 'enzyme4'): 0, (2, 'enzyme4', 'enzyme5'): 4, (2, 'enzyme5', 'enzyme0'): 4, (2, 'enzyme5', 'enzyme1'): 2, (2, 'enzyme5', 'enzyme2'): 1, (2, 'enzyme5', 'enzyme3'): 3, (2, 'enzyme5', 'enzyme4'): 3, (2, 'enzyme5', 'enzyme5'): 0, (3, 'enzyme0', 'enzyme0'): 0, (3, 'enzyme0', 'enzyme1'): 3, (3, 'enzyme0', 'enzyme2'): 3, (3, 'enzyme0', 'enzyme3'): 2, (3, 'enzyme0', 'enzyme4'): 1, (3, 'enzyme0', 'enzyme5'): 3, (3, 'enzyme1', 'enzyme0'): 4, (3, 'enzyme1', 'enzyme1'): 0, (3, 'enzyme1', 'enzyme2'): 3, (3, 'enzyme1', 'enzyme3'): 3, (3, 'enzyme1', 'enzyme4'): 1, (3, 'enzyme1', 'enzyme5'): 3, (3, 'enzyme2', 'enzyme0'): 2, (3, 'enzyme2', 'enzyme1'): 2, (3, 'enzyme2', 'enzyme2'): 0, (3, 'enzyme2', 'enzyme3'): 2, (3, 'enzyme2', 'enzyme4'): 1, (3, 'enzyme2', 'enzyme5'): 1, (3, 'enzyme3', 'enzyme0'): 3, (3, 'enzyme3', 'enzyme1'): 1, (3, 'enzyme3', 'enzyme2'): 1, (3, 'enzyme3', 'enzyme3'): 0, (3, 'enzyme3', 'enzyme4'): 4, (3, 'enzyme3', 'enzyme5'): 4, (3, 'enzyme4', 'enzyme0'): 3, (3, 'enzyme4', 'enzyme1'): 2, (3, 'enzyme4', 'enzyme2'): 3, (3, 'enzyme4', 'enzyme3'): 3, (3, 'enzyme4', 'enzyme4'): 0, (3, 'enzyme4', 'enzyme5'): 3, (3, 'enzyme5', 'enzyme0'): 1, (3, 'enzyme5', 'enzyme1'): 1, (3, 'enzyme5', 'enzyme2'): 3, (3, 'enzyme5', 'enzyme3'): 1, (3, 'enzyme5', 'enzyme4'): 4, (3, 'enzyme5', 'enzyme5'): 0, (4, 'enzyme0', 'enzyme0'): 0, (4, 'enzyme0', 'enzyme1'): 2, (4, 'enzyme0', 'enzyme2'): 4, (4, 'enzyme0', 'enzyme3'): 2, (4, 'enzyme0', 'enzyme4'): 1, (4, 'enzyme0', 'enzyme5'): 1, (4, 'enzyme1', 'enzyme0'): 2, (4, 'enzyme1', 'enzyme1'): 0, (4, 'enzyme1', 'enzyme2'): 3, (4, 'enzyme1', 'enzyme3'): 3, (4, 'enzyme1', 'enzyme4'): 2, (4, 'enzyme1', 'enzyme5'): 4, (4, 'enzyme2', 'enzyme0'): 1, (4, 'enzyme2', 'enzyme1'): 4, (4, 'enzyme2', 'enzyme2'): 0, (4, 'enzyme2', 'enzyme3'): 3, (4, 'enzyme2', 'enzyme4'): 4, (4, 'enzyme2', 'enzyme5'): 3, (4, 'enzyme3', 'enzyme0'): 2, (4, 'enzyme3', 'enzyme1'): 1, (4, 'enzyme3', 'enzyme2'): 2, (4, 'enzyme3', 'enzyme3'): 0, (4, 'enzyme3', 'enzyme4'): 2, (4, 'enzyme3', 'enzyme5'): 1, (4, 'enzyme4', 'enzyme0'): 4, (4, 'enzyme4', 'enzyme1'): 4, (4, 'enzyme4', 'enzyme2'): 2, (4, 'enzyme4', 'enzyme3'): 3, (4, 'enzyme4', 'enzyme4'): 0, (4, 'enzyme4', 'enzyme5'): 1, (4, 'enzyme5', 'enzyme0'): 1, (4, 'enzyme5', 'enzyme1'): 3, (4, 'enzyme5', 'enzyme2'): 1, (4, 'enzyme5', 'enzyme3'): 2, (4, 'enzyme5', 'enzyme4'): 1, (4, 'enzyme5', 'enzyme5'): 0, (5, 'enzyme0', 'enzyme0'): 0, (5, 'enzyme0', 'enzyme1'): 2, (5, 'enzyme0', 'enzyme2'): 2, (5, 'enzyme0', 'enzyme3'): 2, (5, 'enzyme0', 'enzyme4'): 1, (5, 'enzyme0', 'enzyme5'): 3, (5, 'enzyme1', 'enzyme0'): 4, (5, 'enzyme1', 'enzyme1'): 0, (5, 'enzyme1', 'enzyme2'): 2, (5, 'enzyme1', 'enzyme3'): 3, (5, 'enzyme1', 'enzyme4'): 4, (5, 'enzyme1', 'enzyme5'): 2, (5, 'enzyme2', 'enzyme0'): 3, (5, 'enzyme2', 'enzyme1'): 1, (5, 'enzyme2', 'enzyme2'): 0, (5, 'enzyme2', 'enzyme3'): 3, (5, 'enzyme2', 'enzyme4'): 4, (5, 'enzyme2', 'enzyme5'): 3, (5, 'enzyme3', 'enzyme0'): 2, (5, 'enzyme3', 'enzyme1'): 3, (5, 'enzyme3', 'enzyme2'): 3, (5, 'enzyme3', 'enzyme3'): 0, (5, 'enzyme3', 'enzyme4'): 1, (5, 'enzyme3', 'enzyme5'): 2, (5, 'enzyme4', 'enzyme0'): 1, (5, 'enzyme4', 'enzyme1'): 4, (5, 'enzyme4', 'enzyme2'): 1, (5, 'enzyme4', 'enzyme3'): 1, (5, 'enzyme4', 'enzyme4'): 0, (5, 'enzyme4', 'enzyme5'): 3, (5, 'enzyme5', 'enzyme0'): 4, (5, 'enzyme5', 'enzyme1'): 4, (5, 'enzyme5', 'enzyme2'): 3, (5, 'enzyme5', 'enzyme3'): 4, (5, 'enzyme5', 'enzyme4'): 4, (5, 'enzyme5', 'enzyme5'): 0, (6, 'enzyme0', 'enzyme0'): 0, (6, 'enzyme0', 'enzyme1'): 1, (6, 'enzyme0', 'enzyme2'): 1, (6, 'enzyme0', 'enzyme3'): 2, (6, 'enzyme0', 'enzyme4'): 3, (6, 'enzyme0', 'enzyme5'): 3, (6, 'enzyme1', 'enzyme0'): 4, (6, 'enzyme1', 'enzyme1'): 0, (6, 'enzyme1', 'enzyme2'): 2, (6, 'enzyme1', 'enzyme3'): 1, (6, 'enzyme1', 'enzyme4'): 3, (6, 'enzyme1', 'enzyme5'): 2, (6, 'enzyme2', 'enzyme0'): 4, (6, 'enzyme2', 'enzyme1'): 3, (6, 'enzyme2', 'enzyme2'): 0, (6, 'enzyme2', 'enzyme3'): 4, (6, 'enzyme2', 'enzyme4'): 3, (6, 'enzyme2', 'enzyme5'): 1, (6, 'enzyme3', 'enzyme0'): 3, (6, 'enzyme3', 'enzyme1'): 1, (6, 'enzyme3', 'enzyme2'): 3, (6, 'enzyme3', 'enzyme3'): 0, (6, 'enzyme3', 'enzyme4'): 2, (6, 'enzyme3', 'enzyme5'): 2, (6, 'enzyme4', 'enzyme0'): 2, (6, 'enzyme4', 'enzyme1'): 2, (6, 'enzyme4', 'enzyme2'): 2, (6, 'enzyme4', 'enzyme3'): 4, (6, 'enzyme4', 'enzyme4'): 0, (6, 'enzyme4', 'enzyme5'): 1, (6, 'enzyme5', 'enzyme0'): 2, (6, 'enzyme5', 'enzyme1'): 2, (6, 'enzyme5', 'enzyme2'): 3, (6, 'enzyme5', 'enzyme3'): 1, (6, 'enzyme5', 'enzyme4'): 3, (6, 'enzyme5', 'enzyme5'): 0, (7, 'enzyme0', 'enzyme0'): 0, (7, 'enzyme0', 'enzyme1'): 4, (7, 'enzyme0', 'enzyme2'): 2, (7, 'enzyme0', 'enzyme3'): 2, (7, 'enzyme0', 'enzyme4'): 2, (7, 'enzyme0', 'enzyme5'): 2, (7, 'enzyme1', 'enzyme0'): 2, (7, 'enzyme1', 'enzyme1'): 0, (7, 'enzyme1', 'enzyme2'): 3, (7, 'enzyme1', 'enzyme3'): 2, (7, 'enzyme1', 'enzyme4'): 1, (7, 'enzyme1', 'enzyme5'): 4, (7, 'enzyme2', 'enzyme0'): 1, (7, 'enzyme2', 'enzyme1'): 1, (7, 'enzyme2', 'enzyme2'): 0, (7, 'enzyme2', 'enzyme3'): 2, (7, 'enzyme2', 'enzyme4'): 4, (7, 'enzyme2', 'enzyme5'): 2, (7, 'enzyme3', 'enzyme0'): 3, (7, 'enzyme3', 'enzyme1'): 1, (7, 'enzyme3', 'enzyme2'): 3, (7, 'enzyme3', 'enzyme3'): 0, (7, 'enzyme3', 'enzyme4'): 2, (7, 'enzyme3', 'enzyme5'): 3, (7, 'enzyme4', 'enzyme0'): 2, (7, 'enzyme4', 'enzyme1'): 3, (7, 'enzyme4', 'enzyme2'): 4, (7, 'enzyme4', 'enzyme3'): 4, (7, 'enzyme4', 'enzyme4'): 0, (7, 'enzyme4', 'enzyme5'): 3, (7, 'enzyme5', 'enzyme0'): 4, (7, 'enzyme5', 'enzyme1'): 1, (7, 'enzyme5', 'enzyme2'): 1, (7, 'enzyme5', 'enzyme3'): 1, (7, 'enzyme5', 'enzyme4'): 3, (7, 'enzyme5', 'enzyme5'): 0, (8, 'enzyme0', 'enzyme0'): 0, (8, 'enzyme0', 'enzyme1'): 1, (8, 'enzyme0', 'enzyme2'): 2, (8, 'enzyme0', 'enzyme3'): 3, (8, 'enzyme0', 'enzyme4'): 2, (8, 'enzyme0', 'enzyme5'): 4, (8, 'enzyme1', 'enzyme0'): 1, (8, 'enzyme1', 'enzyme1'): 0, (8, 'enzyme1', 'enzyme2'): 1, (8, 'enzyme1', 'enzyme3'): 4, (8, 'enzyme1', 'enzyme4'): 3, (8, 'enzyme1', 'enzyme5'): 1, (8, 'enzyme2', 'enzyme0'): 4, (8, 'enzyme2', 'enzyme1'): 1, (8, 'enzyme2', 'enzyme2'): 0, (8, 'enzyme2', 'enzyme3'): 3, (8, 'enzyme2', 'enzyme4'): 2, (8, 'enzyme2', 'enzyme5'): 4, (8, 'enzyme3', 'enzyme0'): 3, (8, 'enzyme3', 'enzyme1'): 2, (8, 'enzyme3', 'enzyme2'): 4, (8, 'enzyme3', 'enzyme3'): 0, (8, 'enzyme3', 'enzyme4'): 2, (8, 'enzyme3', 'enzyme5'): 3, (8, 'enzyme4', 'enzyme0'): 3, (8, 'enzyme4', 'enzyme1'): 4, (8, 'enzyme4', 'enzyme2'): 1, (8, 'enzyme4', 'enzyme3'): 4, (8, 'enzyme4', 'enzyme4'): 0, (8, 'enzyme4', 'enzyme5'): 1, (8, 'enzyme5', 'enzyme0'): 4, (8, 'enzyme5', 'enzyme1'): 4, (8, 'enzyme5', 'enzyme2'): 1, (8, 'enzyme5', 'enzyme3'): 1, (8, 'enzyme5', 'enzyme4'): 2, (8, 'enzyme5', 'enzyme5'): 0}
