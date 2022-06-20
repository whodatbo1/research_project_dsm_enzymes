nr_machines = 9
nr_jobs = 48
orders = {0: {'product': 'enzyme1', 'due': 30}, 1: {'product': 'enzyme1', 'due': 69}, 2: {'product': 'enzyme1', 'due': 42}, 3: {'product': 'enzyme1', 'due': 88}, 4: {'product': 'enzyme1', 'due': 24}, 5: {'product': 'enzyme1', 'due': 40}, 6: {'product': 'enzyme1', 'due': 28}, 7: {'product': 'enzyme1', 'due': 83}, 8: {'product': 'enzyme4', 'due': 37}, 9: {'product': 'enzyme4', 'due': 63}, 10: {'product': 'enzyme4', 'due': 80}, 11: {'product': 'enzyme4', 'due': 83}, 12: {'product': 'enzyme4', 'due': 33}, 13: {'product': 'enzyme4', 'due': 83}, 14: {'product': 'enzyme4', 'due': 37}, 15: {'product': 'enzyme4', 'due': 17}, 16: {'product': 'enzyme5', 'due': 16}, 17: {'product': 'enzyme5', 'due': 79}, 18: {'product': 'enzyme5', 'due': 62}, 19: {'product': 'enzyme5', 'due': 47}, 20: {'product': 'enzyme5', 'due': 69}, 21: {'product': 'enzyme5', 'due': 71}, 22: {'product': 'enzyme5', 'due': 29}, 23: {'product': 'enzyme5', 'due': 53}, 24: {'product': 'enzyme2', 'due': 64}, 25: {'product': 'enzyme2', 'due': 28}, 26: {'product': 'enzyme2', 'due': 81}, 27: {'product': 'enzyme2', 'due': 74}, 28: {'product': 'enzyme2', 'due': 43}, 29: {'product': 'enzyme2', 'due': 77}, 30: {'product': 'enzyme2', 'due': 70}, 31: {'product': 'enzyme2', 'due': 37}, 32: {'product': 'enzyme0', 'due': 50}, 33: {'product': 'enzyme0', 'due': 49}, 34: {'product': 'enzyme0', 'due': 30}, 35: {'product': 'enzyme0', 'due': 70}, 36: {'product': 'enzyme0', 'due': 85}, 37: {'product': 'enzyme0', 'due': 30}, 38: {'product': 'enzyme0', 'due': 66}, 39: {'product': 'enzyme0', 'due': 77}, 40: {'product': 'enzyme1', 'due': 35}, 41: {'product': 'enzyme1', 'due': 10}, 42: {'product': 'enzyme1', 'due': 90}, 43: {'product': 'enzyme1', 'due': 37}, 44: {'product': 'enzyme1', 'due': 37}, 45: {'product': 'enzyme1', 'due': 57}, 46: {'product': 'enzyme1', 'due': 29}, 47: {'product': 'enzyme1', 'due': 42}}
machines = [0, 1, 2, 3, 4, 5, 6, 7, 8]
jobs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
operations = {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1], 4: [0, 1], 5: [0, 1], 6: [0, 1], 7: [0, 1], 8: [0, 1, 2], 9: [0, 1, 2], 10: [0, 1, 2], 11: [0, 1, 2], 12: [0, 1, 2], 13: [0, 1, 2], 14: [0, 1, 2], 15: [0, 1, 2], 16: [0, 1], 17: [0, 1], 18: [0, 1], 19: [0, 1], 20: [0, 1], 21: [0, 1], 22: [0, 1], 23: [0, 1], 24: [0, 1], 25: [0, 1], 26: [0, 1], 27: [0, 1], 28: [0, 1], 29: [0, 1], 30: [0, 1], 31: [0, 1], 32: [0, 1, 2], 33: [0, 1, 2], 34: [0, 1, 2], 35: [0, 1, 2], 36: [0, 1, 2], 37: [0, 1, 2], 38: [0, 1, 2], 39: [0, 1, 2], 40: [0, 1], 41: [0, 1], 42: [0, 1], 43: [0, 1], 44: [0, 1], 45: [0, 1], 46: [0, 1], 47: [0, 1]}
machineAlternatives = {(0, 0): [0, 1, 2], (0, 1): [3, 4, 5, 6], (1, 0): [0, 1, 2], (1, 1): [3, 4, 5, 6], (1, 2): [7, 8], (2, 0): [0, 1, 2], (2, 1): [3, 4, 5, 6], (2, 2): [7, 8], (3, 0): [0, 1, 2], (3, 1): [3, 4, 5, 6], (3, 2): [7, 8], (4, 0): [0, 1, 2], (4, 1): [3, 4, 5, 6], (5, 0): [0, 1, 2], (5, 1): [3, 4, 5, 6], (0, 2): [7, 8], (4, 2): [7, 8], (5, 2): [7, 8], (6, 0): [0, 1, 2], (6, 1): [3, 4, 5, 6], (7, 0): [0, 1, 2], (7, 1): [3, 4, 5, 6], (8, 0): [0, 1, 2], (8, 1): [3, 4, 5, 6], (9, 0): [0, 1, 2], (9, 1): [3, 4, 5, 6], (10, 0): [0, 1, 2], (10, 1): [3, 4, 5, 6], (11, 0): [0, 1, 2], (11, 1): [3, 4, 5, 6], (8, 2): [7, 8], (9, 2): [7, 8], (6, 2): [7, 8], (7, 2): [7, 8], (10, 2): [7, 8], (11, 2): [7, 8], (12, 0): [0, 1, 2], (12, 1): [3, 4, 5, 6], (12, 2): [7, 8], (13, 0): [0, 1, 2], (13, 1): [3, 4, 5, 6], (13, 2): [7, 8], (14, 0): [0, 1, 2], (14, 1): [3, 4, 5, 6], (14, 2): [7, 8], (15, 0): [0, 1, 2], (15, 1): [3, 4, 5, 6], (16, 0): [3, 4, 5, 6], (16, 1): [7, 8], (17, 0): [3, 4, 5, 6], (17, 1): [7, 8], (15, 2): [7, 8], (16, 2): [7, 8], (17, 2): [7, 8], (18, 0): [3, 4, 5, 6], (18, 1): [7, 8], (18, 2): [7, 8], (19, 0): [3, 4, 5, 6], (19, 1): [7, 8], (19, 2): [7, 8], (20, 0): [3, 4, 5, 6], (20, 1): [7, 8], (20, 2): [7, 8], (21, 0): [3, 4, 5, 6], (21, 1): [7, 8], (21, 2): [7, 8], (22, 0): [3, 4, 5, 6], (22, 1): [7, 8], (22, 2): [7, 8], (23, 0): [3, 4, 5, 6], (23, 1): [7, 8], (23, 2): [7, 8], (24, 0): [3, 4, 5, 6], (24, 1): [7, 8], (25, 0): [3, 4, 5, 6], (25, 1): [7, 8], (26, 0): [3, 4, 5, 6], (26, 1): [7, 8], (27, 0): [3, 4, 5, 6], (27, 1): [7, 8], (28, 0): [3, 4, 5, 6], (28, 1): [7, 8], (29, 0): [3, 4, 5, 6], (29, 1): [7, 8], (24, 2): [7, 8], (25, 2): [7, 8], (26, 2): [7, 8], (27, 2): [7, 8], (28, 2): [7, 8], (29, 2): [7, 8], (30, 0): [3, 4, 5, 6], (30, 1): [7, 8], (30, 2): [7, 8], (31, 0): [3, 4, 5, 6], (31, 1): [7, 8], (31, 2): [7, 8], (32, 0): [0, 1, 2], (32, 1): [3, 4, 5, 6], (32, 2): [7, 8], (33, 0): [0, 1, 2], (33, 1): [3, 4, 5, 6], (33, 2): [7, 8], (34, 0): [0, 1, 2], (34, 1): [3, 4, 5, 6], (34, 2): [7, 8], (35, 0): [0, 1, 2], (35, 1): [3, 4, 5, 6], (35, 2): [7, 8], (36, 0): [0, 1, 2], (36, 1): [3, 4, 5, 6], (37, 0): [0, 1, 2], (37, 1): [3, 4, 5, 6], (38, 0): [0, 1, 2], (38, 1): [3, 4, 5, 6], (39, 0): [0, 1, 2], (39, 1): [3, 4, 5, 6], (40, 0): [0, 1, 2], (40, 1): [3, 4, 5, 6], (41, 0): [0, 1, 2], (41, 1): [3, 4, 5, 6], (36, 2): [7, 8], (37, 2): [7, 8], (38, 2): [7, 8], (39, 2): [7, 8], (40, 2): [7, 8], (41, 2): [7, 8], (42, 0): [0, 1, 2], (42, 1): [3, 4, 5, 6], (42, 2): [7, 8], (43, 0): [0, 1, 2], (43, 1): [3, 4, 5, 6], (43, 2): [7, 8], (44, 0): [0, 1, 2], (44, 1): [3, 4, 5, 6], (44, 2): [7, 8], (45, 0): [0, 1, 2], (45, 1): [3, 4, 5, 6], (45, 2): [7, 8], (46, 0): [0, 1, 2], (46, 1): [3, 4, 5, 6], (46, 2): [7, 8], (47, 0): [0, 1, 2], (47, 1): [3, 4, 5, 6], (47, 2): [7, 8]}
processingTimes = {(0, 0, 3): 8, (0, 0, 4): 8, (0, 0, 5): 8, (0, 0, 6): 8, (0, 1, 7): 3, (0, 1, 8): 3, (1, 0, 0): 3, (1, 0, 1): 3, (1, 0, 2): 3, (1, 1, 3): 2, (1, 1, 4): 2, (1, 1, 5): 2, (1, 1, 6): 2, (1, 2, 7): 4, (1, 2, 8): 4, (2, 0, 0): 3, (2, 0, 1): 3, (2, 0, 2): 3, (2, 1, 3): 2, (2, 1, 4): 2, (2, 1, 5): 2, (2, 1, 6): 2, (2, 2, 7): 4, (2, 2, 8): 4, (3, 0, 0): 3, (3, 0, 1): 3, (3, 0, 2): 3, (3, 1, 3): 2, (3, 1, 4): 2, (3, 1, 5): 2, (3, 1, 6): 2, (3, 2, 7): 4, (3, 2, 8): 4, (4, 0, 0): 3, (4, 0, 1): 3, (4, 0, 2): 3, (4, 1, 3): 2, (4, 1, 4): 2, (4, 1, 5): 2, (4, 1, 6): 2, (5, 0, 3): 8, (5, 0, 4): 8, (5, 0, 5): 8, (5, 0, 6): 8, (5, 1, 7): 3, (5, 1, 8): 3, (0, 0, 0): 3, (0, 0, 1): 3, (0, 0, 2): 3, (0, 1, 3): 2, (0, 1, 4): 2, (0, 1, 5): 2, (0, 1, 6): 2, (0, 2, 7): 4, (0, 2, 8): 4, (2, 0, 3): 8, (2, 0, 4): 8, (2, 0, 5): 8, (2, 0, 6): 8, (2, 1, 7): 3, (2, 1, 8): 3, (4, 2, 7): 4, (4, 2, 8): 4, (5, 0, 0): 3, (5, 0, 1): 3, (5, 0, 2): 3, (5, 1, 3): 2, (5, 1, 4): 2, (5, 1, 5): 2, (5, 1, 6): 2, (4, 0, 3): 8, (4, 0, 4): 8, (4, 0, 5): 8, (4, 0, 6): 8, (4, 1, 7): 3, (4, 1, 8): 3, (1, 0, 3): 8, (1, 0, 4): 8, (1, 0, 5): 8, (1, 0, 6): 8, (1, 1, 7): 3, (1, 1, 8): 3, (3, 0, 3): 8, (3, 0, 4): 8, (3, 0, 5): 8, (3, 0, 6): 8, (3, 1, 7): 3, (3, 1, 8): 3, (5, 2, 7): 4, (5, 2, 8): 4, (6, 0, 0): 3, (6, 0, 1): 3, (6, 0, 2): 3, (6, 1, 3): 2, (6, 1, 4): 2, (6, 1, 5): 2, (6, 1, 6): 2, (7, 0, 0): 3, (7, 0, 1): 3, (7, 0, 2): 3, (7, 1, 3): 2, (7, 1, 4): 2, (7, 1, 5): 2, (7, 1, 6): 2, (8, 0, 3): 3, (8, 0, 4): 3, (8, 0, 5): 3, (8, 0, 6): 3, (8, 1, 7): 3, (8, 1, 8): 3, (9, 0, 3): 3, (9, 0, 4): 3, (9, 0, 5): 3, (9, 0, 6): 3, (9, 1, 7): 3, (9, 1, 8): 3, (10, 0, 0): 5, (10, 0, 1): 5, (10, 0, 2): 5, (10, 1, 3): 4, (10, 1, 4): 4, (10, 1, 5): 4, (10, 1, 6): 4, (11, 0, 0): 5, (11, 0, 1): 5, (11, 0, 2): 5, (11, 1, 3): 4, (11, 1, 4): 4, (11, 1, 5): 4, (11, 1, 6): 4, (8, 0, 0): 5, (8, 0, 1): 5, (8, 0, 2): 5, (8, 1, 3): 4, (8, 1, 4): 4, (8, 1, 5): 4, (8, 1, 6): 4, (8, 2, 7): 7, (8, 2, 8): 7, (9, 0, 0): 5, (9, 0, 1): 5, (9, 0, 2): 5, (9, 1, 3): 4, (9, 1, 4): 4, (9, 1, 5): 4, (9, 1, 6): 4, (9, 2, 7): 7, (9, 2, 8): 7, (10, 0, 3): 3, (10, 0, 4): 3, (10, 0, 5): 3, (10, 0, 6): 3, (10, 1, 7): 3, (10, 1, 8): 3, (11, 0, 3): 3, (11, 0, 4): 3, (11, 0, 5): 3, (11, 0, 6): 3, (11, 1, 7): 3, (11, 1, 8): 3, (6, 2, 7): 4, (6, 2, 8): 4, (7, 2, 7): 4, (7, 2, 8): 4, (10, 2, 7): 7, (10, 2, 8): 7, (11, 2, 7): 7, (11, 2, 8): 7, (6, 0, 3): 8, (6, 0, 4): 8, (6, 0, 5): 8, (6, 0, 6): 8, (6, 1, 7): 3, (6, 1, 8): 3, (7, 0, 3): 8, (7, 0, 4): 8, (7, 0, 5): 8, (7, 0, 6): 8, (7, 1, 7): 3, (7, 1, 8): 3, (12, 0, 0): 5, (12, 0, 1): 5, (12, 0, 2): 5, (12, 1, 3): 4, (12, 1, 4): 4, (12, 1, 5): 4, (12, 1, 6): 4, (12, 2, 7): 7, (12, 2, 8): 7, (13, 0, 0): 5, (13, 0, 1): 5, (13, 0, 2): 5, (13, 1, 3): 4, (13, 1, 4): 4, (13, 1, 5): 4, (13, 1, 6): 4, (13, 2, 7): 7, (13, 2, 8): 7, (14, 0, 0): 5, (14, 0, 1): 5, (14, 0, 2): 5, (14, 1, 3): 4, (14, 1, 4): 4, (14, 1, 5): 4, (14, 1, 6): 4, (14, 2, 7): 7, (14, 2, 8): 7, (15, 0, 3): 3, (15, 0, 4): 3, (15, 0, 5): 3, (15, 0, 6): 3, (15, 1, 7): 3, (15, 1, 8): 3, (16, 0, 3): 8, (16, 0, 4): 8, (16, 0, 5): 8, (16, 0, 6): 8, (16, 1, 7): 3, (16, 1, 8): 3, (17, 0, 3): 8, (17, 0, 4): 8, (17, 0, 5): 8, (17, 0, 6): 8, (17, 1, 7): 3, (17, 1, 8): 3, (15, 0, 0): 5, (15, 0, 1): 5, (15, 0, 2): 5, (15, 1, 3): 4, (15, 1, 4): 4, (15, 1, 5): 4, (15, 1, 6): 4, (15, 2, 7): 7, (15, 2, 8): 7, (16, 0, 0): 5, (16, 0, 1): 5, (16, 0, 2): 5, (16, 1, 3): 4, (16, 1, 4): 4, (16, 1, 5): 4, (16, 1, 6): 4, (16, 2, 7): 7, (16, 2, 8): 7, (17, 0, 0): 5, (17, 0, 1): 5, (17, 0, 2): 5, (17, 1, 3): 4, (17, 1, 4): 4, (17, 1, 5): 4, (17, 1, 6): 4, (17, 2, 7): 7, (17, 2, 8): 7, (12, 0, 3): 3, (12, 0, 4): 3, (12, 0, 5): 3, (12, 0, 6): 3, (12, 1, 7): 3, (12, 1, 8): 3, (13, 0, 3): 3, (13, 0, 4): 3, (13, 0, 5): 3, (13, 0, 6): 3, (13, 1, 7): 3, (13, 1, 8): 3, (14, 0, 3): 3, (14, 0, 4): 3, (14, 0, 5): 3, (14, 0, 6): 3, (14, 1, 7): 3, (14, 1, 8): 3, (18, 0, 0): 5, (18, 0, 1): 5, (18, 0, 2): 5, (18, 1, 3): 4, (18, 1, 4): 4, (18, 1, 5): 4, (18, 1, 6): 4, (18, 2, 7): 7, (18, 2, 8): 7, (19, 0, 0): 5, (19, 0, 1): 5, (19, 0, 2): 5, (19, 1, 3): 4, (19, 1, 4): 4, (19, 1, 5): 4, (19, 1, 6): 4, (19, 2, 7): 7, (19, 2, 8): 7, (20, 0, 0): 5, (20, 0, 1): 5, (20, 0, 2): 5, (20, 1, 3): 4, (20, 1, 4): 4, (20, 1, 5): 4, (20, 1, 6): 4, (20, 2, 7): 7, (20, 2, 8): 7, (21, 0, 0): 5, (21, 0, 1): 5, (21, 0, 2): 5, (21, 1, 3): 4, (21, 1, 4): 4, (21, 1, 5): 4, (21, 1, 6): 4, (21, 2, 7): 7, (21, 2, 8): 7, (22, 0, 0): 5, (22, 0, 1): 5, (22, 0, 2): 5, (22, 1, 3): 4, (22, 1, 4): 4, (22, 1, 5): 4, (22, 1, 6): 4, (22, 2, 7): 7, (22, 2, 8): 7, (23, 0, 0): 5, (23, 0, 1): 5, (23, 0, 2): 5, (23, 1, 3): 4, (23, 1, 4): 4, (23, 1, 5): 4, (23, 1, 6): 4, (23, 2, 7): 7, (23, 2, 8): 7, (20, 0, 3): 8, (20, 0, 4): 8, (20, 0, 5): 8, (20, 0, 6): 8, (20, 1, 7): 3, (20, 1, 8): 3, (21, 0, 3): 8, (21, 0, 4): 8, (21, 0, 5): 8, (21, 0, 6): 8, (21, 1, 7): 3, (21, 1, 8): 3, (22, 0, 3): 8, (22, 0, 4): 8, (22, 0, 5): 8, (22, 0, 6): 8, (22, 1, 7): 3, (22, 1, 8): 3, (23, 0, 3): 8, (23, 0, 4): 8, (23, 0, 5): 8, (23, 0, 6): 8, (23, 1, 7): 3, (23, 1, 8): 3, (18, 0, 3): 8, (18, 0, 4): 8, (18, 0, 5): 8, (18, 0, 6): 8, (18, 1, 7): 3, (18, 1, 8): 3, (19, 0, 3): 8, (19, 0, 4): 8, (19, 0, 5): 8, (19, 0, 6): 8, (19, 1, 7): 3, (19, 1, 8): 3, (24, 0, 3): 3, (24, 0, 4): 3, (24, 0, 5): 3, (24, 0, 6): 3, (24, 1, 7): 3, (24, 1, 8): 3, (25, 0, 0): 3, (25, 0, 1): 3, (25, 0, 2): 3, (25, 1, 3): 2, (25, 1, 4): 2, (25, 1, 5): 2, (25, 1, 6): 2, (26, 0, 0): 3, (26, 0, 1): 3, (26, 0, 2): 3, (26, 1, 3): 2, (26, 1, 4): 2, (26, 1, 5): 2, (26, 1, 6): 2, (27, 0, 0): 3, (27, 0, 1): 3, (27, 0, 2): 3, (27, 1, 3): 2, (27, 1, 4): 2, (27, 1, 5): 2, (27, 1, 6): 2, (28, 0, 0): 3, (28, 0, 1): 3, (28, 0, 2): 3, (28, 1, 3): 2, (28, 1, 4): 2, (28, 1, 5): 2, (28, 1, 6): 2, (29, 0, 0): 3, (29, 0, 1): 3, (29, 0, 2): 3, (29, 1, 3): 2, (29, 1, 4): 2, (29, 1, 5): 2, (29, 1, 6): 2, (24, 0, 0): 3, (24, 0, 1): 3, (24, 0, 2): 3, (24, 1, 3): 2, (24, 1, 4): 2, (24, 1, 5): 2, (24, 1, 6): 2, (24, 2, 7): 4, (24, 2, 8): 4, (25, 2, 7): 4, (25, 2, 8): 4, (26, 2, 7): 4, (26, 2, 8): 4, (27, 2, 7): 4, (27, 2, 8): 4, (28, 2, 7): 4, (28, 2, 8): 4, (29, 2, 7): 4, (29, 2, 8): 4, (25, 0, 3): 3, (25, 0, 4): 3, (25, 0, 5): 3, (25, 0, 6): 3, (25, 1, 7): 3, (25, 1, 8): 3, (26, 0, 3): 3, (26, 0, 4): 3, (26, 0, 5): 3, (26, 0, 6): 3, (26, 1, 7): 3, (26, 1, 8): 3, (27, 0, 3): 3, (27, 0, 4): 3, (27, 0, 5): 3, (27, 0, 6): 3, (27, 1, 7): 3, (27, 1, 8): 3, (28, 0, 3): 3, (28, 0, 4): 3, (28, 0, 5): 3, (28, 0, 6): 3, (28, 1, 7): 3, (28, 1, 8): 3, (29, 0, 3): 3, (29, 0, 4): 3, (29, 0, 5): 3, (29, 0, 6): 3, (29, 1, 7): 3, (29, 1, 8): 3, (30, 0, 0): 3, (30, 0, 1): 3, (30, 0, 2): 3, (30, 1, 3): 2, (30, 1, 4): 2, (30, 1, 5): 2, (30, 1, 6): 2, (30, 2, 7): 4, (30, 2, 8): 4, (31, 0, 0): 3, (31, 0, 1): 3, (31, 0, 2): 3, (31, 1, 3): 2, (31, 1, 4): 2, (31, 1, 5): 2, (31, 1, 6): 2, (31, 2, 7): 4, (31, 2, 8): 4, (32, 0, 0): 8, (32, 0, 1): 8, (32, 0, 2): 8, (32, 1, 3): 4, (32, 1, 4): 4, (32, 1, 5): 4, (32, 1, 6): 4, (32, 2, 7): 4, (32, 2, 8): 4, (33, 0, 0): 8, (33, 0, 1): 8, (33, 0, 2): 8, (33, 1, 3): 4, (33, 1, 4): 4, (33, 1, 5): 4, (33, 1, 6): 4, (33, 2, 7): 4, (33, 2, 8): 4, (34, 0, 0): 8, (34, 0, 1): 8, (34, 0, 2): 8, (34, 1, 3): 4, (34, 1, 4): 4, (34, 1, 5): 4, (34, 1, 6): 4, (34, 2, 7): 4, (34, 2, 8): 4, (35, 0, 0): 8, (35, 0, 1): 8, (35, 0, 2): 8, (35, 1, 3): 4, (35, 1, 4): 4, (35, 1, 5): 4, (35, 1, 6): 4, (35, 2, 7): 4, (35, 2, 8): 4, (30, 0, 3): 3, (30, 0, 4): 3, (30, 0, 5): 3, (30, 0, 6): 3, (30, 1, 7): 3, (30, 1, 8): 3, (31, 0, 3): 3, (31, 0, 4): 3, (31, 0, 5): 3, (31, 0, 6): 3, (31, 1, 7): 3, (31, 1, 8): 3, (32, 0, 3): 8, (32, 0, 4): 8, (32, 0, 5): 8, (32, 0, 6): 8, (32, 1, 7): 3, (32, 1, 8): 3, (33, 0, 3): 8, (33, 0, 4): 8, (33, 0, 5): 8, (33, 0, 6): 8, (33, 1, 7): 3, (33, 1, 8): 3, (34, 0, 3): 8, (34, 0, 4): 8, (34, 0, 5): 8, (34, 0, 6): 8, (34, 1, 7): 3, (34, 1, 8): 3, (35, 0, 3): 8, (35, 0, 4): 8, (35, 0, 5): 8, (35, 0, 6): 8, (35, 1, 7): 3, (35, 1, 8): 3, (36, 0, 3): 8, (36, 0, 4): 8, (36, 0, 5): 8, (36, 0, 6): 8, (36, 1, 7): 3, (36, 1, 8): 3, (37, 0, 3): 8, (37, 0, 4): 8, (37, 0, 5): 8, (37, 0, 6): 8, (37, 1, 7): 3, (37, 1, 8): 3, (38, 0, 3): 8, (38, 0, 4): 8, (38, 0, 5): 8, (38, 0, 6): 8, (38, 1, 7): 3, (38, 1, 8): 3, (39, 0, 3): 8, (39, 0, 4): 8, (39, 0, 5): 8, (39, 0, 6): 8, (39, 1, 7): 3, (39, 1, 8): 3, (40, 0, 3): 8, (40, 0, 4): 8, (40, 0, 5): 8, (40, 0, 6): 8, (40, 1, 7): 3, (40, 1, 8): 3, (41, 0, 3): 8, (41, 0, 4): 8, (41, 0, 5): 8, (41, 0, 6): 8, (41, 1, 7): 3, (41, 1, 8): 3, (36, 0, 0): 8, (36, 0, 1): 8, (36, 0, 2): 8, (36, 1, 3): 4, (36, 1, 4): 4, (36, 1, 5): 4, (36, 1, 6): 4, (36, 2, 7): 4, (36, 2, 8): 4, (37, 0, 0): 8, (37, 0, 1): 8, (37, 0, 2): 8, (37, 1, 3): 4, (37, 1, 4): 4, (37, 1, 5): 4, (37, 1, 6): 4, (37, 2, 7): 4, (37, 2, 8): 4, (38, 0, 0): 8, (38, 0, 1): 8, (38, 0, 2): 8, (38, 1, 3): 4, (38, 1, 4): 4, (38, 1, 5): 4, (38, 1, 6): 4, (38, 2, 7): 4, (38, 2, 8): 4, (39, 0, 0): 8, (39, 0, 1): 8, (39, 0, 2): 8, (39, 1, 3): 4, (39, 1, 4): 4, (39, 1, 5): 4, (39, 1, 6): 4, (39, 2, 7): 4, (39, 2, 8): 4, (40, 0, 0): 3, (40, 0, 1): 3, (40, 0, 2): 3, (40, 1, 3): 2, (40, 1, 4): 2, (40, 1, 5): 2, (40, 1, 6): 2, (40, 2, 7): 6, (40, 2, 8): 6, (41, 0, 0): 3, (41, 0, 1): 3, (41, 0, 2): 3, (41, 1, 3): 2, (41, 1, 4): 2, (41, 1, 5): 2, (41, 1, 6): 2, (41, 2, 7): 6, (41, 2, 8): 6, (42, 0, 0): 3, (42, 0, 1): 3, (42, 0, 2): 3, (42, 1, 3): 2, (42, 1, 4): 2, (42, 1, 5): 2, (42, 1, 6): 2, (42, 2, 7): 6, (42, 2, 8): 6, (43, 0, 0): 3, (43, 0, 1): 3, (43, 0, 2): 3, (43, 1, 3): 2, (43, 1, 4): 2, (43, 1, 5): 2, (43, 1, 6): 2, (43, 2, 7): 6, (43, 2, 8): 6, (44, 0, 0): 3, (44, 0, 1): 3, (44, 0, 2): 3, (44, 1, 3): 2, (44, 1, 4): 2, (44, 1, 5): 2, (44, 1, 6): 2, (44, 2, 7): 6, (44, 2, 8): 6, (45, 0, 0): 3, (45, 0, 1): 3, (45, 0, 2): 3, (45, 1, 3): 2, (45, 1, 4): 2, (45, 1, 5): 2, (45, 1, 6): 2, (45, 2, 7): 6, (45, 2, 8): 6, (46, 0, 0): 3, (46, 0, 1): 3, (46, 0, 2): 3, (46, 1, 3): 2, (46, 1, 4): 2, (46, 1, 5): 2, (46, 1, 6): 2, (46, 2, 7): 6, (46, 2, 8): 6, (47, 0, 0): 3, (47, 0, 1): 3, (47, 0, 2): 3, (47, 1, 3): 2, (47, 1, 4): 2, (47, 1, 5): 2, (47, 1, 6): 2, (47, 2, 7): 6, (47, 2, 8): 6, (42, 0, 3): 8, (42, 0, 4): 8, (42, 0, 5): 8, (42, 0, 6): 8, (42, 1, 7): 3, (42, 1, 8): 3, (43, 0, 3): 8, (43, 0, 4): 8, (43, 0, 5): 8, (43, 0, 6): 8, (43, 1, 7): 3, (43, 1, 8): 3, (44, 0, 3): 8, (44, 0, 4): 8, (44, 0, 5): 8, (44, 0, 6): 8, (44, 1, 7): 3, (44, 1, 8): 3, (45, 0, 3): 8, (45, 0, 4): 8, (45, 0, 5): 8, (45, 0, 6): 8, (45, 1, 7): 3, (45, 1, 8): 3, (46, 0, 3): 8, (46, 0, 4): 8, (46, 0, 5): 8, (46, 0, 6): 8, (46, 1, 7): 3, (46, 1, 8): 3, (47, 0, 3): 8, (47, 0, 4): 8, (47, 0, 5): 8, (47, 0, 6): 8, (47, 1, 7): 3, (47, 1, 8): 3}
changeOvers = {(0, 'enzyme0', 'enzyme0'): 0, (0, 'enzyme0', 'enzyme1'): 3, (0, 'enzyme0', 'enzyme2'): 1, (0, 'enzyme0', 'enzyme3'): 2, (0, 'enzyme0', 'enzyme4'): 2, (0, 'enzyme0', 'enzyme5'): 3, (0, 'enzyme1', 'enzyme0'): 1, (0, 'enzyme1', 'enzyme1'): 0, (0, 'enzyme1', 'enzyme2'): 1, (0, 'enzyme1', 'enzyme3'): 4, (0, 'enzyme1', 'enzyme4'): 3, (0, 'enzyme1', 'enzyme5'): 1, (0, 'enzyme2', 'enzyme0'): 1, (0, 'enzyme2', 'enzyme1'): 1, (0, 'enzyme2', 'enzyme2'): 0, (0, 'enzyme2', 'enzyme3'): 1, (0, 'enzyme2', 'enzyme4'): 3, (0, 'enzyme2', 'enzyme5'): 2, (0, 'enzyme3', 'enzyme0'): 2, (0, 'enzyme3', 'enzyme1'): 2, (0, 'enzyme3', 'enzyme2'): 3, (0, 'enzyme3', 'enzyme3'): 0, (0, 'enzyme3', 'enzyme4'): 4, (0, 'enzyme3', 'enzyme5'): 2, (0, 'enzyme4', 'enzyme0'): 2, (0, 'enzyme4', 'enzyme1'): 4, (0, 'enzyme4', 'enzyme2'): 3, (0, 'enzyme4', 'enzyme3'): 2, (0, 'enzyme4', 'enzyme4'): 0, (0, 'enzyme4', 'enzyme5'): 4, (0, 'enzyme5', 'enzyme0'): 1, (0, 'enzyme5', 'enzyme1'): 3, (0, 'enzyme5', 'enzyme2'): 1, (0, 'enzyme5', 'enzyme3'): 4, (0, 'enzyme5', 'enzyme4'): 3, (0, 'enzyme5', 'enzyme5'): 0, (1, 'enzyme0', 'enzyme0'): 0, (1, 'enzyme0', 'enzyme1'): 4, (1, 'enzyme0', 'enzyme2'): 1, (1, 'enzyme0', 'enzyme3'): 3, (1, 'enzyme0', 'enzyme4'): 3, (1, 'enzyme0', 'enzyme5'): 1, (1, 'enzyme1', 'enzyme0'): 1, (1, 'enzyme1', 'enzyme1'): 0, (1, 'enzyme1', 'enzyme2'): 2, (1, 'enzyme1', 'enzyme3'): 4, (1, 'enzyme1', 'enzyme4'): 3, (1, 'enzyme1', 'enzyme5'): 4, (1, 'enzyme2', 'enzyme0'): 1, (1, 'enzyme2', 'enzyme1'): 2, (1, 'enzyme2', 'enzyme2'): 0, (1, 'enzyme2', 'enzyme3'): 4, (1, 'enzyme2', 'enzyme4'): 3, (1, 'enzyme2', 'enzyme5'): 3, (1, 'enzyme3', 'enzyme0'): 4, (1, 'enzyme3', 'enzyme1'): 1, (1, 'enzyme3', 'enzyme2'): 1, (1, 'enzyme3', 'enzyme3'): 0, (1, 'enzyme3', 'enzyme4'): 1, (1, 'enzyme3', 'enzyme5'): 2, (1, 'enzyme4', 'enzyme0'): 3, (1, 'enzyme4', 'enzyme1'): 4, (1, 'enzyme4', 'enzyme2'): 4, (1, 'enzyme4', 'enzyme3'): 1, (1, 'enzyme4', 'enzyme4'): 0, (1, 'enzyme4', 'enzyme5'): 4, (1, 'enzyme5', 'enzyme0'): 1, (1, 'enzyme5', 'enzyme1'): 1, (1, 'enzyme5', 'enzyme2'): 1, (1, 'enzyme5', 'enzyme3'): 2, (1, 'enzyme5', 'enzyme4'): 1, (1, 'enzyme5', 'enzyme5'): 0, (2, 'enzyme0', 'enzyme0'): 0, (2, 'enzyme0', 'enzyme1'): 1, (2, 'enzyme0', 'enzyme2'): 3, (2, 'enzyme0', 'enzyme3'): 2, (2, 'enzyme0', 'enzyme4'): 3, (2, 'enzyme0', 'enzyme5'): 2, (2, 'enzyme1', 'enzyme0'): 4, (2, 'enzyme1', 'enzyme1'): 0, (2, 'enzyme1', 'enzyme2'): 1, (2, 'enzyme1', 'enzyme3'): 2, (2, 'enzyme1', 'enzyme4'): 3, (2, 'enzyme1', 'enzyme5'): 3, (2, 'enzyme2', 'enzyme0'): 3, (2, 'enzyme2', 'enzyme1'): 1, (2, 'enzyme2', 'enzyme2'): 0, (2, 'enzyme2', 'enzyme3'): 3, (2, 'enzyme2', 'enzyme4'): 4, (2, 'enzyme2', 'enzyme5'): 4, (2, 'enzyme3', 'enzyme0'): 3, (2, 'enzyme3', 'enzyme1'): 1, (2, 'enzyme3', 'enzyme2'): 2, (2, 'enzyme3', 'enzyme3'): 0, (2, 'enzyme3', 'enzyme4'): 3, (2, 'enzyme3', 'enzyme5'): 3, (2, 'enzyme4', 'enzyme0'): 1, (2, 'enzyme4', 'enzyme1'): 4, (2, 'enzyme4', 'enzyme2'): 1, (2, 'enzyme4', 'enzyme3'): 2, (2, 'enzyme4', 'enzyme4'): 0, (2, 'enzyme4', 'enzyme5'): 4, (2, 'enzyme5', 'enzyme0'): 4, (2, 'enzyme5', 'enzyme1'): 2, (2, 'enzyme5', 'enzyme2'): 1, (2, 'enzyme5', 'enzyme3'): 3, (2, 'enzyme5', 'enzyme4'): 3, (2, 'enzyme5', 'enzyme5'): 0, (3, 'enzyme0', 'enzyme0'): 0, (3, 'enzyme0', 'enzyme1'): 3, (3, 'enzyme0', 'enzyme2'): 3, (3, 'enzyme0', 'enzyme3'): 2, (3, 'enzyme0', 'enzyme4'): 1, (3, 'enzyme0', 'enzyme5'): 3, (3, 'enzyme1', 'enzyme0'): 4, (3, 'enzyme1', 'enzyme1'): 0, (3, 'enzyme1', 'enzyme2'): 3, (3, 'enzyme1', 'enzyme3'): 3, (3, 'enzyme1', 'enzyme4'): 1, (3, 'enzyme1', 'enzyme5'): 3, (3, 'enzyme2', 'enzyme0'): 2, (3, 'enzyme2', 'enzyme1'): 2, (3, 'enzyme2', 'enzyme2'): 0, (3, 'enzyme2', 'enzyme3'): 2, (3, 'enzyme2', 'enzyme4'): 1, (3, 'enzyme2', 'enzyme5'): 1, (3, 'enzyme3', 'enzyme0'): 3, (3, 'enzyme3', 'enzyme1'): 1, (3, 'enzyme3', 'enzyme2'): 1, (3, 'enzyme3', 'enzyme3'): 0, (3, 'enzyme3', 'enzyme4'): 4, (3, 'enzyme3', 'enzyme5'): 4, (3, 'enzyme4', 'enzyme0'): 3, (3, 'enzyme4', 'enzyme1'): 2, (3, 'enzyme4', 'enzyme2'): 3, (3, 'enzyme4', 'enzyme3'): 3, (3, 'enzyme4', 'enzyme4'): 0, (3, 'enzyme4', 'enzyme5'): 3, (3, 'enzyme5', 'enzyme0'): 1, (3, 'enzyme5', 'enzyme1'): 1, (3, 'enzyme5', 'enzyme2'): 3, (3, 'enzyme5', 'enzyme3'): 1, (3, 'enzyme5', 'enzyme4'): 4, (3, 'enzyme5', 'enzyme5'): 0, (4, 'enzyme0', 'enzyme0'): 0, (4, 'enzyme0', 'enzyme1'): 2, (4, 'enzyme0', 'enzyme2'): 4, (4, 'enzyme0', 'enzyme3'): 2, (4, 'enzyme0', 'enzyme4'): 1, (4, 'enzyme0', 'enzyme5'): 1, (4, 'enzyme1', 'enzyme0'): 2, (4, 'enzyme1', 'enzyme1'): 0, (4, 'enzyme1', 'enzyme2'): 3, (4, 'enzyme1', 'enzyme3'): 3, (4, 'enzyme1', 'enzyme4'): 2, (4, 'enzyme1', 'enzyme5'): 4, (4, 'enzyme2', 'enzyme0'): 1, (4, 'enzyme2', 'enzyme1'): 4, (4, 'enzyme2', 'enzyme2'): 0, (4, 'enzyme2', 'enzyme3'): 3, (4, 'enzyme2', 'enzyme4'): 4, (4, 'enzyme2', 'enzyme5'): 3, (4, 'enzyme3', 'enzyme0'): 2, (4, 'enzyme3', 'enzyme1'): 1, (4, 'enzyme3', 'enzyme2'): 2, (4, 'enzyme3', 'enzyme3'): 0, (4, 'enzyme3', 'enzyme4'): 2, (4, 'enzyme3', 'enzyme5'): 1, (4, 'enzyme4', 'enzyme0'): 4, (4, 'enzyme4', 'enzyme1'): 4, (4, 'enzyme4', 'enzyme2'): 2, (4, 'enzyme4', 'enzyme3'): 3, (4, 'enzyme4', 'enzyme4'): 0, (4, 'enzyme4', 'enzyme5'): 1, (4, 'enzyme5', 'enzyme0'): 1, (4, 'enzyme5', 'enzyme1'): 3, (4, 'enzyme5', 'enzyme2'): 1, (4, 'enzyme5', 'enzyme3'): 2, (4, 'enzyme5', 'enzyme4'): 1, (4, 'enzyme5', 'enzyme5'): 0, (5, 'enzyme0', 'enzyme0'): 0, (5, 'enzyme0', 'enzyme1'): 2, (5, 'enzyme0', 'enzyme2'): 2, (5, 'enzyme0', 'enzyme3'): 2, (5, 'enzyme0', 'enzyme4'): 1, (5, 'enzyme0', 'enzyme5'): 3, (5, 'enzyme1', 'enzyme0'): 4, (5, 'enzyme1', 'enzyme1'): 0, (5, 'enzyme1', 'enzyme2'): 2, (5, 'enzyme1', 'enzyme3'): 3, (5, 'enzyme1', 'enzyme4'): 4, (5, 'enzyme1', 'enzyme5'): 2, (5, 'enzyme2', 'enzyme0'): 3, (5, 'enzyme2', 'enzyme1'): 1, (5, 'enzyme2', 'enzyme2'): 0, (5, 'enzyme2', 'enzyme3'): 3, (5, 'enzyme2', 'enzyme4'): 4, (5, 'enzyme2', 'enzyme5'): 3, (5, 'enzyme3', 'enzyme0'): 2, (5, 'enzyme3', 'enzyme1'): 3, (5, 'enzyme3', 'enzyme2'): 3, (5, 'enzyme3', 'enzyme3'): 0, (5, 'enzyme3', 'enzyme4'): 1, (5, 'enzyme3', 'enzyme5'): 2, (5, 'enzyme4', 'enzyme0'): 1, (5, 'enzyme4', 'enzyme1'): 4, (5, 'enzyme4', 'enzyme2'): 1, (5, 'enzyme4', 'enzyme3'): 1, (5, 'enzyme4', 'enzyme4'): 0, (5, 'enzyme4', 'enzyme5'): 3, (5, 'enzyme5', 'enzyme0'): 4, (5, 'enzyme5', 'enzyme1'): 4, (5, 'enzyme5', 'enzyme2'): 3, (5, 'enzyme5', 'enzyme3'): 4, (5, 'enzyme5', 'enzyme4'): 4, (5, 'enzyme5', 'enzyme5'): 0, (6, 'enzyme0', 'enzyme0'): 0, (6, 'enzyme0', 'enzyme1'): 1, (6, 'enzyme0', 'enzyme2'): 1, (6, 'enzyme0', 'enzyme3'): 2, (6, 'enzyme0', 'enzyme4'): 3, (6, 'enzyme0', 'enzyme5'): 3, (6, 'enzyme1', 'enzyme0'): 4, (6, 'enzyme1', 'enzyme1'): 0, (6, 'enzyme1', 'enzyme2'): 2, (6, 'enzyme1', 'enzyme3'): 1, (6, 'enzyme1', 'enzyme4'): 3, (6, 'enzyme1', 'enzyme5'): 2, (6, 'enzyme2', 'enzyme0'): 4, (6, 'enzyme2', 'enzyme1'): 3, (6, 'enzyme2', 'enzyme2'): 0, (6, 'enzyme2', 'enzyme3'): 4, (6, 'enzyme2', 'enzyme4'): 3, (6, 'enzyme2', 'enzyme5'): 1, (6, 'enzyme3', 'enzyme0'): 3, (6, 'enzyme3', 'enzyme1'): 1, (6, 'enzyme3', 'enzyme2'): 3, (6, 'enzyme3', 'enzyme3'): 0, (6, 'enzyme3', 'enzyme4'): 2, (6, 'enzyme3', 'enzyme5'): 2, (6, 'enzyme4', 'enzyme0'): 2, (6, 'enzyme4', 'enzyme1'): 2, (6, 'enzyme4', 'enzyme2'): 2, (6, 'enzyme4', 'enzyme3'): 4, (6, 'enzyme4', 'enzyme4'): 0, (6, 'enzyme4', 'enzyme5'): 1, (6, 'enzyme5', 'enzyme0'): 2, (6, 'enzyme5', 'enzyme1'): 2, (6, 'enzyme5', 'enzyme2'): 3, (6, 'enzyme5', 'enzyme3'): 1, (6, 'enzyme5', 'enzyme4'): 3, (6, 'enzyme5', 'enzyme5'): 0, (7, 'enzyme0', 'enzyme0'): 0, (7, 'enzyme0', 'enzyme1'): 4, (7, 'enzyme0', 'enzyme2'): 2, (7, 'enzyme0', 'enzyme3'): 2, (7, 'enzyme0', 'enzyme4'): 2, (7, 'enzyme0', 'enzyme5'): 2, (7, 'enzyme1', 'enzyme0'): 2, (7, 'enzyme1', 'enzyme1'): 0, (7, 'enzyme1', 'enzyme2'): 3, (7, 'enzyme1', 'enzyme3'): 2, (7, 'enzyme1', 'enzyme4'): 1, (7, 'enzyme1', 'enzyme5'): 4, (7, 'enzyme2', 'enzyme0'): 1, (7, 'enzyme2', 'enzyme1'): 1, (7, 'enzyme2', 'enzyme2'): 0, (7, 'enzyme2', 'enzyme3'): 2, (7, 'enzyme2', 'enzyme4'): 4, (7, 'enzyme2', 'enzyme5'): 2, (7, 'enzyme3', 'enzyme0'): 3, (7, 'enzyme3', 'enzyme1'): 1, (7, 'enzyme3', 'enzyme2'): 3, (7, 'enzyme3', 'enzyme3'): 0, (7, 'enzyme3', 'enzyme4'): 2, (7, 'enzyme3', 'enzyme5'): 3, (7, 'enzyme4', 'enzyme0'): 2, (7, 'enzyme4', 'enzyme1'): 3, (7, 'enzyme4', 'enzyme2'): 4, (7, 'enzyme4', 'enzyme3'): 4, (7, 'enzyme4', 'enzyme4'): 0, (7, 'enzyme4', 'enzyme5'): 3, (7, 'enzyme5', 'enzyme0'): 4, (7, 'enzyme5', 'enzyme1'): 1, (7, 'enzyme5', 'enzyme2'): 1, (7, 'enzyme5', 'enzyme3'): 1, (7, 'enzyme5', 'enzyme4'): 3, (7, 'enzyme5', 'enzyme5'): 0, (8, 'enzyme0', 'enzyme0'): 0, (8, 'enzyme0', 'enzyme1'): 1, (8, 'enzyme0', 'enzyme2'): 2, (8, 'enzyme0', 'enzyme3'): 3, (8, 'enzyme0', 'enzyme4'): 2, (8, 'enzyme0', 'enzyme5'): 4, (8, 'enzyme1', 'enzyme0'): 1, (8, 'enzyme1', 'enzyme1'): 0, (8, 'enzyme1', 'enzyme2'): 1, (8, 'enzyme1', 'enzyme3'): 4, (8, 'enzyme1', 'enzyme4'): 3, (8, 'enzyme1', 'enzyme5'): 1, (8, 'enzyme2', 'enzyme0'): 4, (8, 'enzyme2', 'enzyme1'): 1, (8, 'enzyme2', 'enzyme2'): 0, (8, 'enzyme2', 'enzyme3'): 3, (8, 'enzyme2', 'enzyme4'): 2, (8, 'enzyme2', 'enzyme5'): 4, (8, 'enzyme3', 'enzyme0'): 3, (8, 'enzyme3', 'enzyme1'): 2, (8, 'enzyme3', 'enzyme2'): 4, (8, 'enzyme3', 'enzyme3'): 0, (8, 'enzyme3', 'enzyme4'): 2, (8, 'enzyme3', 'enzyme5'): 3, (8, 'enzyme4', 'enzyme0'): 3, (8, 'enzyme4', 'enzyme1'): 4, (8, 'enzyme4', 'enzyme2'): 1, (8, 'enzyme4', 'enzyme3'): 4, (8, 'enzyme4', 'enzyme4'): 0, (8, 'enzyme4', 'enzyme5'): 1, (8, 'enzyme5', 'enzyme0'): 4, (8, 'enzyme5', 'enzyme1'): 4, (8, 'enzyme5', 'enzyme2'): 1, (8, 'enzyme5', 'enzyme3'): 1, (8, 'enzyme5', 'enzyme4'): 2, (8, 'enzyme5', 'enzyme5'): 0}
