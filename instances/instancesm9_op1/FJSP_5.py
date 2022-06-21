nr_machines = 10
nr_jobs = 36
orders = {0: {'product': 'enzyme0', 'due': 37}, 1: {'product': 'enzyme0', 'due': 35}, 2: {'product': 'enzyme0', 'due': 41}, 3: {'product': 'enzyme0', 'due': 54}, 4: {'product': 'enzyme0', 'due': 67}, 5: {'product': 'enzyme0', 'due': 11}, 6: {'product': 'enzyme1', 'due': 43}, 7: {'product': 'enzyme1', 'due': 37}, 8: {'product': 'enzyme1', 'due': 67}, 9: {'product': 'enzyme1', 'due': 42}, 10: {'product': 'enzyme1', 'due': 32}, 11: {'product': 'enzyme1', 'due': 47}, 12: {'product': 'enzyme2', 'due': 23}, 13: {'product': 'enzyme2', 'due': 62}, 14: {'product': 'enzyme2', 'due': 29}, 15: {'product': 'enzyme2', 'due': 37}, 16: {'product': 'enzyme2', 'due': 48}, 17: {'product': 'enzyme2', 'due': 41}, 18: {'product': 'enzyme3', 'due': 19}, 19: {'product': 'enzyme3', 'due': 60}, 20: {'product': 'enzyme3', 'due': 15}, 21: {'product': 'enzyme3', 'due': 60}, 22: {'product': 'enzyme3', 'due': 60}, 23: {'product': 'enzyme3', 'due': 40}, 24: {'product': 'enzyme4', 'due': 31}, 25: {'product': 'enzyme4', 'due': 66}, 26: {'product': 'enzyme4', 'due': 53}, 27: {'product': 'enzyme4', 'due': 48}, 28: {'product': 'enzyme4', 'due': 42}, 29: {'product': 'enzyme4', 'due': 27}, 30: {'product': 'enzyme5', 'due': 46}, 31: {'product': 'enzyme5', 'due': 35}, 32: {'product': 'enzyme5', 'due': 60}, 33: {'product': 'enzyme5', 'due': 19}, 34: {'product': 'enzyme5', 'due': 16}, 35: {'product': 'enzyme5', 'due': 16}}
machines = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
jobs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
operations = {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2], 4: [0, 1, 2], 5: [0, 1, 2], 6: [0, 1], 7: [0, 1], 8: [0, 1], 9: [0, 1], 10: [0, 1], 11: [0, 1], 12: [0, 1], 13: [0, 1], 14: [0, 1], 15: [0, 1], 16: [0, 1], 17: [0, 1], 18: [0, 1, 2], 19: [0, 1, 2], 20: [0, 1, 2], 21: [0, 1, 2], 22: [0, 1, 2], 23: [0, 1, 2], 24: [0, 1, 2], 25: [0, 1, 2], 26: [0, 1, 2], 27: [0, 1, 2], 28: [0, 1, 2], 29: [0, 1, 2], 30: [0, 1], 31: [0, 1], 32: [0, 1], 33: [0, 1], 34: [0, 1], 35: [0, 1]}
machineAlternatives = {(0, 0): [0, 1, 2, 9], (0, 1): [3, 4, 5, 6], (0, 2): [7, 8], (1, 0): [0, 1, 2, 9], (1, 1): [3, 4, 5, 6], (2, 0): [0, 1, 2, 9], (2, 1): [3, 4, 5, 6], (3, 0): [0, 1, 2, 9], (3, 1): [3, 4, 5, 6], (3, 2): [7, 8], (4, 0): [0, 1, 2, 9], (4, 1): [3, 4, 5, 6], (4, 2): [7, 8], (5, 0): [0, 1, 2, 9], (5, 1): [3, 4, 5, 6], (1, 2): [7, 8], (6, 0): [0, 1, 2, 9], (6, 1): [3, 4, 5, 6], (6, 2): [7, 8], (7, 0): [0, 1, 2, 9], (7, 1): [3, 4, 5, 6], (7, 2): [7, 8], (8, 0): [0, 1, 2, 9], (8, 1): [3, 4, 5, 6], (8, 2): [7, 8], (9, 0): [0, 1, 2, 9], (9, 1): [3, 4, 5, 6], (9, 2): [7, 8], (10, 0): [0, 1, 2, 9], (10, 1): [3, 4, 5, 6], (11, 0): [0, 1, 2, 9], (11, 1): [3, 4, 5, 6], (2, 2): [7, 8], (10, 2): [7, 8], (11, 2): [7, 8], (12, 0): [3, 4, 5, 6], (12, 1): [7, 8], (12, 2): [7, 8], (13, 0): [3, 4, 5, 6], (13, 1): [7, 8], (13, 2): [7, 8], (14, 0): [3, 4, 5, 6], (14, 1): [7, 8], (14, 2): [7, 8], (15, 0): [3, 4, 5, 6], (15, 1): [7, 8], (16, 0): [3, 4, 5, 6], (16, 1): [7, 8], (17, 0): [3, 4, 5, 6], (17, 1): [7, 8], (15, 2): [7, 8], (16, 2): [7, 8], (17, 2): [7, 8], (18, 0): [0, 1, 2, 9], (18, 1): [3, 4, 5, 6], (18, 2): [7, 8], (19, 0): [0, 1, 2, 9], (19, 1): [3, 4, 5, 6], (19, 2): [7, 8], (20, 0): [0, 1, 2, 9], (20, 1): [3, 4, 5, 6], (21, 0): [0, 1, 2, 9], (21, 1): [3, 4, 5, 6], (22, 0): [0, 1, 2, 9], (22, 1): [3, 4, 5, 6], (23, 0): [0, 1, 2, 9], (23, 1): [3, 4, 5, 6], (20, 2): [7, 8], (21, 2): [7, 8], (22, 2): [7, 8], (23, 2): [7, 8], (24, 0): [0, 1, 2, 9], (24, 1): [3, 4, 5, 6], (24, 2): [7, 8], (25, 0): [0, 1, 2, 9], (25, 1): [3, 4, 5, 6], (26, 0): [0, 1, 2, 9], (26, 1): [3, 4, 5, 6], (27, 0): [0, 1, 2, 9], (27, 1): [3, 4, 5, 6], (28, 0): [0, 1, 2, 9], (28, 1): [3, 4, 5, 6], (29, 0): [0, 1, 2, 9], (29, 1): [3, 4, 5, 6], (5, 2): [7, 8], (25, 2): [7, 8], (26, 2): [7, 8], (27, 2): [7, 8], (28, 2): [7, 8], (29, 2): [7, 8], (30, 0): [3, 4, 5, 6], (30, 1): [7, 8], (31, 0): [3, 4, 5, 6], (31, 1): [7, 8], (32, 0): [3, 4, 5, 6], (32, 1): [7, 8], (33, 0): [3, 4, 5, 6], (33, 1): [7, 8], (34, 0): [3, 4, 5, 6], (34, 1): [7, 8], (35, 0): [3, 4, 5, 6], (35, 1): [7, 8]}
processingTimes = {(0, 0, 0): 8, (0, 0, 1): 8, (0, 0, 2): 8, (0, 0, 9): 8, (0, 1, 3): 4, (0, 1, 4): 4, (0, 1, 5): 4, (0, 1, 6): 4, (0, 2, 7): 4, (0, 2, 8): 4, (1, 0, 0): 8, (1, 0, 1): 8, (1, 0, 2): 8, (1, 0, 9): 8, (1, 1, 3): 4, (1, 1, 4): 4, (1, 1, 5): 4, (1, 1, 6): 4, (2, 0, 3): 3, (2, 0, 4): 3, (2, 0, 5): 3, (2, 0, 6): 3, (2, 1, 7): 3, (2, 1, 8): 3, (3, 0, 0): 8, (3, 0, 1): 8, (3, 0, 2): 8, (3, 0, 9): 8, (3, 1, 3): 4, (3, 1, 4): 4, (3, 1, 5): 4, (3, 1, 6): 4, (3, 2, 7): 4, (3, 2, 8): 4, (4, 0, 0): 8, (4, 0, 1): 8, (4, 0, 2): 8, (4, 0, 9): 8, (4, 1, 3): 4, (4, 1, 4): 4, (4, 1, 5): 4, (4, 1, 6): 4, (4, 2, 7): 4, (4, 2, 8): 4, (5, 0, 3): 3, (5, 0, 4): 3, (5, 0, 5): 3, (5, 0, 6): 3, (5, 1, 7): 3, (5, 1, 8): 3, (1, 2, 7): 4, (1, 2, 8): 4, (2, 0, 0): 8, (2, 0, 1): 8, (2, 0, 2): 8, (2, 0, 9): 8, (2, 1, 3): 4, (2, 1, 4): 4, (2, 1, 5): 4, (2, 1, 6): 4, (4, 0, 3): 3, (4, 0, 4): 3, (4, 0, 5): 3, (4, 0, 6): 3, (4, 1, 7): 3, (4, 1, 8): 3, (6, 0, 0): 3, (6, 0, 1): 3, (6, 0, 2): 3, (6, 0, 9): 3, (6, 1, 3): 2, (6, 1, 4): 2, (6, 1, 5): 2, (6, 1, 6): 2, (6, 2, 7): 6, (6, 2, 8): 6, (7, 0, 0): 3, (7, 0, 1): 3, (7, 0, 2): 3, (7, 0, 9): 3, (7, 1, 3): 2, (7, 1, 4): 2, (7, 1, 5): 2, (7, 1, 6): 2, (7, 2, 7): 6, (7, 2, 8): 6, (8, 0, 0): 3, (8, 0, 1): 3, (8, 0, 2): 3, (8, 0, 9): 3, (8, 1, 3): 2, (8, 1, 4): 2, (8, 1, 5): 2, (8, 1, 6): 2, (8, 2, 7): 7, (8, 2, 8): 7, (9, 0, 0): 3, (9, 0, 1): 3, (9, 0, 2): 3, (9, 0, 9): 3, (9, 1, 3): 2, (9, 1, 4): 2, (9, 1, 5): 2, (9, 1, 6): 2, (9, 2, 7): 6, (9, 2, 8): 6, (10, 0, 3): 3, (10, 0, 4): 3, (10, 0, 5): 3, (10, 0, 6): 3, (10, 1, 7): 3, (10, 1, 8): 3, (11, 0, 3): 3, (11, 0, 4): 3, (11, 0, 5): 3, (11, 0, 6): 3, (11, 1, 7): 3, (11, 1, 8): 3, (2, 2, 7): 4, (2, 2, 8): 4, (5, 0, 0): 8, (5, 0, 1): 8, (5, 0, 2): 8, (5, 0, 9): 8, (5, 1, 3): 4, (5, 1, 4): 4, (5, 1, 5): 4, (5, 1, 6): 4, (6, 0, 3): 3, (6, 0, 4): 3, (6, 0, 5): 3, (6, 0, 6): 3, (6, 1, 7): 3, (6, 1, 8): 3, (7, 0, 3): 3, (7, 0, 4): 3, (7, 0, 5): 3, (7, 0, 6): 3, (7, 1, 7): 3, (7, 1, 8): 3, (8, 0, 3): 3, (8, 0, 4): 3, (8, 0, 5): 3, (8, 0, 6): 3, (8, 1, 7): 3, (8, 1, 8): 3, (10, 0, 0): 3, (10, 0, 1): 3, (10, 0, 2): 3, (10, 0, 9): 3, (10, 1, 3): 2, (10, 1, 4): 2, (10, 1, 5): 2, (10, 1, 6): 2, (10, 2, 7): 6, (10, 2, 8): 6, (11, 0, 0): 3, (11, 0, 1): 3, (11, 0, 2): 3, (11, 0, 9): 3, (11, 1, 3): 2, (11, 1, 4): 2, (11, 1, 5): 2, (11, 1, 6): 2, (11, 2, 7): 6, (11, 2, 8): 6, (12, 0, 0): 4, (12, 0, 1): 4, (12, 0, 2): 4, (12, 0, 9): 4, (12, 1, 3): 6, (12, 1, 4): 6, (12, 1, 5): 6, (12, 1, 6): 6, (12, 2, 7): 6, (12, 2, 8): 6, (13, 0, 0): 4, (13, 0, 1): 4, (13, 0, 2): 4, (13, 0, 9): 4, (13, 1, 3): 6, (13, 1, 4): 6, (13, 1, 5): 6, (13, 1, 6): 6, (13, 2, 7): 6, (13, 2, 8): 6, (14, 0, 0): 4, (14, 0, 1): 4, (14, 0, 2): 4, (14, 0, 9): 4, (14, 1, 3): 6, (14, 1, 4): 6, (14, 1, 5): 6, (14, 1, 6): 6, (14, 2, 7): 6, (14, 2, 8): 6, (15, 0, 3): 3, (15, 0, 4): 3, (15, 0, 5): 3, (15, 0, 6): 3, (15, 1, 7): 3, (15, 1, 8): 3, (16, 0, 3): 3, (16, 0, 4): 3, (16, 0, 5): 3, (16, 0, 6): 3, (16, 1, 7): 3, (16, 1, 8): 3, (17, 0, 3): 3, (17, 0, 4): 3, (17, 0, 5): 3, (17, 0, 6): 3, (17, 1, 7): 3, (17, 1, 8): 3, (9, 0, 3): 3, (9, 0, 4): 3, (9, 0, 5): 3, (9, 0, 6): 3, (9, 1, 7): 3, (9, 1, 8): 3, (15, 0, 0): 4, (15, 0, 1): 4, (15, 0, 2): 4, (15, 0, 9): 4, (15, 1, 3): 6, (15, 1, 4): 6, (15, 1, 5): 6, (15, 1, 6): 6, (15, 2, 7): 6, (15, 2, 8): 6, (16, 0, 0): 4, (16, 0, 1): 4, (16, 0, 2): 4, (16, 0, 9): 4, (16, 1, 3): 6, (16, 1, 4): 6, (16, 1, 5): 6, (16, 1, 6): 6, (16, 2, 7): 6, (16, 2, 8): 6, (17, 0, 0): 4, (17, 0, 1): 4, (17, 0, 2): 4, (17, 0, 9): 4, (17, 1, 3): 6, (17, 1, 4): 6, (17, 1, 5): 6, (17, 1, 6): 6, (17, 2, 7): 6, (17, 2, 8): 6, (18, 0, 0): 4, (18, 0, 1): 4, (18, 0, 2): 4, (18, 0, 9): 4, (18, 1, 3): 6, (18, 1, 4): 6, (18, 1, 5): 6, (18, 1, 6): 6, (18, 2, 7): 6, (18, 2, 8): 6, (19, 0, 0): 4, (19, 0, 1): 4, (19, 0, 2): 4, (19, 0, 9): 4, (19, 1, 3): 6, (19, 1, 4): 6, (19, 1, 5): 6, (19, 1, 6): 6, (19, 2, 7): 6, (19, 2, 8): 6, (20, 0, 3): 8, (20, 0, 4): 8, (20, 0, 5): 8, (20, 0, 6): 8, (20, 1, 7): 3, (20, 1, 8): 3, (21, 0, 3): 8, (21, 0, 4): 8, (21, 0, 5): 8, (21, 0, 6): 8, (21, 1, 7): 3, (21, 1, 8): 3, (22, 0, 3): 8, (22, 0, 4): 8, (22, 0, 5): 8, (22, 0, 6): 8, (22, 1, 7): 3, (22, 1, 8): 3, (23, 0, 3): 8, (23, 0, 4): 8, (23, 0, 5): 8, (23, 0, 6): 8, (23, 1, 7): 3, (23, 1, 8): 3, (12, 0, 3): 3, (12, 0, 4): 3, (12, 0, 5): 3, (12, 0, 6): 3, (12, 1, 7): 3, (12, 1, 8): 3, (13, 0, 3): 3, (13, 0, 4): 3, (13, 0, 5): 3, (13, 0, 6): 3, (13, 1, 7): 3, (13, 1, 8): 3, (14, 0, 3): 3, (14, 0, 4): 3, (14, 0, 5): 3, (14, 0, 6): 3, (14, 1, 7): 3, (14, 1, 8): 3, (20, 0, 0): 4, (20, 0, 1): 4, (20, 0, 2): 4, (20, 0, 9): 4, (20, 1, 3): 6, (20, 1, 4): 6, (20, 1, 5): 6, (20, 1, 6): 6, (20, 2, 7): 6, (20, 2, 8): 6, (21, 0, 0): 4, (21, 0, 1): 4, (21, 0, 2): 4, (21, 0, 9): 4, (21, 1, 3): 6, (21, 1, 4): 6, (21, 1, 5): 6, (21, 1, 6): 6, (21, 2, 7): 6, (21, 2, 8): 6, (22, 0, 0): 4, (22, 0, 1): 4, (22, 0, 2): 4, (22, 0, 9): 4, (22, 1, 3): 6, (22, 1, 4): 6, (22, 1, 5): 6, (22, 1, 6): 6, (22, 2, 7): 6, (22, 2, 8): 6, (23, 0, 0): 4, (23, 0, 1): 4, (23, 0, 2): 4, (23, 0, 9): 4, (23, 1, 3): 6, (23, 1, 4): 6, (23, 1, 5): 6, (23, 1, 6): 6, (23, 2, 7): 6, (23, 2, 8): 6, (24, 0, 0): 5, (24, 0, 1): 5, (24, 0, 2): 5, (24, 0, 9): 5, (24, 1, 3): 4, (24, 1, 4): 4, (24, 1, 5): 4, (24, 1, 6): 4, (24, 2, 7): 7, (24, 2, 8): 7, (25, 0, 3): 8, (25, 0, 4): 8, (25, 0, 5): 8, (25, 0, 6): 8, (25, 1, 7): 3, (25, 1, 8): 3, (26, 0, 3): 8, (26, 0, 4): 8, (26, 0, 5): 8, (26, 0, 6): 8, (26, 1, 7): 3, (26, 1, 8): 3, (27, 0, 3): 8, (27, 0, 4): 8, (27, 0, 5): 8, (27, 0, 6): 8, (27, 1, 7): 3, (27, 1, 8): 3, (28, 0, 3): 8, (28, 0, 4): 8, (28, 0, 5): 8, (28, 0, 6): 8, (28, 1, 7): 3, (28, 1, 8): 3, (29, 0, 3): 8, (29, 0, 4): 8, (29, 0, 5): 8, (29, 0, 6): 8, (29, 1, 7): 3, (29, 1, 8): 3, (5, 2, 7): 4, (5, 2, 8): 4, (25, 0, 0): 5, (25, 0, 1): 5, (25, 0, 2): 5, (25, 0, 9): 5, (25, 1, 3): 4, (25, 1, 4): 4, (25, 1, 5): 4, (25, 1, 6): 4, (25, 2, 7): 7, (25, 2, 8): 7, (26, 0, 0): 5, (26, 0, 1): 5, (26, 0, 2): 5, (26, 0, 9): 5, (26, 1, 3): 4, (26, 1, 4): 4, (26, 1, 5): 4, (26, 1, 6): 4, (26, 2, 7): 7, (26, 2, 8): 7, (27, 0, 0): 5, (27, 0, 1): 5, (27, 0, 2): 5, (27, 0, 9): 5, (27, 1, 3): 4, (27, 1, 4): 4, (27, 1, 5): 4, (27, 1, 6): 4, (27, 2, 7): 7, (27, 2, 8): 7, (28, 0, 0): 5, (28, 0, 1): 5, (28, 0, 2): 5, (28, 0, 9): 5, (28, 1, 3): 4, (28, 1, 4): 4, (28, 1, 5): 4, (28, 1, 6): 4, (28, 2, 7): 7, (28, 2, 8): 7, (29, 0, 0): 5, (29, 0, 1): 5, (29, 0, 2): 5, (29, 0, 9): 5, (29, 1, 3): 4, (29, 1, 4): 4, (29, 1, 5): 4, (29, 1, 6): 4, (29, 2, 7): 7, (29, 2, 8): 7, (30, 0, 3): 8, (30, 0, 4): 8, (30, 0, 5): 8, (30, 0, 6): 8, (30, 1, 7): 3, (30, 1, 8): 3, (31, 0, 3): 8, (31, 0, 4): 8, (31, 0, 5): 8, (31, 0, 6): 8, (31, 1, 7): 3, (31, 1, 8): 3, (32, 0, 3): 8, (32, 0, 4): 8, (32, 0, 5): 8, (32, 0, 6): 8, (32, 1, 7): 3, (32, 1, 8): 3, (33, 0, 3): 8, (33, 0, 4): 8, (33, 0, 5): 8, (33, 0, 6): 8, (33, 1, 7): 3, (33, 1, 8): 3, (34, 0, 3): 8, (34, 0, 4): 8, (34, 0, 5): 8, (34, 0, 6): 8, (34, 1, 7): 3, (34, 1, 8): 3, (35, 0, 3): 8, (35, 0, 4): 8, (35, 0, 5): 8, (35, 0, 6): 8, (35, 1, 7): 3, (35, 1, 8): 3}
changeOvers = {(0, 'enzyme0', 'enzyme0'): 0, (0, 'enzyme0', 'enzyme1'): 3, (0, 'enzyme0', 'enzyme2'): 1, (0, 'enzyme0', 'enzyme3'): 2, (0, 'enzyme0', 'enzyme4'): 2, (0, 'enzyme0', 'enzyme5'): 3, (0, 'enzyme1', 'enzyme0'): 1, (0, 'enzyme1', 'enzyme1'): 0, (0, 'enzyme1', 'enzyme2'): 1, (0, 'enzyme1', 'enzyme3'): 4, (0, 'enzyme1', 'enzyme4'): 3, (0, 'enzyme1', 'enzyme5'): 1, (0, 'enzyme2', 'enzyme0'): 1, (0, 'enzyme2', 'enzyme1'): 1, (0, 'enzyme2', 'enzyme2'): 0, (0, 'enzyme2', 'enzyme3'): 1, (0, 'enzyme2', 'enzyme4'): 3, (0, 'enzyme2', 'enzyme5'): 2, (0, 'enzyme3', 'enzyme0'): 2, (0, 'enzyme3', 'enzyme1'): 2, (0, 'enzyme3', 'enzyme2'): 3, (0, 'enzyme3', 'enzyme3'): 0, (0, 'enzyme3', 'enzyme4'): 4, (0, 'enzyme3', 'enzyme5'): 2, (0, 'enzyme4', 'enzyme0'): 2, (0, 'enzyme4', 'enzyme1'): 4, (0, 'enzyme4', 'enzyme2'): 3, (0, 'enzyme4', 'enzyme3'): 2, (0, 'enzyme4', 'enzyme4'): 0, (0, 'enzyme4', 'enzyme5'): 4, (0, 'enzyme5', 'enzyme0'): 1, (0, 'enzyme5', 'enzyme1'): 3, (0, 'enzyme5', 'enzyme2'): 1, (0, 'enzyme5', 'enzyme3'): 4, (0, 'enzyme5', 'enzyme4'): 3, (0, 'enzyme5', 'enzyme5'): 0, (1, 'enzyme0', 'enzyme0'): 0, (1, 'enzyme0', 'enzyme1'): 4, (1, 'enzyme0', 'enzyme2'): 1, (1, 'enzyme0', 'enzyme3'): 3, (1, 'enzyme0', 'enzyme4'): 3, (1, 'enzyme0', 'enzyme5'): 1, (1, 'enzyme1', 'enzyme0'): 1, (1, 'enzyme1', 'enzyme1'): 0, (1, 'enzyme1', 'enzyme2'): 2, (1, 'enzyme1', 'enzyme3'): 4, (1, 'enzyme1', 'enzyme4'): 3, (1, 'enzyme1', 'enzyme5'): 4, (1, 'enzyme2', 'enzyme0'): 1, (1, 'enzyme2', 'enzyme1'): 2, (1, 'enzyme2', 'enzyme2'): 0, (1, 'enzyme2', 'enzyme3'): 4, (1, 'enzyme2', 'enzyme4'): 3, (1, 'enzyme2', 'enzyme5'): 3, (1, 'enzyme3', 'enzyme0'): 4, (1, 'enzyme3', 'enzyme1'): 1, (1, 'enzyme3', 'enzyme2'): 1, (1, 'enzyme3', 'enzyme3'): 0, (1, 'enzyme3', 'enzyme4'): 1, (1, 'enzyme3', 'enzyme5'): 2, (1, 'enzyme4', 'enzyme0'): 3, (1, 'enzyme4', 'enzyme1'): 4, (1, 'enzyme4', 'enzyme2'): 4, (1, 'enzyme4', 'enzyme3'): 1, (1, 'enzyme4', 'enzyme4'): 0, (1, 'enzyme4', 'enzyme5'): 4, (1, 'enzyme5', 'enzyme0'): 1, (1, 'enzyme5', 'enzyme1'): 1, (1, 'enzyme5', 'enzyme2'): 1, (1, 'enzyme5', 'enzyme3'): 2, (1, 'enzyme5', 'enzyme4'): 1, (1, 'enzyme5', 'enzyme5'): 0, (2, 'enzyme0', 'enzyme0'): 0, (2, 'enzyme0', 'enzyme1'): 1, (2, 'enzyme0', 'enzyme2'): 3, (2, 'enzyme0', 'enzyme3'): 2, (2, 'enzyme0', 'enzyme4'): 3, (2, 'enzyme0', 'enzyme5'): 2, (2, 'enzyme1', 'enzyme0'): 4, (2, 'enzyme1', 'enzyme1'): 0, (2, 'enzyme1', 'enzyme2'): 1, (2, 'enzyme1', 'enzyme3'): 2, (2, 'enzyme1', 'enzyme4'): 3, (2, 'enzyme1', 'enzyme5'): 3, (2, 'enzyme2', 'enzyme0'): 3, (2, 'enzyme2', 'enzyme1'): 1, (2, 'enzyme2', 'enzyme2'): 0, (2, 'enzyme2', 'enzyme3'): 3, (2, 'enzyme2', 'enzyme4'): 4, (2, 'enzyme2', 'enzyme5'): 4, (2, 'enzyme3', 'enzyme0'): 3, (2, 'enzyme3', 'enzyme1'): 1, (2, 'enzyme3', 'enzyme2'): 2, (2, 'enzyme3', 'enzyme3'): 0, (2, 'enzyme3', 'enzyme4'): 3, (2, 'enzyme3', 'enzyme5'): 3, (2, 'enzyme4', 'enzyme0'): 1, (2, 'enzyme4', 'enzyme1'): 4, (2, 'enzyme4', 'enzyme2'): 1, (2, 'enzyme4', 'enzyme3'): 2, (2, 'enzyme4', 'enzyme4'): 0, (2, 'enzyme4', 'enzyme5'): 4, (2, 'enzyme5', 'enzyme0'): 4, (2, 'enzyme5', 'enzyme1'): 2, (2, 'enzyme5', 'enzyme2'): 1, (2, 'enzyme5', 'enzyme3'): 3, (2, 'enzyme5', 'enzyme4'): 3, (2, 'enzyme5', 'enzyme5'): 0, (3, 'enzyme0', 'enzyme0'): 0, (3, 'enzyme0', 'enzyme1'): 3, (3, 'enzyme0', 'enzyme2'): 3, (3, 'enzyme0', 'enzyme3'): 2, (3, 'enzyme0', 'enzyme4'): 1, (3, 'enzyme0', 'enzyme5'): 3, (3, 'enzyme1', 'enzyme0'): 4, (3, 'enzyme1', 'enzyme1'): 0, (3, 'enzyme1', 'enzyme2'): 3, (3, 'enzyme1', 'enzyme3'): 3, (3, 'enzyme1', 'enzyme4'): 1, (3, 'enzyme1', 'enzyme5'): 3, (3, 'enzyme2', 'enzyme0'): 2, (3, 'enzyme2', 'enzyme1'): 2, (3, 'enzyme2', 'enzyme2'): 0, (3, 'enzyme2', 'enzyme3'): 2, (3, 'enzyme2', 'enzyme4'): 1, (3, 'enzyme2', 'enzyme5'): 1, (3, 'enzyme3', 'enzyme0'): 3, (3, 'enzyme3', 'enzyme1'): 1, (3, 'enzyme3', 'enzyme2'): 1, (3, 'enzyme3', 'enzyme3'): 0, (3, 'enzyme3', 'enzyme4'): 4, (3, 'enzyme3', 'enzyme5'): 4, (3, 'enzyme4', 'enzyme0'): 3, (3, 'enzyme4', 'enzyme1'): 2, (3, 'enzyme4', 'enzyme2'): 3, (3, 'enzyme4', 'enzyme3'): 3, (3, 'enzyme4', 'enzyme4'): 0, (3, 'enzyme4', 'enzyme5'): 3, (3, 'enzyme5', 'enzyme0'): 1, (3, 'enzyme5', 'enzyme1'): 1, (3, 'enzyme5', 'enzyme2'): 3, (3, 'enzyme5', 'enzyme3'): 1, (3, 'enzyme5', 'enzyme4'): 4, (3, 'enzyme5', 'enzyme5'): 0, (4, 'enzyme0', 'enzyme0'): 0, (4, 'enzyme0', 'enzyme1'): 2, (4, 'enzyme0', 'enzyme2'): 4, (4, 'enzyme0', 'enzyme3'): 2, (4, 'enzyme0', 'enzyme4'): 1, (4, 'enzyme0', 'enzyme5'): 1, (4, 'enzyme1', 'enzyme0'): 2, (4, 'enzyme1', 'enzyme1'): 0, (4, 'enzyme1', 'enzyme2'): 3, (4, 'enzyme1', 'enzyme3'): 3, (4, 'enzyme1', 'enzyme4'): 2, (4, 'enzyme1', 'enzyme5'): 4, (4, 'enzyme2', 'enzyme0'): 1, (4, 'enzyme2', 'enzyme1'): 4, (4, 'enzyme2', 'enzyme2'): 0, (4, 'enzyme2', 'enzyme3'): 3, (4, 'enzyme2', 'enzyme4'): 4, (4, 'enzyme2', 'enzyme5'): 3, (4, 'enzyme3', 'enzyme0'): 2, (4, 'enzyme3', 'enzyme1'): 1, (4, 'enzyme3', 'enzyme2'): 2, (4, 'enzyme3', 'enzyme3'): 0, (4, 'enzyme3', 'enzyme4'): 2, (4, 'enzyme3', 'enzyme5'): 1, (4, 'enzyme4', 'enzyme0'): 4, (4, 'enzyme4', 'enzyme1'): 4, (4, 'enzyme4', 'enzyme2'): 2, (4, 'enzyme4', 'enzyme3'): 3, (4, 'enzyme4', 'enzyme4'): 0, (4, 'enzyme4', 'enzyme5'): 1, (4, 'enzyme5', 'enzyme0'): 1, (4, 'enzyme5', 'enzyme1'): 3, (4, 'enzyme5', 'enzyme2'): 1, (4, 'enzyme5', 'enzyme3'): 2, (4, 'enzyme5', 'enzyme4'): 1, (4, 'enzyme5', 'enzyme5'): 0, (5, 'enzyme0', 'enzyme0'): 0, (5, 'enzyme0', 'enzyme1'): 2, (5, 'enzyme0', 'enzyme2'): 2, (5, 'enzyme0', 'enzyme3'): 2, (5, 'enzyme0', 'enzyme4'): 1, (5, 'enzyme0', 'enzyme5'): 3, (5, 'enzyme1', 'enzyme0'): 4, (5, 'enzyme1', 'enzyme1'): 0, (5, 'enzyme1', 'enzyme2'): 2, (5, 'enzyme1', 'enzyme3'): 3, (5, 'enzyme1', 'enzyme4'): 4, (5, 'enzyme1', 'enzyme5'): 2, (5, 'enzyme2', 'enzyme0'): 3, (5, 'enzyme2', 'enzyme1'): 1, (5, 'enzyme2', 'enzyme2'): 0, (5, 'enzyme2', 'enzyme3'): 3, (5, 'enzyme2', 'enzyme4'): 4, (5, 'enzyme2', 'enzyme5'): 3, (5, 'enzyme3', 'enzyme0'): 2, (5, 'enzyme3', 'enzyme1'): 3, (5, 'enzyme3', 'enzyme2'): 3, (5, 'enzyme3', 'enzyme3'): 0, (5, 'enzyme3', 'enzyme4'): 1, (5, 'enzyme3', 'enzyme5'): 2, (5, 'enzyme4', 'enzyme0'): 1, (5, 'enzyme4', 'enzyme1'): 4, (5, 'enzyme4', 'enzyme2'): 1, (5, 'enzyme4', 'enzyme3'): 1, (5, 'enzyme4', 'enzyme4'): 0, (5, 'enzyme4', 'enzyme5'): 3, (5, 'enzyme5', 'enzyme0'): 4, (5, 'enzyme5', 'enzyme1'): 4, (5, 'enzyme5', 'enzyme2'): 3, (5, 'enzyme5', 'enzyme3'): 4, (5, 'enzyme5', 'enzyme4'): 4, (5, 'enzyme5', 'enzyme5'): 0, (6, 'enzyme0', 'enzyme0'): 0, (6, 'enzyme0', 'enzyme1'): 1, (6, 'enzyme0', 'enzyme2'): 1, (6, 'enzyme0', 'enzyme3'): 2, (6, 'enzyme0', 'enzyme4'): 3, (6, 'enzyme0', 'enzyme5'): 3, (6, 'enzyme1', 'enzyme0'): 4, (6, 'enzyme1', 'enzyme1'): 0, (6, 'enzyme1', 'enzyme2'): 2, (6, 'enzyme1', 'enzyme3'): 1, (6, 'enzyme1', 'enzyme4'): 3, (6, 'enzyme1', 'enzyme5'): 2, (6, 'enzyme2', 'enzyme0'): 4, (6, 'enzyme2', 'enzyme1'): 3, (6, 'enzyme2', 'enzyme2'): 0, (6, 'enzyme2', 'enzyme3'): 4, (6, 'enzyme2', 'enzyme4'): 3, (6, 'enzyme2', 'enzyme5'): 1, (6, 'enzyme3', 'enzyme0'): 3, (6, 'enzyme3', 'enzyme1'): 1, (6, 'enzyme3', 'enzyme2'): 3, (6, 'enzyme3', 'enzyme3'): 0, (6, 'enzyme3', 'enzyme4'): 2, (6, 'enzyme3', 'enzyme5'): 2, (6, 'enzyme4', 'enzyme0'): 2, (6, 'enzyme4', 'enzyme1'): 2, (6, 'enzyme4', 'enzyme2'): 2, (6, 'enzyme4', 'enzyme3'): 4, (6, 'enzyme4', 'enzyme4'): 0, (6, 'enzyme4', 'enzyme5'): 1, (6, 'enzyme5', 'enzyme0'): 2, (6, 'enzyme5', 'enzyme1'): 2, (6, 'enzyme5', 'enzyme2'): 3, (6, 'enzyme5', 'enzyme3'): 1, (6, 'enzyme5', 'enzyme4'): 3, (6, 'enzyme5', 'enzyme5'): 0, (7, 'enzyme0', 'enzyme0'): 0, (7, 'enzyme0', 'enzyme1'): 4, (7, 'enzyme0', 'enzyme2'): 2, (7, 'enzyme0', 'enzyme3'): 2, (7, 'enzyme0', 'enzyme4'): 2, (7, 'enzyme0', 'enzyme5'): 2, (7, 'enzyme1', 'enzyme0'): 2, (7, 'enzyme1', 'enzyme1'): 0, (7, 'enzyme1', 'enzyme2'): 3, (7, 'enzyme1', 'enzyme3'): 2, (7, 'enzyme1', 'enzyme4'): 1, (7, 'enzyme1', 'enzyme5'): 4, (7, 'enzyme2', 'enzyme0'): 1, (7, 'enzyme2', 'enzyme1'): 1, (7, 'enzyme2', 'enzyme2'): 0, (7, 'enzyme2', 'enzyme3'): 2, (7, 'enzyme2', 'enzyme4'): 4, (7, 'enzyme2', 'enzyme5'): 2, (7, 'enzyme3', 'enzyme0'): 3, (7, 'enzyme3', 'enzyme1'): 1, (7, 'enzyme3', 'enzyme2'): 3, (7, 'enzyme3', 'enzyme3'): 0, (7, 'enzyme3', 'enzyme4'): 2, (7, 'enzyme3', 'enzyme5'): 3, (7, 'enzyme4', 'enzyme0'): 2, (7, 'enzyme4', 'enzyme1'): 3, (7, 'enzyme4', 'enzyme2'): 4, (7, 'enzyme4', 'enzyme3'): 4, (7, 'enzyme4', 'enzyme4'): 0, (7, 'enzyme4', 'enzyme5'): 3, (7, 'enzyme5', 'enzyme0'): 4, (7, 'enzyme5', 'enzyme1'): 1, (7, 'enzyme5', 'enzyme2'): 1, (7, 'enzyme5', 'enzyme3'): 1, (7, 'enzyme5', 'enzyme4'): 3, (7, 'enzyme5', 'enzyme5'): 0, (8, 'enzyme0', 'enzyme0'): 0, (8, 'enzyme0', 'enzyme1'): 1, (8, 'enzyme0', 'enzyme2'): 2, (8, 'enzyme0', 'enzyme3'): 3, (8, 'enzyme0', 'enzyme4'): 2, (8, 'enzyme0', 'enzyme5'): 4, (8, 'enzyme1', 'enzyme0'): 1, (8, 'enzyme1', 'enzyme1'): 0, (8, 'enzyme1', 'enzyme2'): 1, (8, 'enzyme1', 'enzyme3'): 4, (8, 'enzyme1', 'enzyme4'): 3, (8, 'enzyme1', 'enzyme5'): 1, (8, 'enzyme2', 'enzyme0'): 4, (8, 'enzyme2', 'enzyme1'): 1, (8, 'enzyme2', 'enzyme2'): 0, (8, 'enzyme2', 'enzyme3'): 3, (8, 'enzyme2', 'enzyme4'): 2, (8, 'enzyme2', 'enzyme5'): 4, (8, 'enzyme3', 'enzyme0'): 3, (8, 'enzyme3', 'enzyme1'): 2, (8, 'enzyme3', 'enzyme2'): 4, (8, 'enzyme3', 'enzyme3'): 0, (8, 'enzyme3', 'enzyme4'): 2, (8, 'enzyme3', 'enzyme5'): 3, (8, 'enzyme4', 'enzyme0'): 3, (8, 'enzyme4', 'enzyme1'): 4, (8, 'enzyme4', 'enzyme2'): 1, (8, 'enzyme4', 'enzyme3'): 4, (8, 'enzyme4', 'enzyme4'): 0, (8, 'enzyme4', 'enzyme5'): 1, (8, 'enzyme5', 'enzyme0'): 4, (8, 'enzyme5', 'enzyme1'): 4, (8, 'enzyme5', 'enzyme2'): 1, (8, 'enzyme5', 'enzyme3'): 1, (8, 'enzyme5', 'enzyme4'): 2, (8, 'enzyme5', 'enzyme5'): 0, (9, 'enzyme0', 'enzyme0'): 0, (9, 'enzyme0', 'enzyme1'): 4, (9, 'enzyme0', 'enzyme2'): 2, (9, 'enzyme0', 'enzyme3'): 3, (9, 'enzyme0', 'enzyme4'): 2, (9, 'enzyme0', 'enzyme5'): 4, (9, 'enzyme1', 'enzyme0'): 2, (9, 'enzyme1', 'enzyme1'): 0, (9, 'enzyme1', 'enzyme2'): 3, (9, 'enzyme1', 'enzyme3'): 4, (9, 'enzyme1', 'enzyme4'): 3, (9, 'enzyme1', 'enzyme5'): 4, (9, 'enzyme2', 'enzyme0'): 4, (9, 'enzyme2', 'enzyme1'): 1, (9, 'enzyme2', 'enzyme2'): 0, (9, 'enzyme2', 'enzyme3'): 3, (9, 'enzyme2', 'enzyme4'): 4, (9, 'enzyme2', 'enzyme5'): 4, (9, 'enzyme3', 'enzyme0'): 3, (9, 'enzyme3', 'enzyme1'): 2, (9, 'enzyme3', 'enzyme2'): 4, (9, 'enzyme3', 'enzyme3'): 0, (9, 'enzyme3', 'enzyme4'): 2, (9, 'enzyme3', 'enzyme5'): 3, (9, 'enzyme4', 'enzyme0'): 3, (9, 'enzyme4', 'enzyme1'): 4, (9, 'enzyme4', 'enzyme2'): 4, (9, 'enzyme4', 'enzyme3'): 4, (9, 'enzyme4', 'enzyme4'): 0, (9, 'enzyme4', 'enzyme5'): 3, (9, 'enzyme5', 'enzyme0'): 4, (9, 'enzyme5', 'enzyme1'): 4, (9, 'enzyme5', 'enzyme2'): 1, (9, 'enzyme5', 'enzyme3'): 1, (9, 'enzyme5', 'enzyme4'): 3, (9, 'enzyme5', 'enzyme5'): 0}