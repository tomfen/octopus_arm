import glob
import os
from subprocess import Popen

import time

from agent_handler import Handler
import numpy as np


def calculate_score(log_file_path):
    with open(log_file_path) as file:
        score = sum(float(x) for x in file.read().split())
    return score

for log_file in glob.iglob('*.log'):
    os.remove(log_file)

scores = []

tests = glob.glob(os.path.join('tests', '*.xml'))
tests = sorted(tests, key=lambda s: s[-7:-5])

for test in tests:
    server_command = 'javaw -jar ./octopus-environment.jar external_gui %s 7777' % test
    with Popen(server_command) as proc:
        Handler().run(['localhost', '7777', '1', test])
        proc.kill()

        log_file = glob.glob('*.log')[0]
        score = calculate_score(log_file)

        scores.append(score)
        print('%s:\t%.2f' % (test, score))

        time.sleep(0.1)  # HACK
        os.remove(log_file)

print('mean: %.2f' % np.mean(scores))
