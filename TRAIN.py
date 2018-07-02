import glob
import os
from pathlib import Path
from subprocess import Popen, PIPE

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

tests = glob.glob(os.path.join('tests_some', '*.xml'))
tests = sorted(tests, key=lambda s: s[-7:-4])

current_dir = str(Path().absolute())

first_placeholder = ""

import platform
if platform.system() is not "Windows":
    first_placeholder = "exec"

GUI = True
gui_placeholder = "external"

if GUI:
    gui_placeholder = "external_gui"

for test in tests:
    print(test, end=': ')
    server_command = '{} java  -Djava.endorsed.dirs={}/environment/lib -jar {}/environment/octopus-environment.jar {} {} 7777'.format(first_placeholder, current_dir, current_dir, gui_placeholder, (os.path.join(current_dir, test)))
    with Popen(server_command, shell=True, stdout=PIPE) as proc:

        # for line in proc.stdout:
        #     print(line)
        Handler().run(['localhost', '7777', '1', test])
        proc.kill()

        log_file = glob.glob('*.log')[0]
        score = calculate_score(log_file)

        scores.append(score)
        print('%s:\t%.2f' % (test, score))

        time.sleep(0.1)  # HACK
        os.remove(log_file)

print('mean: %.2f' % np.mean(scores))
