import glob
import os
from pathlib import Path
from subprocess import Popen, PIPE

import time
import datetime

from agent_handler import Handler
import numpy as np


def calculate_score(log_file_path):
    with open(log_file_path) as file:
        score = sum(float(x) for x in file.read().split())
    return score

for log_file in glob.iglob('*.log'):
    os.remove(log_file)

scores = []
current_dir = str(Path().absolute())
tests = glob.glob(os.path.join(current_dir,'tests', '*.xml'))#('tests_some', '*.xml'))
tests = sorted(tests, key=lambda s: s[-7:-4])

every_4th = tests[::4]

# every_4th_below = [e for e in every_4th if "-" in e]
# every_4th_above =  [e for e in every_4th if "-" not in e]
# every_4th_since_130 = every_4th[:130]
# every_4th_below = [every_4th_above[-5]]

tests = every_4th


first_placeholder = ""

import platform
if platform.system() is not "Windows":
    first_placeholder = "exec"

GUI = False
MODE = "external"

if GUI:
    MODE = "external_gui"


generation_file_path = "generation_means"
# for generation in range(0, 3):
print("started at {}".format(datetime.datetime.now()))
for test in tests:
    print(test, end=': ')
    server_command = '{} java  -Djava.endorsed.dirs={}/environment/lib -jar {}/environment/octopus-environment.jar {} {} 7777'.format(first_placeholder, current_dir, current_dir, MODE, (os.path.join(current_dir, test))
                                                                                                                                      #, generation
                                                                                                                                      )
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

print('generation__mean: {}'.format(
    #generation,
    np.mean(scores)) )
with open(generation_file_path,"a") as f_g:
    f_g.write('{} generation__mean: {} \n'.format(
    #generation,
        datetime.datetime.now(),
    np.mean(scores)) )
print("finished at {}".format(datetime.datetime.now()))
