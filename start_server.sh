#!/usr/bin/env bash

kill `ps|grep environment/octopus-environment.jar|cut -d' ' -f1|xargs` 2>/dev/null
sleep 1


#java -Djava.endorsed.dirs=./lib -jar octopus-environment.jar INTERNAL settings.xml 1000
java  -Djava.endorsed.dirs=./environment/lib -jar ./environment/octopus-environment.jar external_gui ./tests_sample/test_0.200.xml 7777