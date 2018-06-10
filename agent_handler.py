# Echo client program
import socket
import sys
import random

import time

import Agent

LINE_SEPARATOR = b'\n'
BUF_SIZE = 4096  # in bytes


class Handler:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def getArgs(self, args):
        try:
            return args[0], int(args[1]), int(args[2]), 'sample_agents/weights.txt'
        except ValueError:
            print('usage: python agent_handler <server> <port> <num-episodes> [<agent-specific parameters>]')
            sys.exit(1)

    def connect(self, host, port):
        while True:
            try:
                self.sock.connect((host, port))
                return
            except socket.error:
                time.sleep(0.2)

    def sendStr(self, s):
        self.sock.send(s.encode() + LINE_SEPARATOR)

    def receive(self, numTokens):
        data = [b'']
        while len(data) <= numTokens:
            rawData = data[-1] + self.sock.recv(BUF_SIZE)
            del data[-1]
            data = data + rawData.split(LINE_SEPARATOR)

        del data[-1]
        return data

    def sendAction(self, action):
        # sends all the components of the action one by one
        for a in action:
            self.sendStr(str(a).replace('.', ',', 1))

    def run(self, args):

        host, port, numEpisodes, agentParams = self.getArgs(args)

        self.connect(host, port)

        self.sendStr('GET_TASK')
        data = self.receive(2)
        stateDim = int(data[0])
        actionDim = int(data[1])

        # instantiate agent
        agent = Agent.Agent(stateDim, actionDim, agentParams)

        self.sendStr('START_LOG')
        self.sendStr(agent.getName())

        for i in range(numEpisodes):
            self.sendStr('START')
            data = self.receive(2 + stateDim)

            terminalFlag = int(data[0])
            state = [float(x) for x in data[2:]] #map(float, data[2:])
            action = agent.start(state)

            while not terminalFlag:
                self.sendStr('STEP')
                self.sendStr(str(actionDim))
                self.sendAction(action)

                data = self.receive(3 + stateDim)
                if not (len(data) == stateDim + 3):
                    print('Communication error: calling agent.cleanup()')
                    agent.cleanup()
                    sys.exit(1)

                reward = float(data[0])
                terminalFlag = int(data[1])
                state = [float(x) for x in data[3:]] #map(float, data[3:])

                if terminalFlag == 0:
                    action = agent.step(reward, state)
                else:
                    agent.end(reward)
