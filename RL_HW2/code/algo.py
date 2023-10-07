import numpy as np
from arguments import get_args

from abc import abstractmethod


class QAgent:
    def __init__(self,):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


class Myagent:
    def __init__(self):
        super(Myagent, self).__init__()
        self.args = get_args()
        self.lr = self.args.lr
        self.gamma = self.args.gamma
        self.Qtable = np.zeros((8, 8, 4))

    def update(self, ob, action, ob_next, reward, done):
        x, y = ob.astype(np.int32)
        x1, y1 = ob_next.astype(np.int32)

        Q_predict = self.Qtable[x, y, action]

        if done:
            Q_target = reward
        else:
            Q_target = reward + self.gamma * np.max(self.Qtable[x1, y1, :])

        self.Qtable[x, y, action] += self.lr * (Q_target - Q_predict)

    def select_action(self, ob):
        x, y = ob.astype(np.int32)
        return np.argmax(self.Qtable[x, y, :])
