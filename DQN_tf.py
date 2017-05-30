# coding: utf-8
from collections import deque
import os

import numpy as np
import time

import tensorflow as tf

import gym
from gym import wrappers

np.random.seed(7)

# 過去何コマを見るか
STATE_NUM = 4


class DQNAgent():
    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, epsilon=0.99):
        # parameters
        # self.name = os.path.splitext(os.path.basename(__file__))[0]
        # self.environment_name = environment_name
        self.enable_actions = [0,1]
        self.n_actions = len(self.enable_actions)
        self.minibatch_size = 32
        self.replay_memory_size = 300*100
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.exploration = 0.1
        self.epsilon = epsilon
        self.experienceMemory=[] # 経験メモリ（エピソードローカル）
        self.experienceMemory_local=[] # 経験メモリ（エピソードローカル）
        self.memSize = 300*100  # 経験メモリのサイズ(300サンプリングx100エピソード)
        self.experienceMemory_local=[] # 経験メモリ（エピソードローカル）
        self.memPos = 0 #メモリのインデックス
        self.batch_num = 32 # 学習に使うバッチサイズ
        self.gamma = 0.9       # 割引率
        self.loss=0
        self.total_reward_award=np.ones(100)*-1000 #100エピソード



        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)

        # model
        self.init_model()

        # variables
        self.current_loss = 0.0

    def init_model(self):

        # input layer (1 x 4)
        self.x = tf.placeholder(tf.float32, [4])

        # flatten (64)
        x_flat = tf.reshape(self.x, [-1, 4])

        # fully connected layer (32)
        W_fc1 = tf.Variable(tf.truncated_normal([4 ,16], stddev=0.01))
        b_fc1 = tf.Variable(tf.zeros([16]))
        h_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

        # output layer (n_actions)
        W_out = tf.Variable(tf.truncated_normal([16, self.n_actions], stddev=0.01))
        b_out = tf.Variable(tf.zeros([self.n_actions]))
        self.y = tf.matmul(h_fc1, W_out) + b_out

        # loss function
        self.y_ = tf.placeholder(tf.float32, [self.n_actions])
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))

        # train operation
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.training = optimizer.minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_action_value(self, state):
        #Q_values(self, state): #
        # Q(state, action) of all actions
        return self.sess.run(self.y, feed_dict={self.x: state})[0]

    def reduce_epsilon(self):
        self.epsilon-=1.0/100000

    def get_epsilon(self):
        return self.epsilon

    def get_action(self, state, train):
        if train==True and np.random.rand() < self.epsilon:
            # random
            return np.random.choice(self.enable_actions)
        else:
            # max_action Q(state, action)
            return self.enable_actions[np.argmax(self.get_action_value(state))]

    def experience_local(self,old_seq, action, reward, new_seq):
        #エピソードローカルな記憶
        self.experienceMemory_local.append( np.hstack([old_seq,action,reward,new_seq]) )

    def store_experience(self, state, action, reward, state_1, train):
        self.D.append((state, action, reward, state_1, train))


    def experience_global(self,total_reward):
        #グローバルな記憶
        #ベスト100に入る経験を取り込む
        if np.min(self.total_reward_award)<total_reward:
            i=np.argmin(self.total_reward_award)
            self.total_reward_award[i]=total_reward

            # GOOD EXPERIENCE REPLAY
            for x in self.experienceMemory_local:
                self.experience( x )

        #一定確率で優秀でないものも取り込む
        if np.random.random()<0.01:
            # # NORMAL EXPERIENCE REPLAY
            for x in self.experienceMemory_local:
                self.experience( x )

        self.experienceMemory_local=[]

    def experience(self,x):
        if len(self.experienceMemory)>self.memSize:
            self.experienceMemory[int(self.memPos%self.memSize)]=x
            self.memPos+=1
        else:
            self.experienceMemory.append( x )

    def experience_replay(self):
        state_minibatch = []
        y_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = self.enable_actions.index(action_j)
            state_j
            y_j = self.get_action_value(state_j)

            if terminal:
                y_j[action_j_index] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action')
                y_j[action_j_index] = reward_j + self.discount_factor * np.max(self.get_action_value(state_j_1))  # NOQA

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)

        # training
        self.sess.run(self.training, feed_dict={self.x: state_minibatch, self.y_: y_minibatch})

        # for log
        self.current_loss = self.sess.run(self.loss, feed_dict={self.x: state_minibatch, self.y_: y_minibatch})


    def update_model(self,old_seq, action, reward, new_seq):
        '''
        モデルを更新する
        '''
        # 経験メモリにたまってない場合は更新しない
        if len(self.experienceMemory)<self.batch_num:
            return

        # 経験メモリからバッチを作成
        memsize=len(self.experienceMemory)
        batch_index = list(np.random.randint(0,memsize,(self.batch_num)))
        batch =np.array( [self.experienceMemory[i] for i in batch_index ])
        x = batch[:,0:STATE_NUM].reshape( (self.batch_num,-1)).astype(np.float32)
        targets=[]
        for j in range(x.shape[0]):
            targets.append(np.ndarray.tolist(self.get_action_value(x[j])))
        targets = np.array(targets)

        for i in range(self.batch_num):
            #[ seq..., action, reward, seq_new]
            a = batch[i,STATE_NUM]
            r = batch[i, STATE_NUM+1]
            ai=int((a+1)/2) #±1 をindex(0,1)に。
            new_seq= batch[i,(STATE_NUM+2):(STATE_NUM*2+2)]
            targets[i,ai]=( r+ self.gamma * np.max(self.get_action_value(new_seq)))
        t =targets

        # training
        for i in range(x.shape[0]):
            x1 = x[i]
            t2 = t[i]
            self.sess.run(self.training, feed_dict={self.x: x1, self.y_: t2})
            # for log
            self.current_loss = self.sess.run(self.loss, feed_dict={self.x: x1 ,self.y_: t2})


class pendulumEnvironment():
    '''
    model
    '''
    def __init__(self):
        self.env = wrappers.Monitor(gym.make('CartPole-v0'), './private/tmp/cartpole-experiment-3', force = True)

    def reset(self):
        self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def monitor_close(self):
        self.env.close()

# シミュレータ。
class simulator:
    def __init__(self, environment, agent):
        self.agent = agent
        self.env = environment
        self.num_seq=STATE_NUM
        self.reset_seq()
        self.learning_rate=1.0
        self.highscore=0
        self.log=[]

    def reset_seq(self):
        self.seq=np.zeros(self.num_seq)

    def push_seq(self, state):
        self.seq[1:self.num_seq]=self.seq[0:self.num_seq-1]
        self.seq[0]=state

    def run(self, train=True):

        self.env.reset()
        self.reset_seq()
        total_reward=0

        for i in range(300):
            # 現在のstateからなるシーケンスを保存
            old_seq = self.seq.copy()

            # エージェントの行動を決める
            action = self.agent.get_action(old_seq,train)

            # 環境に行動を入力する
            observation, reward, done, info =  self.env.step(action)
            total_reward +=reward

            # 結果を観測してstateとシーケンスを更新する
            state = observation[2]
            self.push_seq(state)
            new_seq = self.seq.copy()

            # エピソードローカルなメモリに記憶する
            self.agent.store_experience(old_seq, action, reward, new_seq,train)
            self.agent.experience_local(old_seq, action, reward, new_seq)

            if done:
                print("Episode finished after {} timesteps".format(i+1))
                break

        # エピソードローカルなメモリ内容をグローバルなメモリに移す
        self.agent.experience_global(total_reward)

        if train:
            # 学習用メモリを使ってモデルを更新する
            self.agent.update_model(old_seq, action, reward, new_seq)
            # self.agent.experience_replay()
            self.agent.reduce_epsilon()

        return total_reward

if __name__ == '__main__':
    agent=DQNAgent()
    env=pendulumEnvironment()
    sim=simulator(env,agent)

    best_reward1 = 0
    for i in range(1000):
        total_reward1 = sim.run(train=True)
        if best_reward1 < total_reward1:
            best_reward1 = total_reward1

        print(str(i) + " " + str(total_reward1) + " " + str(best_reward1))
        env.reset()

        if best_reward1 > 195:
            break

    env.monitor_close()
    gym.upload('./private/tmp/cartpole-experiment-3', api_key='your_key')
