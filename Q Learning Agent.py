import random

import gym
import time
import numpy as np


# 'query' means that the process to find the action with maximum q value for each state
# 'sample' means the process to select the final action after considering epsilon
# 'learn' means the process to update the Q value of each action in each state

import gym
import numpy as np
import pandas as pd

class QLearningAgent_aggregate(object):
    def __init__(self, obs_n, act_n, learning_rate=0.05, gamma=0.9, e_greed=0.1):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = {}
        self.agg = {}

    def state_aggregate(self,state):

        index= state[0].sum() + 2*state[1].sum()
        #print('state',state)
        #print('index',index)
        if index not in self.agg.keys():
            self.agg[index] = (state[0], state[1])
        return index


    # including exploration
    def select_action(self,obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            if not self.Q.__contains__(obs):
                self.Q[obs] = np.zeros(env.nA)

            Q_list = self.Q[obs]
            # print('Qlist',Q_list)
            maxQ = np.max(Q_list)
            # print('maxQ',maxQ)
            action_list = np.where(Q_list == maxQ)[0]
            action = np.random.choice(action_list)
        else:
            action = env.action_choice(env.size)
        return action

# generate an index to represent a state
    def hash(self,obs):
        total = 0
        #print('obs_F',obs[0])

        for i in range(env.size):
            i=np.unravel_index(i,env.shape)
            total=31*total+(obs[0][i[0],i[1]]*obs[1][i[0],i[1]])

        #print('total',total)
        #print('%',total%env.size)

        return total%env.nS

    def query(self, obs):
        if not self.Q.__contains__(obs):
            self.Q[obs] = np.zeros(env.nA)

        Q_list = self.Q[obs]
        # print('Qlist',Q_list)
        maxQ = np.max(Q_list)
        # print('maxQ',maxQ)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        return action



    def learn(self, obs, action, reward, next_obs, done):

        if not self.Q.__contains__(obs):
            self.Q[obs]=np.zeros(env.nA)
        if not self.Q.__contains__(next_obs):
            self.Q[next_obs]=np.zeros(env.nA)

        query_Q = self.Q[obs][action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs]) # Q-learning
        self.Q[obs][action] += self.lr * (target_Q - query_Q)


    # save Q table in a file
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # read data in files to q table
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')


        
        
        

########################################

class QLearningAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.05, gamma=0.9, e_greed=0.1):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = {}

    # including exploration
    def sample(self,obs):
        #print('sampleX',env.X)
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.query(obs)
        else:
            action = env.action_choice(env.size)
        return action

# generate an index to represent a state
    def hash(self,obs):
        total = 0
        #print('obs_F',obs[0])

        for i in range(env.size):
            i=np.unravel_index(i,env.shape)
            total=31*total+(obs[0][i[0],i[1]]*obs[1][i[0],i[1]])

        #print('total',total)
        #print('%',total%env.size)
        #print('hash',total)
        #print('ns',env.nS)
        return total%env.nS


    def query(self, obs):
        if not self.Q.__contains__(obs):
            self.Q[obs]=np.zeros(env.nA)
        #print('queryX', env.X)
        Q_list = self.Q[obs]
        #print('Qlist',Q_list)
        maxQ = np.max(Q_list)
        #print('maxQ',maxQ)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        return action


    def learn(self, obs, action, reward, next_obs, done):

        if not self.Q.__contains__(obs):
            self.Q[obs]=np.zeros(env.nA)
        if not self.Q.__contains__(next_obs):
            self.Q[next_obs]=np.zeros(env.nA)

        query_Q = self.Q[obs][action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs]) # Q-learning
        self.Q[obs][action] += self.lr * (target_Q - query_Q)



    # save Q table in a file
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # read data in files to q table
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')




##################################################

class QLearningAgent_MDPs(object):
    def __init__(self, obs_n, act_n, learning_rate=0.05, gamma=0.9, e_greed=0.1):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q1 = {}
        self.Q2 = {}
        self.Q3 = {}
        self.Q4 = {}
    # including exploration
    def sample(self,obs,size,table_i):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.query(obs,size,table_i)  #input和sample相同，output是0,1，2,3
        else:
            action = env.action_choice(size) # 根据localMDP的size随机选取action（在这里是0,1,2,3）
            #print('sample',action)
        return action

# generate an index to represent a state
    def hash(self,obs,MDP_Shape):
        total = 0
        #print('obs_F',obs[0])
        MDP_size = np.product(MDP_Shape)
        for i in range(MDP_size):
            i=np.unravel_index(i,MDP_Shape)
            total=31*total+(obs[0][i[0],i[1]]*obs[1][i[0],i[1]])

        #print('total',total)
        #print('%',total%env.size)
        nS = env.NumberOfStates(MDP_size)
        #print('nS',nS)
        return total%nS



    def query(self, obs,size,table_i):

        if table_i == 0:  
            q_table = self.Q1
        elif table_i == 1:
            q_table = self.Q2
        elif table_i == 2:
            q_table = self.Q3
        else:
            q_table = self.Q4
        #####
        
        ##### 
        if not q_table.__contains__(obs):
            q_table[obs]=np.zeros(size)

        Q_list = q_table[obs]
        #print('Qlist',Q_list)
        maxQ = np.max(Q_list)
        #print('maxQ',maxQ)
        #print()
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        #print('query',action)
        #action_candidacy = []
        #for i in range(env.size):
         #   if (i % env.single_action_space) == 0:
          #      action_candidacy.append(i)
        return action

    def test_query(self,obs1,obs2,obs3,obs4,size):


        if not self.Q1.__contains__(obs1):
            self.Q1[obs1]=np.zeros(size)
        if not self.Q2.__contains__(obs2):
            self.Q2[obs2]=np.zeros(size)
        if not self.Q3.__contains__(obs3):
            self.Q3[obs3]=np.zeros(size)
        if not self.Q4.__contains__(obs4):
            self.Q4[obs4]=np.zeros(size)

        actions_list = {}

        Q_list = self.Q1[obs1]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        actions_list[(0,action)] = maxQ

        Q_list = self.Q2[obs2]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        actions_list[(1,action)] = maxQ

        Q_list = self.Q3[obs3]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        actions_list[(2,action)] = maxQ

        Q_list = self.Q4[obs4]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        actions_list[(3,action)] = maxQ

        return actions_list


    def learn(self, obs, action, reward, next_obs, done,table_i,size):

        if table_i == 0:
            q_table = self.Q1
        elif table_i == 1:
            q_table = self.Q2
        elif table_i == 2:
            q_table = self.Q3
        else:
            q_table = self.Q4

        if not q_table.__contains__(obs):
            q_table[obs] = np.zeros(size)
        if not q_table.__contains__(next_obs):
            q_table[next_obs]=np.zeros(size)
        #print('queryac',action)
        query_Q = q_table[obs][action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(q_table[next_obs]) # Q-learning
        q_table[obs][action] += self.lr * (target_Q - query_Q)

        if table_i == 0:
            self.Q1 = q_table
        elif table_i == 1:
            self.Q2 = q_table
        elif table_i == 2:
            self.Q3 = q_table
        else:
            self.Q4 = q_table

    # save Q table in a file
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # read data in files to q table
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')

# train

##############################################


def run_episode(env, agent):
    total_steps = 0
    total_reward = 0

    env.reset()
    obs=(env.F,env.X)
    obs = agent.hash(obs)

    while True:

        #print('x', env.X)
        #print('f',env.F)
        #print('C',env.C)

        action = agent.sample(obs)
        #print('action',action)

        next_obs,reward,done, info=env.step(action)

        next_obs=agent.hash(next_obs)
        #print('X', env.X)
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs
        total_reward += reward

        total_steps += 1
        if done:
            break

        #print('F', env.F)

        #print('C', env.C)
    return total_reward, total_steps

def run_episode_LocalMDPs(env, agent,times,totalTimes):
    total_steps = 0
    total_reward = 0
    NumOfMDP = 4
    localMDP_length = env.action_space_length
    x = 0
    y = 0 
    localMDP_shape = (localMDP_length,localMDP_length)
    localMDP_size = np.prod(localMDP_shape)
    numOfLocalMDP_row = env.shape[0] / env.action_space_length

    env.reset()


    if times < totalTimes/NumOfMDP:
        localMDP_index = env.MDP_indices[0]
        table_index = 0
    elif times < totalTimes / NumOfMDP * 2:
        localMDP_index = env.MDP_indices[1]
        table_index = 1
    elif times < totalTimes / NumOfMDP * 3:
        localMDP_index = env.MDP_indices[2]
        table_index = 2
    else:
        localMDP_index = env.MDP_indices[3]
        table_index = 3

    localMDP_index = np.unravel_index(localMDP_index,env.shape)
    index_x = localMDP_index[0]
    index_y = localMDP_index[1]
    obs = local_MDP_stateupdate(env,localMDP_length,index_x,index_y,localMDP_shape)
    obs = agent.hash(obs,localMDP_shape)


    while True:

        local_action = agent.sample(obs,localMDP_size,table_index) 
        #print('action',action)

        action = np.unravel_index(local_action, localMDP_shape)

        col = table_index % numOfLocalMDP_row * env.action_space_length  # 
        row = int(table_index / numOfLocalMDP_row) * env.action_space_length


        action0 = action[0] + row
        action1 = action[1] + col

        #print('action0',action0)
        #print('action1',action1)

        action = np.ravel_multi_index((int(action0), int(action1)), env.shape)

        next_obs,reward,done, info=env.step(action)

        next_obs = local_MDP_stateupdate(env,localMDP_length,index_x,index_y,localMDP_shape)
        next_obs=agent.hash(next_obs,localMDP_shape) #

        agent.learn(obs, local_action, reward, next_obs, done, table_index,localMDP_size)

        obs = next_obs
        total_reward += reward
        total_steps += 1
        if done:
            break

        #print('F',env.F)
        #print('X', env.X)
        #print('C', env.C)
    return total_reward, total_steps

def local_MDP_stateupdate(env,length,x_index, y_index,shape):
    local_F = np.random.random_integers(0,0,size=shape)
    local_X = np.random.random_integers(0, 0, size=shape)
    for x in range(length):
        for y in range(length):
            local_x = x_index + x
            local_y = y_index + y
            local_F[x,y] = env.F[local_x,local_y]
            local_X[x,y] = env.X[local_x,local_y]

    obs = (local_F,local_X)
    return obs

def run_episode_aggregate(env, agent):
    total_steps = 0
    total_reward = 0

    env.reset()
    obs=(env.F,env.X)
    obs = agent.state_aggregate(obs)

    while True:

        action = agent.select_action(obs)
        #print('action',action)
        next_obs,reward,done,info=env.step(action)
        next_obs=agent.state_aggregate(next_obs)

        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs
        total_reward += reward
        total_steps += 1
        if done:
            break

        #print('F',env.F)
        #print('X', env.X)
        #print('C', env.C)
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    testing_number = 5000

    for i in range(testing_number):
        env.reset()
        obs = (env.F, env.X)

        #print('obs', obs)
        while True:
            obs = agent.hash(obs)
            action = agent.query(obs)
            print('action',action)
            next_obs, reward, done, info = env.step(action)
            #print('next_obs,', next_obs)
            #print('reward',reward)
            #print('done', done)

            total_reward += reward
            obs = next_obs
            if done:
                break

    return total_reward/testing_number

def test_episode_MDPs(env, agent):
    total_reward = 0
    testing_number = 5000

    localMDP_index1 = np.unravel_index(env.MDP_indices[0], env.shape)
    localMDP_index2 = np.unravel_index(env.MDP_indices[1], env.shape)
    localMDP_index3 = np.unravel_index(env.MDP_indices[2], env.shape)
    localMDP_index4 = np.unravel_index(env.MDP_indices[3], env.shape)
    localMDP_shape = (env.action_space_length,env.action_space_length)
    localMDP_size = np.prod(localMDP_shape)
    numOfLocalMDP_row = env.shape[0] / env.action_space_length

    for i in range(testing_number):
        env.reset()
        obs = (env.F, env.X)
        #print('obs', obs)
        while True:
            obs1 = local_MDP_stateupdate(env,env.action_space_length,localMDP_index1[0],localMDP_index1[1],localMDP_shape)
            obs2 = local_MDP_stateupdate(env,env.action_space_length,localMDP_index2[0],localMDP_index2[1],localMDP_shape)
            obs3 = local_MDP_stateupdate(env,env.action_space_length,localMDP_index3[0],localMDP_index3[1],localMDP_shape)
            obs4 = local_MDP_stateupdate(env,env.action_space_length,localMDP_index4[0],localMDP_index4[1],localMDP_shape)

            print('obs',obs)
            #print('obs1',obs1)
            #print('obs2',obs2)
            #print('obs3',obs3)
            #print('obs4',obs4)

            obsq = agent.hash(obs,env.shape)
            obs1 = agent.hash(obs1,localMDP_shape)
            obs2 = agent.hash(obs2,localMDP_shape)
            obs3 = agent.hash(obs3,localMDP_shape)
            obs4 = agent.hash(obs4,localMDP_shape)
            print('obs1',obs1)
            print('obs2',obs2)
            print('obs3',obs3)
            print('obs4',obs4)
            print('obsq',obsq)

            action_list = agent.test_query(obs1,obs2,obs3,obs4,localMDP_size)

            #print('action_list',action_list)

            ##### 本来想每个test——query只读取一个obs；动态变量名
            '''
            action_list = {}
            for i in range(4):
                action,qValue = agent.test_query(env,)
                action_list[]
            '''
            action = max(action_list, key=lambda x: action_list[x])
            #print('action',action)
            table_index = action[0]
            local_action = action[1]
            local_action = np.unravel_index(local_action,localMDP_shape)
            #print('localaction',local_action)


            col = table_index % numOfLocalMDP_row * env.action_space_length # 从local复原到full grid之下的（x,y）坐标
            row = int(table_index / numOfLocalMDP_row) * env.action_space_length

            #print('row',row)
            #print('col',col)
            action0 = local_action[0] + row
            action1 = local_action[1] + col

            #print('finalaction',(action0,action1))
            action = np.ravel_multi_index((int(action0),int(action1)),env.shape)
            print('action',action)

            #print('action',action)
            next_obs, reward, done, info = env.step(action)
            #print('next_obs,', next_obs)
            #print('reward',reward)
            #print('done', done)

            total_reward += reward
            obs = next_obs
            if done:
                break

    return total_reward/testing_number


def test_episode_aggregate(env, agent):
    total_reward = 0
    testing_number = 5000

    for i in range(testing_number):
        env.reset()
        obs = (env.F, env.X)
        #print('aobs', obs)
        while True:
            obs = agent.state_aggregate(obs)
            action = agent.query(obs)
            #print('action',action)
            next_obs, reward, done, info= env.step(action)
            #print('next_obs,', next_obs)
            #print('reward',reward)
            #print('done', done)

            total_reward += reward
            obs = next_obs
            if done:
                break

    return total_reward/testing_number

def random_test_episode(env):
    total_reward = 0
    testing_number = 5000

    for i in range(testing_number):
        env.reset()
        #print('robs',(env.F,env.X))
        while True:
            action = env.action_choice(env.size)
            #print('action', action)
            next_obs, reward, done,info = env.step(action)
            #print('next_obs,', next_obs)
            #print('reward', reward)
            #print('done', done)

            total_reward += reward
            if done:
                break


    return total_reward/testing_number


env = gym.make("Wildfire-v4")


# create an agent
agent = QLearningAgent(
    obs_n=env.nS,
    act_n=env.nA,
    learning_rate=0.1,
    gamma=0.90,
    e_greed=0.1)

agent_aggregate = QLearningAgent_aggregate(
    obs_n=env.nS,
    act_n=env.nA,
    learning_rate=0.1,
    gamma=0.90,
    e_greed=0.1)

agent_MDPs = QLearningAgent_MDPs(
    obs_n=env.nS,
    act_n=env.nA,
    learning_rate=0.1,
    gamma=0.90,
    e_greed=0.1)

# training
for episode in range(20000):
    ep_reward, ep_steps = run_episode(env, agent)
    print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

    ep_reward, ep_steps = run_episode_LocalMDPs(env,agent_MDPs,episode,20000)
    print('LocalMDPs:Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

    #ep_reward, ep_steps = run_episode_aggregate(env, agent_aggregate)
    #print('Agg:Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))
    #ep_reward, ep_steps = run_episode_action_aggregate(env, agent_action_aggregate)
    #print('Action_Agg:Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

# Test the results of training
test_reward = test_episode(env, agent)
random_reward = random_test_episode(env)
#test_reward_aggregate = test_episode_aggregate(env, agent_aggregate)
test_reward_localMDPs = test_episode_MDPs(env, agent_MDPs)


print('test reward = %.1f' % (test_reward))
print('random test reward= %.1f' % (random_reward))
#print('agg test reward= %.1f' % (test_reward_aggregate))
print('localMDPs test reward= %.1f' % (test_reward_localMDPs))

df = pd.DataFrame([agent.Q])
df.to_csv('D:/2021Fall/RL/DP/Aggregation/Qtable.csv')
#print('Q_table',agent.Q)
print('length', len(agent.Q))
#print('agg_Q_table',agent_aggregate.Q)
#print('agg_length', len(agent_aggregate.Q))
#print('action_agg_length', len(agent_action_aggregate.Q))
print('local1', agent_MDPs.Q1)
print('local2', agent_MDPs.Q2)
print('local3', agent_MDPs.Q3)
print('local4', agent_MDPs.Q4)

print('locallength1', len(agent_MDPs.Q1))
print('locallength2', len(agent_MDPs.Q2))
print('locallength3', len(agent_MDPs.Q3))
print('locallength4', len(agent_MDPs.Q4))
print('C',env.C)







