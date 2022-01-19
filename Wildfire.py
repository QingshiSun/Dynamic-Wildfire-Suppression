
import numpy as np
import sys
from gym.envs.toy_text import discrete
from scipy.stats import bernoulli
import math
from scipy.special import comb
import copy
import random

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

#const WIND_DIR_DICT = (north=0,
#                         north_east=1,
#                         east=2,
#                         south_east=3,
#                         south=4,
#                         south_west=5,
#                         west=6,
#                         north_west=7)

class WildFireEnv_local(discrete.DiscreteEnv):
    def __init__(self):
        self.shape=(40,40)
        self.map=np.zeros(self.shape)
        self.size=np.prod(self.shape)

        self.action_space_length = 2 
        self.single_action_space = self.action_space_length * self.action_space_length

        #
        self.action_aggregate_nA = int(self.size/self.single_action_space)

        self.initial_wind=[1,1,4]
        self.initial_burn_procent=0.3
        self.default_fuel= 3
        self.suppression_prob=0.9
        # high_cost_perc = 0.2
        # mid_cost_perc = 0.3 # percentage of California forest coverage
        # low_cost_perc = 1 - high_cost_perc - mid_cost_perc # 0.5
        self.cost_percent=[0.2,0.3,0.5]
        self.cost_values=[-10,-5,-2]

        self.MDP_indices = self.action_aggregate() #每个local mdp的左上角cell的index
        #print('MDP_indices',self.MDP_indices)

        self.action_size=1
        self.nS= self.NumberOfStates(self.size)


        self.nA=comb(self.shape[0]*self.shape[1],self.action_size,exact=True)

        self.adjacent_cells=4


        self.reset()

        self.reset_Cost()

        self.P={}

        for s in range(self.size):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.adjacent_cells)}
            self.P[s][UP] = self.transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self.transition_prob(position, [0, 1])
            self.P[s][DOWN] = self.transition_prob(position, [1, 0])
            self.P[s][LEFT] = self.transition_prob(position, [0, -1])

        #print('main_x',self.X)

    def NumberOfStates(self,size):
        nS = 1
        for i in range(size):
            nS *= ((self.default_fuel+1)*2-1)
        return nS


    def reset(self):


        self.F=np.random.random_integers(0,self.default_fuel,size=self.shape)
        self.X=np.random.binomial(1, self.initial_burn_procent, size=self.shape)
        #print('X',X)

        #C=np.random.choice(self.cost_values, size=self.shape, p=self.cost_percent)
        for index in range(self.size):
            index = np.unravel_index(index, self.shape)
            if self.F[index]==0:
                self.X[index]=0
        #print('initialX', X)

        #return F,X

    def reset_Cost(self):
        #print('costX',self.X)
        self.C = np.random.choice(self.cost_values, size=self.shape, p=self.cost_percent)
        #return C

    def action_aggregate(self):
        actions = []
        for i in range(self.size):
            i = np.unravel_index(i,self.shape)
            if ((i[0] % self.action_space_length) == 0) & ((i[1] % self.action_space_length) == 0):
                actions.append(np.ravel_multi_index(i,self.shape))
        return actions

### 

    def edge_coordinates(self, coord):

        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def transition_prob(self,current,delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self.edge_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        fuel_level=self.F[new_position[0]][new_position[1]]


        wind_strength = self.initial_wind[0]
        #acc是啥
        wind_acc = self.initial_wind[1]
        wind_dir = self.initial_wind[2]
        longest_dist = math.dist([0, 0], [self.shape[0]-1, self.shape[1]-1])
        to_norm = wind_strength * wind_acc * self.default_fuel

        '''
        for i in range(nS):
            fuel_level = F[i]
            coord_i = np.unravel_index(i, self.shape)
        
        for de in [-self.shape[1],-1,1,self.shape[1]]:
                coord_j=np.unravel_index(i+de, self.shape)
                coord_j=self.edge_coordinates(coord_j)
        '''
        rel_pos = self.relative_direction(current, new_position)
        wind_factor = self.find_wind_dir_factor(wind_dir, rel_pos)
        distance_ij = math.dist([current[0], current[1]], [new_position[0], new_position[1]])

        '''
        if distance_ij > NEIGHBOR_DIST #limit the number of cells
            distance_ij = 0
        '''

        rel_dist = distance_ij / longest_dist
        lambda_ij = wind_strength * (wind_acc * rel_dist) * wind_factor * fuel_level / to_norm
        prob = 1 - math.exp(-lambda_ij)

        return (prob,new_state)

    def find_wind_dir_factor(self, wind_dir, cell_dir):
        #print('wind',self.X)
        alignment = abs(wind_dir - cell_dir)
        if alignment == 0:
            return 1
        elif alignment == 1:
            return 0.5
        else:
            return 0  # the rest are not considered

    def relative_direction(self,cart_i,cart_j):
        #print('relative',self.X)
        row_i = cart_i[0]
        col_i = cart_i[1]
        row_j = cart_j[0]
        col_j = cart_j[1]
        if row_i == row_j:
            if col_j > col_i:
                return 2
            else:
                return 6

        elif col_i == col_j:
            if row_j > row_i:  # j is south of i
                return 4
            else:
                return 0

####



    def fuel_consumption(self):
        #print('consumeX0', self.X)
        for index in range(self.size):
            index=np.unravel_index(index,self.shape)
            if self.F[index[0]][index[1]]*self.X[index[0]][index[1]]!=0:
                self.F[index[0]][index[1]]=self.F[index[0]][index[1]]-1
        #print('consumeX',self.X)

    def fire_spread(self):
        #print('spread',self.X)
        product=1
        F_last = copy.deepcopy(self.F)
        X_last = copy.deepcopy(self.X)
        #print('xlast',X_last)

        for s in range(self.size):
            s1=np.unravel_index(s,self.shape)
            for a in range(self.adjacent_cells):
                product*=(1-self.P[s][a][0])*(X_last[s1[0]][s1[1]])
            rho1=1-product
            if X_last[s1[0]][s1[1]]==0 & F_last[s1[0]][s1[1]]>0 & bernoulli.rvs(rho1,size=1)==1:
                self.X[s1[0]][s1[1]]=1


    # grarantee that cells having no fuel are not burning
    def update_fire_states(self):
        #print('update',self.X)
        is_consumed= (self.F!=0) | (self.X==0)
        #print('updateF',self.F)
        #print('consumed',is_consumed)
        self.X=self.X-np.logical_not(is_consumed)*1
        #print('update1', self.X)

        ### 

    def action_choice(self,size):
        #print('choice_X', self.X)
        return np.random.random_integers(0, size - 1)
        # return np.random.random_integers(0,self.size-1,size=self.action_size)这是给array

    def action_aggregate_choice(self):
        
        action = np.random.random_integers(0, self.size - 1)
        #print('initialaction',action)
        action = np.unravel_index(action, self.shape)
        x = action[0]
        y = action[1]
        while (x % self.action_space_length) != 0:
            x = x - 1 # 
        while (y % self.action_space_length) != 0:
            y = y - 1
        action = np.ravel_multi_index((x, y), self.shape) #
        #print('choice', action)
        #print('index',np.where(self.actions == action))
        index = np.where(self.MDP_indices == action)[0][0] 
        return index

##

    def suppress(self,action):
        #print('sup',self.X)

        action=np.unravel_index(action, self.shape)
        if bernoulli.rvs(self.suppression_prob,size=1)==1:
            self.X[action[0]][action[1]]=0

    def action_aggregate_suppress(self,action):
        action=self.MDP_indices[action]
        #print('supac', action)
        action = np.unravel_index(action,self.shape)

        #print('actions',self.actions)
        action_list = []
        action_list.append(action)
        action_list.append((action[0],action[1]+1))
        action_list.append((action[0]+self.shape[1],action[1]))
        action_list.append((action[0]+self.shape[1],action[1]+1))
        index = np.random.random_integers(0,self.action_size-1)

        if bernoulli.rvs(self.suppression_prob,size=1)==1:
            self.X[action_list[index][0]][action_list[index][1]]=0


    def reward_once(self):
        #print('rewardx',self.X)
        #print('once', np.multiply(self.X,self.C))
        return np.sum(np.multiply(self.X,self.C))


    def terminal(self):
        return (self.X==0).all()

    def step(self,a):

        # first the fire spread
        #print('stepx',self.X)
        self.update_fire_states()
        self.fire_spread()

        # second consider the suppression
        self.suppress(a)
        # third consider the fuel consumption
        self.fuel_consumption()

        next_obs = [self.F, self.X]
        reward = self.reward_once()
        done = self.terminal()
        info = 1

        return next_obs,reward,done,info

    def action_aggregate_step(self,a):
        # first the fire spread
        self.update_fire_states()
        self.fire_spread()

        # second consider the suppression
        self.action_aggregate_suppress(a)
        # third consider the fuel consumption
        self.fuel_consumption()

        next_obs = [self.F, self.X]
        reward = self.reward_once()
        done = self.terminal()
        info = 1
        return next_obs,reward,done, info







