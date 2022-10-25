import random as r
import numpy as np
from network_dict import network_dictionary
import itertools
import copy



class ProcessingNetwork:
    # Multiclass Queuing Network class
    def __init__(self, A, D, alpha, mu,  name):
        self.alpha = np.asarray(alpha)  # arrival rates
        self.mu = np.asarray(mu)  # service rates
        self.uniform_rate = np.sum(alpha)+np.sum(mu)  # uniform rate for uniformization
        self.p_arriving = np.divide(self.alpha, self.uniform_rate)
        self.p_compl = np.divide(self.mu, self.uniform_rate)
        self.cumsum_rates = np.unique(np.cumsum(np.asarray([self.p_arriving, self.p_compl])))
        # print("np.asarray([self.p_arriving, self.p_compl])", np.asarray([self.p_arriving, self.p_compl]))
        # print("np.cumsum(np.asarray([self.p_arriving, self.p_compl])", np.cumsum(np.asarray([self.p_arriving, self.p_compl])))
        # print("cumsum_rates", self.cumsum_rates)

        self.A = np.asarray(A)  # each row represents activity: -1 means job is departing, +1 means job is arriving
        self.routing_matrix = 1 * (self.A > 0)
        self.D = np.asarray(D)  # ith row represents buffers that associated to the ith stations

        self.M = np.diag(1./self.mu)
        self.alpha_total = np.dot(np.linalg.inv(np.eye(len(A)) - self.routing_matrix.T) , self.alpha[np.newaxis].T)
        self.rho = np.dot(self.D,np.dot(self.M,self.alpha_total)) # load of each station

        self.action_size = np.prod(np.sum(D, axis=1))  # total number of possible actions
        self.action_size_per_buffer = [sum(D[i]) for i in range(len(D))]  # number of possible actions for each station
        self.stations_num = np.shape(D)[0]  # number of stations
        self.buffers_num = len(mu)  # number of buffers

        self.activities_num = len(self.cumsum_rates)  # number of activities
        print('activities num: ', self.activities_num)
        self.network_name = name

        # if self.network_name[:11] == 'criss_cross' or self.network_name == 'reentrant': # choose "optimal" policy for comparison
        #     self.comparison_policy = self.policy_list(name)
        # else:
        #     self.comparison_policy = self.policy_list('LBFS')



        self.dict_absolute_to_binary_action, self.dict_absolute_to_per_server_action = self.absolute_to_binary()
        self.actions = list(self.dict_absolute_to_binary_action.values())  # list of all actions
        self.list, self.next_state_list = self.next_state_list()


    @classmethod
    def from_name(cls, network_name: str):
        # another constructor for the standard queuing networks
        # based on a queuing network name, find the queuing network info in the 'network_dictionary.py'
        return cls(A=network_dictionary[network_name]['A'],
                   D=network_dictionary[network_name]['D'],
                   alpha=network_dictionary[network_name]['alpha'],
                   mu=network_dictionary[network_name]['mu'],
                   name=network_dictionary[network_name]['name'])



    def absolute_to_binary(self):
        """
        :return:
        dict_absolute_to_binary_action: Python dictionary where keys are 'act_ind' action representation,
                                        values are 'action_full' representation
        dict_absolute_to_per_server_action: Python dictionary where keys are 'act_ind' action representation,
                                            values are 'action_for_server' representation
        act_ind - all possible actions are numerated GLOBALLY as 0, 1, 2, ...
        action_full - buffers that have priority are equal to 1, otherwise 0
        action_for_server - all possible actions FOR EACH STATIONS are numerated as 0, 1, 2, ...
        For the simple reentrant line.
        If priority is given to the first class:
            act_ind = [0]
            action_full = [1, 1, 0]
            action_for_server = [0, 0]
        If priority is given to the third class:
            act_ind = [1]
            action_full = [0, 1, 1]
            action_for_server = [1, 0]
        """

        dict_absolute_to_binary_action = {}
        dict_absolute_to_per_server_action = {}

        actions_buffers = [[a] for a in range(self.action_size_per_buffer[0])]

        for ar_i in range(1, self.stations_num):
            a =[]
            for c in actions_buffers:
                for b in range(self.action_size_per_buffer[ar_i]):
                    a.append(c+[b])
            actions_buffers = a

        assert len(actions_buffers) == self.action_size
        for i, k in enumerate(actions_buffers):
            dict_absolute_to_binary_action[i] = self.action_to_binary(k)
            dict_absolute_to_per_server_action[i] = k



        return dict_absolute_to_binary_action, dict_absolute_to_per_server_action




    def action_to_binary(self, act_ind):
        """
        change action representation
        :param act_ind: all possible actions are numerated GLOBALLY as 0, 1, 2, ...
        :return: buffers that have priority are equal to 1, otherwise 0
        For the simple reentrant line.
        If priority is given to the first class:
        act_ind = [0]
        action_full = [1, 1, 0]
        If priority is given to the third class:
        act_ind = [1]
        action_full = [0, 1, 1]
        """

        action_full = np.zeros(self.buffers_num)
        for i in range(len(self.D)):
            res_act = act_ind[i]
            k = -1

            for act in range(len(self.D[0])):
                if self.D[i][act] == 1:
                    k += 1
                    if res_act == k:
                        break
            action_full[act] = 1

        return action_full

    def next_state(self, state, action):
        """
        generate the next state
        :param state: current state
        :param action: action
        :return: next state
        """
        w = r.random()
        wi = 0
        while w > self.cumsum_rates[wi]:
            wi+=1 #  activity that will be processed

        q = np.asarray(state) > 0 # list of non-empty and empty buffers
        # int(wi - np.sum(self.alpha > 0)) is ser_i, index of the service event; or which buffer is serviced
        # action[ser_i] being 1: the action will prioritize this buffer ser_i
        # action[ser_i] being 0: the action will not prioritize this buffer ser_i
        actions_coin = (action[int(wi - np.sum(self.alpha > 0))] == 1)  # indicates if the activity is legitimate

        state_next = state + self.list[(tuple(q), actions_coin, wi)]
        return state_next

    def next_state_nClassesPerformance(self, state, action):
        """
        generate the next state
        :param state: current state
        :param action: action
        :return: next state
        """
        w = r.random()
        wi = 0
        is_arrival = False
        while w > self.cumsum_rates[wi]:
            wi+=1 #  activity that will be processed

        q = np.asarray(state) > 0 # list of non-empty and empty buffers
        # int(wi - np.sum(self.alpha > 0)) is ser_i, index of the service event; or which buffer is serviced
        # action[ser_i] being 1: the action will prioritize this buffer ser_i
        # action[ser_i] being 0: the action will not prioritize this buffer ser_i
        actions_coin = (action[int(wi - np.sum(self.alpha > 0))] == 1)  # indicates if the activity is legitimate
        if wi < np.sum(self.alpha > 0) :
            is_arrival = True
        state_next = state + self.list[(tuple(q), actions_coin, wi)]
        return state_next, is_arrival

    def next_state_list(self):
        """
        :return:
        list: Python dictionary s.t. keys are ( state > 0, action, activity), values are jobs transitions
        list_next_states: set of all possible jobs transitions
        """
        list = {}
        s_D = np.shape(self.D)

        '''
        #### compute the set of all posible actions ########################
        set_act = []
        actions = [s for s in itertools.product([0, 1], repeat=self.buffers_num)]
        for a in actions:
            i=0
            while i != s_D[0]:
                d = np.asarray(self.D[i])
                if np.dot(d, np.asarray(a))!= 1:
                    break
                i += 1
            if i == s_D[0]:
                set_act.append(a)
        self.actions = set_act # set of all possible actions
        #######################
        '''

        adjoint_buffers = {} # Python dictionary: key is a buffer, value is a list of buffers associated to the same station
        for i in range(0, s_D[0]):
            for j in range(0, s_D[1]):
                if self.D[i][j] ==1:
                    d = copy.copy(self.D[i])  # TODO: np.copy?
                    d[j] = 0
                    adjoint_buffers[j] = copy.copy(d)
        self.adjoint_buffers = adjoint_buffers

        for a in [False, True]:#self.actions:  # indicator that activity 'w' is legitimate
            for s in itertools.product([0, 1], repeat=self.buffers_num):  # combination of non-empty, empty buffers
                for w in range(0, int(np.sum(self.alpha>0)+np.sum(self.mu>0))):  # activity

                        ar = np.asarray(s, 'int8') # current buffer state (empty or not)
                        if w < np.sum(self.alpha>0):  # arrival activity
                            el = np.nonzero(self.alpha)[0][w] # determines which arrival it corresponds to
                            arriving = np.zeros(self.buffers_num, 'int8')
                            arriving[el] = 1
                            list[(tuple(ar), a, w)] = arriving
                        elif ar[w - np.sum(self.alpha>0)]>0 and \
                                (a or np.sum(np.dot(ar, adjoint_buffers[w - np.sum(self.alpha>0)]))==0):# service activity is possible
                            # ser_id = w - np.sum(self.alpha>0) indicates which service activity it is, 
                            # or essentially which buffer it is
                            # Cond 1. ar[ser_id]>0  indicates that this buffer is not empty,
                            # and a service being completed in this buffer indeed makes sense
                            # Cond 2. np.sum(np.dot(ar, adjoint_buffers[ser_id])): determine if all neighbors 
                            # of w have an empty buffer
                            # If a == True, meaning that we indeed have been prioritizing this job, we 
                            # obvisouly can accept this completion activity;
                            # If a == False, meaning we were prioritizing somehting else, we only accept
                            # this completion activity if all other adjoint buffers are empty, in which
                            # case we assume we have been working on this only available job.
                            list[(tuple(ar), a, w)] = self.A[w - np.sum(self.alpha>0)]

                        else:  # service activity is not possible. Fake transition
                            list[(tuple(ar), a, w)] = np.zeros(self.buffers_num, 'int8')


        list_next_states = np.asarray([ list[(tuple(np.ones(self.buffers_num)), 1, w)] for w in range(0, int(np.sum(self.alpha>0)+np.sum(self.mu>0)))])
        return list, list_next_states


    # TODO: simplify the method
    def policy_list(self, policy):
        """
        :param policy: name of the queuing network
        :return: optimal policy for the queuing network for comparison
        """


        if policy == 'criss_cross':
            p = np.load('policy, criss-cross, n=140, average, 15.1915.npy', allow_pickle=True)

            list = p.item()
        elif policy == 'criss_crossIH':
            p = np.load('policy, criss-crossIH, skipping=1, n = 150, beta = 1, ac = 9.964.npy', allow_pickle=True)

            list = p.item()
        elif policy == 'criss_crossBM':
            p = np.load('policy, criss-crossBM, skipping=1, n = 150, beta = 1, ac = 2.82.npy', allow_pickle=True)

            list = p.item()
        elif policy == 'criss_crossIM':
            p = np.load('policy, criss-crossIM, skipping=1, n = 150, beta = 1, ac = 2.079.npy', allow_pickle=True)

            list = p.item()
        elif policy == 'criss_crossBL':
            p = np.load('policy, criss-crossBL, skipping=1, n = 150, beta = 1, ac = 0.841.npy', allow_pickle=True)

            list = p.item()
        elif policy == 'criss_crossBL':
            p = np.load('criss-crossBL, skipping=1, n = 150, beta = 1, ac = 0.842.npy', allow_pickle=True)

            list = p.item()
        elif policy == 'criss_crossIL':
            p = np.load('policy, criss-crossIL, skipping=1, n = 150, beta = 1, ac = 0.67.npy', allow_pickle=True)

            list = p.item()

        elif policy == 'reentrant':
            p = np.load('policy, RV_reentrance, n=140, disc = 0.9998, 25.58434.npy', allow_pickle=True)

            list = p.item()


        else:
            list = {}

            for state in itertools.product([0, 1], repeat=self.buffers_num):
                a = np.zeros(np.size(state), 'int8')
                state = np.asarray(state, 'int8')


                for k in range(0, self.stations_num):
                    d = self.D[k]

                    d_nz = np.transpose(np.nonzero(d))



                    if policy == 'FBFS':
                        j = 0
                        while state[d_nz[j]] == 0:
                            j += 1
                            if j == len(d_nz):
                                j -= 1
                                break

                        a[d_nz[j]] = 1

                    elif policy == 'LBFS':
                        j = len(d_nz) - 1
                        while state[d_nz[j]] == 0:
                            j -= 1
                            if j == -1:
                                j += 1
                                break
                        a[d_nz[j]] = 1
                    elif policy == 'cmu':
                        ind = np.argsort(self.p_compl * d)
                        j = len(d) - 1
                        while state[ind[j]] == 0:
                            j -= 1
                            if j == -1 or d[ind[j]] == 0:
                                j += 1
                                break

                        a[ind[j]] = 1

                list[tuple(state)] = a

        return list


    def next_state_prob(self, states_array):
        """
        Compute probability of each transition for each action for the criss-cross network
        :param states_array: array of states
        :return: probability of each transition
        """
        states = 1*(states_array > 0) # whether buffer is non-empty
        arrivals_num = self.activities_num - self.buffers_num # number of arrival flows
        prod_for_actions_list = []
        for action_ind, action in enumerate(self.actions):
            #'constr_trans[i, j]' equals to 1 if ith buffer has to be non-empty (-1 -- empty)
            # to make jth activity possible; if 0, then it can be either
            constr_trans = np.zeros((self.buffers_num, self.activities_num))
            for b_i in range(self.buffers_num):
                constr_trans[b_i, b_i + arrivals_num] = 1
                if action[b_i] == 0:
                    # if we are not priorityzing this buffer, then this b_i's neighbors
                    # all have to be empty so that this activity is valid
                    for adj_buf_i in np.nonzero(self.adjoint_buffers[b_i]):
                        constr_trans[adj_buf_i,  b_i + arrivals_num] = -1

            # activities legitimacy for the actual data
            possible_activities = 1*((states @ constr_trans) > 0) # (N, activities_num), 1 meaning possible
            for arv_i in range(arrivals_num):
                possible_activities[:, arv_i] = 1 # arrival activities always possible
            # probability for each activity, except fake one
            prob = possible_activities @ np.diag(np.hstack([self.p_arriving[self.p_arriving > 0], self.p_compl[self.p_compl > 0]]))
            prob_fake_transition = 1 - np.sum(prob, axis=1)  # probability of a fake transition for each time_step
            prod_for_actions = np.hstack([prob, prob_fake_transition[:, np.newaxis]])
            prod_for_actions_list.append(prod_for_actions)

        return prod_for_actions_list


    def random_proportional_policy_distr(self, state):
        """
        Return probability distribution of actions for each station based on Random proportional policy
        :param state: system state
        :return: distribution of action according to random proportional policy
        """
        distr = []
        for server in range(self.stations_num):
            distr_one_server = np.zeros((1, np.sum(self.D[server])))
            z_sum = np.sum(state[self.D[server]>0]) # sum of jobs that wait to be processed in 'server'
            if z_sum > 0:
                all_states = state / z_sum
                distr_one_server = np.reshape(all_states[self.D[server]>0], (1, np.sum(self.D[server])))
            else:
                distr_one_server[0]  = 1./sum(self.D[server])
            distr.append(distr_one_server)
        return distr

# network = ProcessingNetwork.from_name('criss_crossIM')