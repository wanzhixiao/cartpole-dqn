import tensorflow as tf
import numpy as np

class DeepQNetwork:
    def __init__(self,n_actions,n_features,learning_rate,
                 gamma=0.9,e_greedy=0.9,batch_size=64,memory_size=1000,replace_target_iter=300,
                 output_graph=False):
        # 初始化参数
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.replace_target_iter = replace_target_iter

        #动作选择时的贪婪系数
        self.epsilon = e_greedy
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        #训练次数计数
        self.learn_step_counter = 0
        #replay buffer [s, a, r, s_]
        self.memory = np.zeros((self.memory_size,n_features*2+2))

        #构建评估网络和目标网络
        self._build_net()

        #获两个网络的取参数并替换
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        '''
        构建评估网络和目标网络
        :return:
        '''
        self.s = tf.placeholder(tf.float32,[None,self.n_features],name='s')         #输入状态
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  #用于计算损失

        #输出化权重和偏置
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        #评估网络，构建两层神经网络
        with tf.variable_scope('eval_net'):
            #参数变量集合名，第一层神经元数
            c_names,n_l1 =  ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES],10
            #第一层，collections用来收集参数为目标网络赋值
            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w1',[self.n_features,n_l1],initializer=w_initializer,collections=c_names)
                b1 = tf.get_variable('b1',[1,n_l1],initializer=b_initializer,collections=c_names)
                h1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)
            # 第二层，collections用来收集参数为目标网络赋值
            with tf.variable_scope('layer2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(h1,w2) + b2
            #计算损失
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval,self.q_target))
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        #目标网络，与评估网络相同
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            c_names, n_l1 = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10
            # 第一层，collections用来收集参数为目标网络赋值
            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                h1 = tf.nn.relu(tf.matmul(self.s_,w1) + b1)
            # 第二层，collections用来收集参数为目标网络赋值
            with tf.variable_scope('layer2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(h1, w2) + b2

    def store_transition(self, s, a, r, s_):
    # 存储训练过程中的表现
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))  #水平方向堆叠 当前状态、动作、奖励、下一动作
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # 如何选择一个action
        #增加一个batch维度便于喂入feed_dict
        #observation = np.reshape(observation,[1,self.n_features])
        observation = observation[np.newaxis, :]
        #代理在选择动作时使用epsilon贪婪策略
        if(np.random.uniform() <= self.epsilon):
            action = np.random.randint(0, self.n_actions)
        else:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        return action

    def learn(self):
        # 参数如何更新
        #检查是否更新目标网络的参数
        if(self.learn_step_counter % self.replace_target_iter == 0):
            self.sess.run(self.replace_target_op)
            print('target_params replaced at learn step:{}'.format(self.learn_step_counter+1))
        #从replay buffer中获取批量随机样本
        if(self.memory_counter>self.memory_size):
            #从memeory_size个数中随机选取batch_size个索引
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            #从memory_counter个数中随机选取min(memory_counter,batch_size)个索引
            sample_index = np.random.choice(self.memory_counter,size=self.batch_size)
        sample = self.memory[sample_index,:]
        #训练agent
        q_next,q_eval = self.sess.run(
            [self.q_next,self.q_eval],
            feed_dict={self.s_:sample[:,-self.n_features:],#固定的参数
                       self.s:sample[:,:self.n_features]} #最新的参数
        )

        #更改目标网络的参数和q值
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = sample[:,self.n_features].astype(int)
        reward = sample[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        #训练评估网络
        _,self.cost  = self.sess.run([self._train_op,self.loss],
                                feed_dict={self.s:sample[:,:self.n_features],self.q_target:q_target})
        self.cost_his.append(self.cost)
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

import gym
def run_cartpole():
    step = 0
    for episode in range(300):
        #重置环境
        observation = env.reset()
        print('eposide:',episode)
        while(True):
            #渲染环境
            env.render()
            #获取动作
            action = dqn.choose_action(observation)
            #评估动作，获取下一个环境状态，奖励，游戏结束标识
            observation_,reward,done,_ = env.step(action)
            #存到replay buffer，便于回放
            dqn.store_transition(observation,action,reward,observation_)

            #超过200步后开始学习（经验回放），一开始replay buffer中没有经验
            if (step > 200) and (step % 5 == 0):
                    dqn.learn()
            if done:
                break
            #贪婪系数衰减
            dqn.epsilon = max(dqn.epsilon_min, dqn.epsilon_decay * dqn.epsilon)  # decrease epsilon
            observation = observation_
            step += 1


if __name__ == '__main__':
    env_string = 'CartPole-v0'
    env = gym.make(env_string)
    n_actions,n_features = env.action_space.n,env.observation_space.shape[0]
    dqn = DeepQNetwork(n_actions,n_features,
                       learning_rate=0.01,
                       gamma=0.9,  # reward decay rate
                       batch_size=64,
                       memory_size=1000,
                       e_greedy=0.9,
                       replace_target_iter=200,
                       output_graph=True)
    run_cartpole()
    dqn.plot_cost()
