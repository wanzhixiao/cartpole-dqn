from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import gym

class Network(tf.keras.Model):
    def __init__(self,
                 n_actions,
                 hidden_size):

        #initialize
        super(Network, self).__init__()

        #build model
        self.net = tf.keras.Sequential([
            layers.Dense(hidden_size),
            layers.Dropout(0.2),
            tf.keras.layers.ReLU(),
            layers.Dense(n_actions)
        ])

    def call(self, observation, training=None):
        '''
        :param observation: current state of enviroment
        :return: action
        '''
        observation = observation[tf.newaxis, :]
        print('observation: ',observation.shape)
        action_value = self.net(observation)
        action = tf.argmax(action_value)
        return action

class ReplayBuffer(object):

    def __init__(self,
                 memory_size,
                 n_features,
                 epsilon,
                 n_actions,
                 batch_size,
                 epsilon_min,
                 epsilon_decay):

        self.memory_counter = 0
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = np.zeros((self.memory_size,n_features*2+2))

    def store_transition(self, s, a, r, s_):
        '''
        :param s: 当前的状态
        :param a: 当前的动作
        :param r: 将离值
        :param s_: 下一个动作
        :return:
        '''
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))  #水平方向堆叠 当前状态、动作、奖励、下一动作
        index = self.memory_counter % self.memory_size #最近的经验留在memory中
        self.memory[index,:] = transition
        self.memory_counter  += 1

    def choose_action(self,model,observation):
        '''
        :param model: eval_net
        :param observation: 当前的状态
        :return:
        '''

        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0,self.n_actions)
        else:
            action_value = model(observation)
            action = np.argmax(action_value)
        return action

    def sample(self):

        #从replay buffer中获取批量随机样本
        if self.memory_counter>self.memory_size:
            #从memeory_size个数中随机选取batch_size个索引
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,self.batch_size)
        sample = self.memory[sample_index,:]

        return sample

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)  # decrease epsilon

class DeepQNetwork(tf.keras.Model):

    def __init__(self,
                 n_actions,
                 n_features,
                 hidden_size,
                 memory_size,
                 epsilon,
                 batch_size,
                 gamma,
                 replace_target_iter,
                 epsilon_min=0.01,
                 epsilon_decay=0.995):

        super(DeepQNetwork, self).__init__()

        self.learn_step_counter = 0
        self.replace_target_iter = replace_target_iter
        self.eval_net = Network(n_actions, hidden_size)
        self.target_net = Network(n_actions, hidden_size)
        self.replay_buffer = ReplayBuffer(memory_size, n_features, epsilon, n_actions, batch_size, epsilon_min=epsilon_min,
                                     epsilon_decay=epsilon_decay)

        self.n_features = n_features
        self.batch_size = batch_size
        self.gamma = gamma

    def call(self,observation_):
        # 1.检查是否更新目标网络的参数
        print(1)
        if(self.learn_step_counter % self.replace_target_iter == 0):

            #先调用以下target_model,初始化其权重
            _ = self.target_net(observation_)
            print(2)
            # pretrain_model_weights = tf.saved_model.load('./checkpoints/tf_model')
            # params_dict = {}
            # for v in pretrain_model_weights.trainable_variables:
            #     params_dict[v.name] = v.read_value()

            # 加载 除最后一层外 的预训练参数到模型，以便下一步finetune
            for idx, layer in enumerate(self.target_net.trainable_variables):
                layer.assign(self.eval_net.trainable_variables[idx])

            print('target_params replaced at learn step:{}'.format(self.learn_step_counter+1))
        print(3)
        # 2.从replay buffer中获取批量随机样本
        sample = self.replay_buffer.sample()

        print('sample:',sample.shape)
        print('eval_net',sample[:, :self.n_features].shape)
        print('target_net',sample[:, -self.n_features:].shape)

        # 3.训练agent
        q_eval = self.eval_net(sample[:, :self.n_features])  # 需要计算损失， 最新的参数
        print(3.5)
        q_next = self.target_net(sample[:, -self.n_features:])  # 为什么会有两倍的n_features?， 固定的参数
        print(4)
        # 4.计算目标网络的预测值
        q_target = tf.tile(q_eval,[1,1])
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = sample[:, self.n_features].astype(int)  # (s, [a, r], s_)
        reward = sample[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)  # eval_net的label
        print(5)
        #5.eval_net 预测
        q_eval_new = self.eval_net(sample[:, :self.n_features])
        self.learn_step_counter += 1
        print(6)
        return q_eval_new, q_target


    def choose_action(self,observation):
        return self.replay_buffer.choose_action(self.eval_net,observation)

    def store_transition(self,observation,action,reward,observation_):
        self.replay_buffer.store_transition(observation,action,reward,observation_)

    def update_epsilon(self):
        self.replay_buffer.update_epsilon()

def train_step(
       model,
       env,
       loss_func,
       optimizer
    ):

    global  step
    observation = env.reset()
    with tf.GradientTape() as tape:

        while (True):
            # 渲染环境
            env.render()
            # 获取动作
            action = model.choose_action(observation)
            # 评估动作，获取下一个环境状态，奖励，游戏结束标识
            observation_, reward, done, _ = env.step(action)
            # 存到replay buffer，便于回放
            model.store_transition(observation, action, reward, observation_)

            # 超过200步后开始学习（经验回放），一开始replay buffer中没有经验
            if (step > 200) and (step % 5 == 0):
                q_eval, q_target = model(observation_)
                #计算评估网络的loss
                loss = loss_func(q_eval, q_target)
                grads = tape.gradient(loss,model.trainable_variables)
                optimizer.apply_gradient(zip(grads,model.trainable_variables))

            print(step)
            if done:
                break

            # 贪婪系数衰减
            model.update_epsilon()
            observation = observation_
            step += 1
step = 0
def run_cartpole(env,
                 hidden_size,
                 epsilon,
                 memory_size,
                 batch_size,
                 gamma,
                 replace_target_iter):

    n_actions, n_features = env.action_space.n, env.observation_space.shape[0]

    model = DeepQNetwork(
                n_actions,
                 n_features,
                 hidden_size,
                 memory_size,
                 epsilon,
                 batch_size,
                 gamma,
                 replace_target_iter)

    loss_func = tf.losses.MeanSquaredError() #loss_func是用MSE?
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    global  step
    for episode in range(300):
        # 重置环境
        print('eposide:', episode)

        train_step(model,env, loss_func,optimizer)


if __name__ == '__main__':

    env = gym.make("CartPole-v0")
    # Set seed for experiment reproducibility
    seed = 42
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Small epsilon value for stabilizing division operations
    eps = np.finfo(np.float32).eps.item()

    hidden_size = 32
    epsilon = 0.9
    memory_size = 1000
    batch_size = 64
    gamma = 0.9
    replace_target_iter = 200

    run_cartpole(
        env,
        hidden_size,
        epsilon,
        memory_size,
        batch_size,
        gamma,
        replace_target_iter
    )