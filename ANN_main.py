# FileName:ANN_main.py
# coding = utf-8
# Created by Hzq
from ANN_utils import *
from CAR_utils import *
from time import sleep
import datetime


def s_1():
    train_x = np.linspace(0, 2*np.pi, 6)
    train_y = np.zeros((train_x.shape[0], 2))
    train_y[:, 0] = np.sin(train_x)
    train_y[:, 1] = np.cos(train_x)

    model = MlpNn(input_shape=1)
    model.add_layer(32, 'tanh')
    model.add_layer(32, 'tanh')
    model.add_layer(16, 'tanh')
    model.add_layer(16, 'tanh')
    model.add_layer(2, 'linear')
    model.compile(optimizer='adam')
    for i in range(1000):
        model.fit(train_x, train_y, epoch=1, learn_rate=0.001, verbose=1)
    a = np.mat([np.pi/3])
    print(model.predict(a), a.shape)


def s_4():
    env = EnvCar()

    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)

    batch_size = 16
    EPISODES = 1000
    max_time = 2500

    for e in range(EPISODES):
        state = env.reset()[:, 0]
        state = np.reshape(state, [1, state_size])

        for time in range(max_time):
            action = agent.act(state)

            next_state, reward, done, _ = env.step(delay=1, display=1, action=action, mode='map', generation=e, verbose=1)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            if done or time == max_time-1:
                print("episode: {}/{}, score: {}, e: {:.2}, len: {}"
                      .format(e, EPISODES, time, agent.epsilon, len(agent.memory)))
                if e % 5 == 0:
                    agent.save()
                break
            # start_time = datetime.datetime.now()
            if len(agent.memory) > batch_size and time % 2 == 0:
                agent.replay(batch_size)
            # end_time = datetime.datetime.now()
            # print(len(agent.memory), end_time - start_time)


s_4()

