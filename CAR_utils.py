# coding = utf-8
import cv2 as cv
import random
from collections import deque
from ANN_utils import *
import os


class Car:
    def __init__(self, start_pos):
        self.image = np.array(cv.imread('source/car0.png'))
        self.posx = start_pos[0]
        self.posy = start_pos[1]
        self.h = self.image.shape[0]
        self.w = self.image.shape[1]
        self.h2 = self.h//2
        self.w2 = self.w//2
        self.gray = []
        self.forwarding = np.array([1., 0.])
        self.go = np.array([1., 0.])
        self.speed = 6
        self.angle = 0
        self.feel_length = 80
        self.feel_size = 9

    def record(self):
        self.posx = np.uint16(self.posx)
        self.posy = np.uint16(self.posy)
        self.h = self.image.shape[0]
        self.w = self.image.shape[1]
        self.h2 = self.h // 2
        self.w2 = self.w // 2
        self.gray = np.ones(self.image.shape[:2], dtype=np.uint8)

    def confict(self, height, width, area):
        if self.posx < self.w2:
            self.posx = self.w2
        elif self.posx > width - self.w2:
            self.posx = width - self.w2
        if self.posy < self.h2:
            self.posy = self.h2
        elif self.posy > height - self.h2:
            self.posy = height - self.h2

        offset = 6
        t = np.sum(self.gray[:-2*offset, :-2*offset] * area[self.posy-self.h2+offset:self.posy+self.h2-offset,
                   self.posx-self.w2+offset:self.posx+self.w2-offset])
        if t > 0:
            self.speed = 0
            return 1
        else:
            return 0

    def move(self, turn):
        if turn == 1:
            self.angle += 9
            if self.angle > 180:
                self.angle = self.angle - 360
        elif turn == 2:
            self.angle -= 9
            if self.angle < -180:
                self.angle = self.angle + 360

        self.go = .6 * self.go + .4 * rotate_vector(self.forwarding, self.angle)
        self.go *= 1/np.linalg.norm(self.go)
        # angle = self.angle
        angle = np.arccos(np.dot(np.array(self.go), np.array([1., 0]))) * 180/np.pi
        if self.go[1] > 0:
            angle = -angle

        # print(self.forwarding)
        # load image
        for i in range(-165, 181, 15):
            if i+7 >= self.angle > i-8:
                img = np.array(cv.imread('source/car' + str(i) + '.png'))
        if self.angle < -165:
            img = np.array(cv.imread('source/car180.png'))

        # cut image
        cut_up = 0
        cut_down = 0
        cut_left = 0
        cut_right = 0
        for i in range(0, img.shape[0]):
            if np.sum(img[i, :]) == 0:
                cut_left += 1
            else:
                break
        for i in range(img.shape[0]-1, -1, -1):
            if np.sum(img[i, :]) == 0:
                cut_right += 1
            else:
                if (cut_left + cut_right) % 2 != 0:
                    if cut_left > 0:
                        cut_left -= 1
                    elif cut_right > 0:
                        cut_right -= 1
                break
        for i in range(0, img.shape[1]):
            if np.sum(img[:, i]) == 0:
                cut_up += 1
            else:
                break
        for i in range(img.shape[1]-1, -1, -1):
            if np.sum(img[:, i]) == 0:
                cut_down += 1
            else:
                if (cut_up + cut_down) % 2 != 0:
                    if cut_up > 0:
                        cut_up -= 1
                    elif cut_down > 0:
                        cut_down -= 1
                break

        self.image = []
        self.image = img[cut_left:-cut_right, cut_up:-cut_down]

        # print(cut_left, cut_right, cut_up, cut_down)

        self.posx = round(self.posx + self.go[0] * self.speed)
        self.posy = round(self.posy + self.go[1] * self.speed)
        # print(self.posx, self.posy)

    def feel(self, area):
        prex = self.posx + self.go[0] * 8
        prey = self.posy + self.go[1] * 8
        t0 = self.go
        t1 = rotate_vector(self.go, angle=20)
        t2 = rotate_vector(self.go, angle=40)
        t3 = rotate_vector(self.go, angle=60)
        t4 = rotate_vector(self.go, angle=80)
        t5 = rotate_vector(self.go, angle=-20)
        t6 = rotate_vector(self.go, angle=-40)
        t7 = rotate_vector(self.go, angle=-60)
        t8 = rotate_vector(self.go, angle=-80)
        feel = np.zeros((self.feel_size, 3), dtype=np.uint16)
        feel_length = self.feel_length
        for i in range(feel_length):
            y = np.uint16(prey + i * t0[1])
            x = np.uint16(prex + i * t0[0])
            feel[0] = i, y, x
            if area[y, x]:
                break
        for i in range(feel_length):
            y = np.uint16(prey + i * t1[1])
            x = np.uint16(prex + i * t1[0])
            feel[1] = i, y, x
            if area[y, x]:
                break
        for i in range(feel_length):
            y = np.uint16(prey + i * t2[1])
            x = np.uint16(prex + i * t2[0])
            feel[2] = i, y, x
            if area[y, x]:
                break
        for i in range(feel_length):
            y = np.uint16(prey + i * t3[1])
            x = np.uint16(prex + i * t3[0])
            feel[3] = i, y, x
            if area[y, x]:
                break
        for i in range(feel_length):
            y = np.uint16(prey + i * t4[1])
            x = np.uint16(prex + i * t4[0])
            feel[4] = i, y, x
            if area[y, x]:
                break
        for i in range(feel_length):
            y = np.uint16(prey + i * t5[1])
            x = np.uint16(prex + i * t5[0])
            feel[5] = i, y, x
            if area[y, x]:
                break
        for i in range(feel_length):
            y = np.uint16(prey + i * t6[1])
            x = np.uint16(prex + i * t6[0])
            feel[6] = i, y, x
            if area[y, x]:
                break
        for i in range(feel_length):
            y = np.uint16(prey + i * t7[1])
            x = np.uint16(prex + i * t7[0])
            feel[7] = i, y, x
            if area[y, x]:
                break
        for i in range(feel_length):
            y = np.uint16(prey + i * t8[1])
            x = np.uint16(prex + i * t8[0])
            feel[8] = i, y, x
            if area[y, x]:
                break
        return feel


class EnvCar:
    def __init__(self):
        self.fitness = 0
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.tick = 0
        self.canvas = []
        self.window_name = 'ICAR'
        self.width, self.height = (1200, 1200)
        self.ori_pos = self.width // 2 - 150, self.height // 2 - 20
        self.car = Car((self.ori_pos[0], self.ori_pos[1]))
        self.area = np.zeros((self.width, self.height), dtype=np.uint8)
        self.end_flag = 0
        self.done = 0
        self.info = ' '
        self.side = []
        self.side_change = 0
        self.cnt = 0
        self.side_t = 0
        self.side_dir = 0
        self.chang_flag = 0
        self.load_flag = 0
        self.state_size = self.car.feel_size
        self.action_size = 3

        # init window
        cv.namedWindow(self.window_name, 0)
        cv.resizeWindow(self.window_name, (600, 600))
        cv.moveWindow(self.window_name, 400, 1)

    def reset(self):
        self.car = Car((self.ori_pos[0], self.ori_pos[1]))
        self.fitness = 0
        self.tick = 0
        self.done = 0
        return self.car.feel(self.area)

    def add_map(self):
        if not self.load_flag:
            self.load_flag = 1
            t = np.array(cv.imread('source/map.png'))
            t = t[:, :, 0] > 0
            t = t * 1
            board_row = 200
            board_col = 200
            t[:, :board_row] = 1
            t[:board_col, :] = 1
            t[:, -board_row:] = 1
            t[-board_col:, :] = 1
            self.area = t
            self.height, self.width = self.area.shape

    def step(self, action=0, delay=50, display=None, human_control=True, mode=None, generation=0, verbose=True):
        self.tick += 1
        turn = action
        getkey = cv.waitKey(delay)
        if getkey == 27:
            self.end_flag = 1
        if human_control:
            if getkey == ord('a'):
                turn = 1
            elif getkey == ord('d'):
                turn = 2

        # move car
        self.car.move(turn)
        self.car.record()
        if mode == 'sliding':
            board_row = 200
            board_col = 250
            self.area[:, :board_row] = 1
            self.area[:board_col, :] = 1
            self.area[:, -board_row:] = 1
            self.area[-board_col:, :] = 1
            self.car.posx -= 1
        elif mode == 'map':
            self.add_map()
        else:
            raise Exception('MODE ERROR!')

        # limit area
        self.done = self.car.confict(self.height, self.width, self.area)
        self.car.record()
        # feel the road
        feel_arr = self.car.feel(self.area)
        state = feel_arr[:, 0]/self.car.feel_length
        # reward_part1 = np.sqrt(np.sum(np.square(self.car.posx-self.width//2)+np.square(self.car.posy-self.height//2)))
        # reward = 0.1 + reward_part1 / (5000+reward_part1) + np.mean(np.sort(state)[:3])
        # reward = self.tick * 1.2 / (self.tick + 300.)
        reward = np.sort(state)[0]   # Successful ! For saving
        # reward = .2 * np.mean(np.sort(state)[1]) + 0.8 * np.sort(state)[0]
        if reward < 0.25:
            reward = 0

        if verbose and not self.end_flag:
            # write string
            string = str(self.car.posx) + ' ' + str(self.car.posy) + ' ' + str(self.car.angle) + '   ' +\
                     str(np.uint(state*10))
            string2 = 'time = ' + str(self.tick) + '  reward = ' + str(np.uint(reward*100))
            string3 = 'generation = ' + str(generation)

            t_area = self.area[self.car.posy-200:self.car.posy+200,
                               self.car.posx-200:self.car.posx+200]
            canvas = np.zeros((t_area.shape[0], t_area.shape[1], 3), dtype=np.uint8)
            canvas[:, :, 0] = t_area * 100
            canvas[:, :, 1] = t_area * 100
            canvas[:, :, 2] = t_area * 200
            # draw car
            canvas[t_area.shape[0]//2 - self.car.h2:t_area.shape[0]//2 + self.car.h2,
                   t_area.shape[1] // 2 - self.car.w2:t_area.shape[1]//2 + self.car.w2] += self.car.image
            # draw sensor
            for i in range(feel_arr.shape[0]):
                ty = np.uint16(feel_arr[i][1] - self.car.posy + t_area.shape[0]//2)
                tx = np.uint16(feel_arr[i][2] - self.car.posx + t_area.shape[1]//2)
                canvas[ty-2:ty+2, tx-2:tx+2] = (0, 255, 255)

            canvas = cv.putText(canvas, string, (0, 15), self.font, 0.6, (69, 137, 148), 2)
            canvas = cv.putText(canvas, string2, (0, 30), self.font, 0.6, (148, 137, 69), 2)
            canvas = cv.putText(canvas, string3, (0, 45), self.font, 0.6, (137, 69, 148), 2)

            cv.imshow(self.window_name, canvas)

        if self.end_flag:
            cv.destroyAllWindows()
        return state, reward, self.done, self.info


class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95                   # 计算未来奖励时的折算率
        self.epsilon = 1.0                  # agent 最初探索环境时选择 action 的探索率
        self.epsilon_min = 0.01             # agent 控制随机探索的阈值
        self.epsilon_decay = 0.998            # 随着 agent 玩游戏越来越好，降低探索率
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = MlpNn(self.state_size)
        model.add_layer(8, activation='tanh')
        model.add_layer(7, activation='tanh')
        model.add_layer(6, activation='tanh')
        model.add_layer(5, activation='tanh')
        model.add_layer(self.action_size, activation='linear')
        model.compile(optimizer='adam')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epoch=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        pass


def rotate_vector(v, angle):  # with the clock
    angle = angle * np.pi / 180  # convert
    t0 = v[0] * np.cos(angle) + v[1] * np.sin(angle)
    t1 = - v[0] * np.sin(angle) + v[1] * np.cos(angle)
    return np.array([t0, t1])

