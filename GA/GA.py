import numpy as np
import pygame
import random
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from pygame.locals import *

def relu_l(array):
    return np.minimum(np.maximum(array, 0), 1)

class NetWork:
    def __init__(self):
        self.input  = np.array([0 for _ in range(8)]) # 输入层  8个神经元
        self.l2     = np.array([0 for _ in range(4)]) # 隐含层1 4个神经元
        self.l3     = np.array([0 for _ in range(4)]) # 隐含层2 4个神经元
        self.output = np.array([0 for _ in range(4)]) # 输出层  4个神经元
        self.chromosome = np.random.randn(100)
        self.rebuild_from_chromosome()
    def run(self, input):
        # 输入格式：
        # ↑ ↗ → ↘ ↓ ↙ ← ↖
        self.input  = input
        self.l2     = relu_l(np.dot(self.w_in_to_2,  self.input) + self.b_in_to_2)
        self.l3     = relu_l(np.dot(self.w_2_to_3,   self.l2)    + self.b_2_to_3)
        self.output = relu_l(np.dot(self.w_3_to_out, self.l3)    + self.b_3_to_out)
        # 输出格式：
        # ↑ ← ↓ →
        return self.output

    def rebuild_from_chromosome(self):
        self.w_in_to_2  = self.chromosome[                    :32                      ].reshape(4, 8)
        self.w_2_to_3   = self.chromosome[32                  :32 + 16                 ].reshape(4, 4)
        self.w_3_to_out = self.chromosome[32 + 16             :32 + 16 + 16            ].reshape(4, 4)
        self.b_in_to_2  = self.chromosome[32 + 16 + 16        :32 + 16 + 16 + 4        ]
        self.b_2_to_3   = self.chromosome[32 + 16 + 16 + 4    :32 + 16 + 16 + 4 + 4    ]
        self.b_3_to_out = self.chromosome[32 + 16 + 16 + 4 + 4:32 + 16 + 16 + 4 + 4 + 4]

    @classmethod
    def crossover(cls, a, b):
        crosspoint = random.randint(0, 99)
        c_chromosome = np.array([*a.chromosome[:crosspoint], *b.chromosome[crosspoint:]])
        c = cls()
        c.chromosome = c_chromosome
        c.rebuild_from_chromosome()
        return c

class Player:
    def __init__(self, action_from, pos, color):
        self.action_from = action_from
        self.x, self.y = pos
        self.direction = [0, 0]
        self.fitness = 0
        self.color = color

    def update(self, screen, food_pos, show):
        food_x, food_y = food_pos
        inputs = np.array(
            [
                int((food_x == self.x) and (food_y <  self.y)),
                int((food_x >  self.x) and (food_y <  self.y)),
                int((food_x >  self.x) and (food_y == self.y)),
                int((food_x >  self.x) and (food_y >  self.y)),
                int((food_x == self.x) and (food_y >  self.y)),
                int((food_x <  self.x) and (food_y >  self.y)),
                int((food_x <  self.x) and (food_y == self.y)),
                int((food_x <  self.x) and (food_y <  self.y))
            ]
        )
        action = self.action_from.run(inputs)
        self.direction = [0, 0]
        self.direction[0] += action[3] * 2 # x+
        self.direction[0] -= action[1] * 2 # x-
        self.direction[1] += action[2] * 2 # y+
        self.direction[1] -= action[0] * 2 # y-

        self.x += self.direction[0]
        self.y += self.direction[1]
        if self.x > 400:
            self.x = 400
        if self.x < 0:
            self.x = 0
        if self.y > 400:
            self.y = 400
        if self.y < 0:
            self.y = 0

        distance = np.sqrt(
            abs((food_x - self.x) ** 2) + \
            abs((food_y - self.y) ** 2))
        self.fitness = 566 - distance

        if show:
            rect = pygame.Rect(self.x - 5, self.y - 5, 10, 10)
            pygame.draw.rect(screen, self.color, rect)

    def __gt__(self, b):
        return self.fitness > b.fitness

    def __int__(self):
        return self.fitness

def main1(gen_end, show):
    if show:
        pygame.init()
        screen = pygame.display.set_mode((400, 400))
        players = [Player(NetWork(), [200, 200], 
                          (
                              round(_ * 0.51), 0, round((500 - _) * 0.51)
                          )) for _ in range(500)]
        font = pygame.font.SysFont(None, 50)
    else:
        screen = None
        players = [Player(NetWork(), [200, 200], (0, 0, 0)) for _ in range(500)]
    highest_fitnesses = []
    average_fitnesses = []

    for gen in tqdm(range(1, gen_end + 1, 1)):
        foodpos = [random.randint(0, 400), random.randint(0, 400)]
        for _ in range(200):
            if show:
                screen.fill(0)
                for event in pygame.event.get():
                    if event.type == QUIT:
                        return 0
            for player in players:
                player.update(screen, foodpos, show)
            if show:
                foodrect = pygame.Rect(foodpos[0] - 5, foodpos[1] - 5, 10, 10)
                pygame.draw.rect(screen, (0, 255, 0), foodrect)
                text = font.render("GEN {}".format(gen), True, (255, 255, 255))
                screen.blit(text, (0, 0))
                pygame.display.update()

        # 排序并算出最高适应度和平均适应度
        players.sort()
        highest_fitness = players[-1].fitness
        highest_fitnesses.append(highest_fitness)
        sum_fitness = 0
        for player in players:
            sum_fitness += player.fitness
        average_fitness = sum_fitness / len(players)
        average_fitnesses.append(average_fitness)

        # 选择（选择离目标最近的50个）
        players = players[-50:]

        # 交叉（随机选择450组，单点交叉后放入队列末端）
        for _ in range(450):
            p1 = random.choice(players)
            p2 = random.choice(players)
            n3 = NetWork.crossover(p1.action_from, p2.action_from)
            p3 = Player(n3, [200, 200], (0, 0, 0))
            players.append(p3)

        # 变异（每个player有一定几率改变染色体中任意五个数）
        for i in range(len(players)):
            if random.randint(1, 10) == 1 and i > 0:
                for _ in range(5):
                    mutationpoint = random.randint(0, 99)
                    players[i].action_from.chromosome[mutationpoint] = np.random.rand()

        # 将所有玩家放回起点，重置适应值，上色
        for i in range(len(players)):
            players[i].x = 200
            players[i].y = 200
            players[i].fitness = 0
            if show:
                players[i].color = (
                    round(i * 0.51), 0, round((500 - i) * 0.51)
                )

    # 演化结束，将适应度最高的玩家的染色体放入res.json
    players.sort()
    res = {
        'gen': gen_end,
        'chromosome': list(players[-1].action_from.chromosome)
    }
    res2 = {
        'gen_end': gen_end,
        'highest_fitnesses': highest_fitnesses,
        'average_fitnesses': average_fitnesses
    }
    with open("res.json", "w") as f_obj:
        json.dump(res, f_obj)
    with open("fitnesses.json", 'w') as f_obj:
        json.dump(res2, f_obj)
        
    pygame.quit()

def main2():
    with open("fitnesses.json", "r") as f_obj:
        res = json.load(f_obj)
    #fig = plt.Figure()
    x = list(range(1, res['gen_end'] + 1))
    y = res['highest_fitnesses']
    y2 = res['average_fitnesses']

    highest, = plt.plot(x, y, label="highest fitness")
    average, = plt.plot(x, y2, label="average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(handles=[highest, average], loc='best')
    plt.show()

def main3():
    pygame.init()
    with open("res.json", "r") as f_obj:
        res = json.load(f_obj)
    gen = res['gen']
    net = NetWork()
    net.chromosome = np.array(res["chromosome"])
    net.rebuild_from_chromosome()
    player = Player(net, (200, 200), (255, 255, 255))
    foodpos = [random.randint(0, 400), random.randint(0, 400)]
    font = pygame.font.SysFont(None, 50)
    screen = pygame.display.set_mode((400, 400))
    while True:
        screen.fill(0)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
        player.update(screen, foodpos, True)
        foodrect = pygame.Rect(foodpos[0] - 5, foodpos[1] - 5, 10, 10)
        pygame.draw.rect(screen, (0, 255, 0), foodrect)
        if foodrect.collidepoint((player.x, player.y)):
            foodpos = [random.randint(0, 400), random.randint(0, 400)]
        text = font.render("GEN {}".format(gen), True, (255, 255, 255))
        screen.blit(text, (0, 0))
        pygame.display.update()

if __name__ == '__main__':
    main1(int(input("进化代数：")), int(input("显示训练过程？（1/0）")))
    main2()
    main3()
