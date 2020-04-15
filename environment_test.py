import gym

from losing_connect_four.player import RandomPlayer, DeepQPlayer

env = gym.make('ConnectFour-v1')

done = False

while not done:
    random_player = RandomPlayer(env)
    dq_player = DeepQPlayer(env)
    action = random_player.get_next_action()

    state, reward, done, _ = env.step(action)

    env.render()
    env.change_player()
