import gym

from losing_connect_four.player import RandomPlayer

env = gym.make('ConnectFour-v1')

done = False

while not done:
    random_player = RandomPlayer(env)
    action = random_player.get_next_action()
    result = env.step(action)
    done = result.is_done()
    env.render()
    env.change_player()
