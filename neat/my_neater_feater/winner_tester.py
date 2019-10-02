import neat
import pickle
import random
import os
from my_neater_feater.my_neat import winner_of_population_after_1, my_sigmoid


new_ls = pickle.load(open('no_remake.pck', 'rb'))

# set up config
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Add my sigmoid function which ranges between -1 and 1.
config.genome_config.add_activation('my_sigmoid', my_sigmoid)

# import winner from last winner file or import population checkpooint
option = input('Do you want to import winner from checkpoint or last winner?')
if option == 'checkpoint':
    file = input("Enter checkpoint file name")
    winner = winner_of_population_after_1(file)
else:
    l_winner = pickle.load(open('winner_net', 'rb'))
    winner = l_winner[0]

print('\nBest genome:\n{!s}'.format(winner))
print('\nOutput:')
winner_net = neat.nn.RecurrentNetwork.create(winner, config)
test_games = 100
outputs = 14
n_correct_answers = 0
turns = 0
for game in new_ls[-test_games:]:
    print('\n\n\nNEW GAME\n')
    correct_answer = game[1]
    liberal_seat = random.choice([correct_answer.index(element) for element in correct_answer[:7] if element == 1])
    memory = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
    memory[liberal_seat] = 1
    memory[liberal_seat + 7] = 0
    for turn in game[0]:
        turns+=1
        inputs = list(map(float, turn)) + memory
        output = winner_net.activate(inputs)
        memory = output.copy()
        print([round(i, 2) for i in output[:outputs]])
        print(correct_answer[:outputs])
        c_answers = outputs - sum([abs(round(out) - real) for out, real in zip(output[:outputs], correct_answer[:outputs])])
        n_correct_answers +=c_answers
        print(f'network had {c_answers}/{outputs} answers correct after turn {game[0].index(turn)}\n')
        memory[liberal_seat] = 1
        memory[liberal_seat + 7] = 0

print(f'Network ended up with guess percentage of {(n_correct_answers/(outputs*turns))*100}%')
node_names = {-1: 'input', 0: 'output'}

