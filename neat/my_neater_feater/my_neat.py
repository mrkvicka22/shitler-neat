import os
import pickle
import random

import math
import neat

from my_neater_feater import visualize

all_games = pickle.load(open('no_remake.pck', 'rb'))


def my_sigmoid(x):
    return (2/(1+math.e**-x))-1


def winner_of_population_after_1(file_name):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Add my sigmoid function which ranges between -1 and 1.
    config.genome_config.add_activation('my_sigmoid', my_sigmoid)

    pop = neat.Checkpointer.restore_checkpoint(file_name)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pe = neat.ParallelEvaluator(25, eval_genome)
    winner = pop.run(pe.evaluate, generations)
    return winner


def guessed_hitler_part(output,correct_answer, hitler_seat):
    output1 = output[-7:].copy()
    output1.pop(hitler_seat)
    correct_answer1 = correct_answer[-7:].copy()
    correct_answer1.pop(hitler_seat)
    first_part = 2 / 3 * sum([(out-real)**2 for out,real in zip(output1, correct_answer1)])
    second_part = 1 / 3 * ((output[hitler_seat] - correct_answer[hitler_seat]) ** 2)
    return first_part+second_part



def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    # Create a random sequence, and feed it to the network with the
    # second input set to zero.
    net.reset()

    # Declare variables for evaluation
    error = 0
    hit_error = 0
    games = 10
    turns = 0

    # Pick random sample of games of games from all the games to evaluate
    sample = random.sample(all_games,games)

    # Go through each game in sample games.
    for game in sample:
        # get correct answer list
        correct_answer = game[1]

        # Choose one of the liberal seats
        liberal_seat = random.choice([correct_answer.index(element) for element in correct_answer[:7] if element == 1])

        # Get index of Hitler's seat
        hitler_seat = correct_answer[-7:].index(1)

        # Starting memory with the liberal seat being set to 1 as network should acknowledge that it is liberal.
        # Also it's probability of being hitler should be set to 0
        memory = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1/6, 1/6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
        memory[liberal_seat] = 1
        memory[liberal_seat + 7] = 0

        # Evaluate each turn in games turns.
        for turn in game[0]:
            # For the first part of the input read turn info. And second file should be memory.
            # Memory in every turn other than first is passed from last 14 numbers of network's output.
            inputs = list(map(float, turn)) + memory

            # Create new output by calling evaluate.
            output = net.activate(inputs)

            # Make memory copy of the output so you can use it as input on next turn.
            memory = output.copy()

            # Sum up all the errors.
            error += sum([(out-real)**2 for out, real in zip(output[:7], correct_answer[:7])])

            # If all of the last 7 outputs are correct network is rewarded by means of reducing error.
            hit_error += guessed_hitler_part(output, correct_answer, hitler_seat)

            memory[liberal_seat] = 1
            memory[liberal_seat + 7] = 0
            turns += 1
    # Return final fitness of a genome.
    return 50*((1 - (error / (turns * 7)))+(1-hit_error/(turns*6)))


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(restore:bool,generations:int,prefix='neat-checkpoint-', file_name=''):
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Add my sigmoid function which ranges between -1 and 1.
    config.genome_config.add_activation('my_sigmoid', my_sigmoid)

    # Evaluate if you want to run from restore
    if restore:
        pop = neat.Checkpointer.restore_checkpoint(file_name)
    else:
        pop = neat.Population(config)
    print("initialized population")

    # Add reporters to print stuff and keep track of things. One of them makes checkpoints every ten generations
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(10,filename_prefix=prefix))

    # PE makes multi threading possible
    pe = neat.ParallelEvaluator(8, eval_genome)

    # Run the whole simulation
    winner = pop.run(pe.evaluate, generations)

    # Convert to phenotype (network)
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)

    # Save winner of this run to a file
    with open('winner_net', 'wb') as f:
        pickle.dump([winner, winner_net],f)

    stats.save()

    # Show output of the most fit genome against a random input.
    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    test_games = 10
    for game in all_games[-test_games:]:
        print('\n\n\nNEW GAME\n')
        correct_answer = game[1]
        hitler_seat = correct_answer[-7:].index(1)
        liberal_seat = random.choice([correct_answer.index(element) for element in correct_answer[:7] if element == 1])
        memory = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
        memory[liberal_seat] = 1
        memory[liberal_seat + 7] = 0
        for turn in game[0]:
            inputs = list(map(float, turn)) + memory
            output = winner_net.activate(inputs)
            memory = output.copy()
            print([round(i,2) for i in output])
            print(correct_answer)
            print(f'network had {14 - sum([(abs(round(out) - real)) ** 2 for out, real in zip(output, correct_answer)])}'
                  f'/14 answers correct after turn{game[0].index(turn)}')
            memory[liberal_seat] = 1
            memory[liberal_seat + 7] = 0

    node_names = {-1: 'input', 0: 'output'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True, filename='species.svg')


if __name__ == '__main__':
    ans = input('do you want to run from scratch or from restored')
    generations = int(input('Enter for how many generations you wan this to run:'))
    prefix = input('Enter prefix for your run:')

    if ans == 'scratch':
        restore = False
        run(restore, generations,prefix)

    elif ans == 'restored':
        restore = True
        file_name = input('Enter the name of your restore file')
        run(restore, generations,prefix,file_name=file_name)
