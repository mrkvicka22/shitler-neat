import os
import pickle
import random

import math
import neat

from my_neater_feater import visualize

all_games = pickle.load(open('no_remake.pck', 'rb'))


def fitness_hit_part(output, correct_answer, hitler_seat):
    error_all = sum([abs(out)-real for out,real in zip(output,correct_answer)]) ** 2
    return error_all


def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    net.reset()

    # Declare variables for evaluation
    error = 0
    training_games = 100
    turns = 0

    # Pick random sample of games of games from all the games to evaluate
    sample = random.sample(all_games, training_games)

    # Go through each game in sample games.
    for game in all_games[:training_games]:
        # get correct answer list
        correct_answer = game[1][-7:]

        # Choose one of the liberal seats
        liberal_seat = random.choice([correct_answer.index(element) for element in correct_answer[:7] if element == 0])

        # Get index of Hitler's seat
        hitler_seat = correct_answer.index(1)

        # Starting memory with the liberal seat being set to 1 as network should acknowledge that it is liberal.
        # Also it's probability of being hitler should be set to 0
        memory = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
        memory[liberal_seat] = 0

        # Evaluate each turn in games turns.
        for turn in game[0]:
            # For the first part of the input read turn info. And second file should be memory.
            # Memory in every turn other than first is passed from last 14 numbers of network's output.
            inputs = list(map(float, turn)) + memory

            # Create new output by calling evaluate.
            output = net.activate(inputs)

            # Make memory copy of the output so you can use it as input on next turn.
            memory = output.copy()

            # Product of lib,hit error is divided by 7 and subtracted from 1. It is added to error
            e = fitness_hit_part(output, correct_answer, hitler_seat)
            error += e
            turns += 1
    # Return final fitness of a genome.
    return 100 * (1 - (error / turns))


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(restore: bool, generations: int, prefix='neat-checkpoint-', file_name=''):
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'hitty_config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

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
    pop.add_reporter(neat.Checkpointer(10, filename_prefix=prefix))

    # PE makes multi threading possible
    pe = neat.ParallelEvaluator(8, eval_genome)

    # Run the whole simulation
    winner = pop.run(pe.evaluate, generations)

    # Convert to phenotype (network)
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)

    # Save winner of this run to a file
    with open('winner_net', 'wb') as f:
        pickle.dump([winner, winner_net], f)
    stats.save()

    # Show output of the most fit genome against a random input.
    print('\nBest genome:\n{!s}'.format(winner))

    # Declare variables for evaluation
    testing_games = 100
    turns = 0
    correct_t = 0
    correct_h = 0

    # Pick random sample of games of games from all the games to evaluate
    sample = random.sample(all_games, testing_games)

    # Go through each game in sample games.
    for game in sample:
        # get correct answer list
        print('\n\n\n******************************************** NEW GAME ********************************************')
        correct_answer = game[1][-7:]

        # Choose one of the liberal seats
        liberal_seat = random.choice([correct_answer.index(element) for element in correct_answer[:7] if element == 0])

        # Get index of Hitler's seat
        hitler_seat = correct_answer.index(1)

        # Starting memory with the liberal seat being set to 1 as network should acknowledge that it is liberal.
        # Also it's probability of being hitler should be set to 0
        memory = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
        memory[liberal_seat] = 0

        # Evaluate each turn in games turns.
        for turn in game[0]:
            # For the first part of the input read turn info. And second file should be memory.
            # Memory in every turn other than first is passed from last 14 numbers of network's output.
            inputs = list(map(float, turn)) + memory

            # Create new output by calling evaluate.
            output = winner_net.activate(inputs)
            print(output)
            print(correct_answer)
            # Make memory copy of the output so you can use it as input on next turn.
            memory = output.copy()
            if output.index(max(output)) == hitler_seat and output[hitler_seat] > 1/6:
                correct_h += 1
            # Product of lib,hit error is divided by 7 and subtracted from 1. It is added to error
            correct_t += 7 - sum(round(abs(out-real)) for out,real in zip(output,correct_answer))
            print(7- sum(round(abs(out - real)) for out, real in zip(output, correct_answer)),'/7','\n')
            turns += 1

    print(f'\n\nNetwork had {(correct_t/(turns*7))*100}% accuracy. ')
    print(f'guessed hitler correctly{(correct_h/turns)*100}%')
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
        run(restore, generations, prefix)

    elif ans == 'restored':
        restore = True
        file_name = input('Enter the name of your restore file')
        run(restore, generations, prefix, file_name=file_name)
