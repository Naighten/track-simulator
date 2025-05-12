import pickle
import neat
import numpy as np


def run_neat(env, checkpoint_path=None):
    config_path = "neatmodel/config.txt"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            total_reward = 0
            obs, _ = env.reset()

            while True:
                action = np.array(net.activate(obs))
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                if done:
                    break

            genome.fitness = total_reward

    if checkpoint_path:
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    else:
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 100)

    # Save best genome
    with open("neatmodel/best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    return winner