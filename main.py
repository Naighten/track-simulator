import argparse
from CarGameEnv import CarGameEnv
from neatmodel.utils import run_neat
from rl.agent import RLAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['neat', 'rl'], required=True)
    parser.add_argument('--checkpoint', help='Path to NEAT checkpoint')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    env = CarGameEnv(render_mode='human' if args.render else None)

    if args.algo == 'neat':
        run_neat(env, args.checkpoint)
    elif args.algo == 'rl':
        agent = RLAgent(env)
        if args.checkpoint:
            agent.load_neat_checkpoint(args.checkpoint)
        agent.train()


if __name__ == '__main__':
    main()