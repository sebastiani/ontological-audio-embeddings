import sys
sys.path.append('/home/akasha/projects/')
import yaml
from argparse import ArgumentParser
from ontological_audio_embeddings.src.base.BaseModel import BaseModel


parser = ArgumentParser()
parser.add_argument('--mode', type=str, help='train/infer')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--data', type=str, help='data to infer on, etiher this or define on a config file')
args = parser.parse_args()

def main():
    # loading config file
    with open(args.config, 'r') as f:
        params = yaml.load(f)

    model = BaseModel(params)

    if args.mode == 'train':
        model.train(params)

if __name__ == '__main__':
    main()
