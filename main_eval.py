import argparse
import os
import pickle
import sys
import logging
from disvae import Evaluator
from utils.helpers import FormatterNoDuplicate, check_bounds, set_seed
from utils.viz_helpers import get_samples
from utils.helpers import get_config_section, get_device
from main import RES_DIR
from disvae.utils.modelIO import load_model, load_metadata
from disvae.models.losses import LOSSES, RECON_DIST, get_loss_f
from utils.datasets import get_dataloaders

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())

def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")
    description = "CLI for plotting using pretrained models of `disvae`"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    parser.add_argument('name', type=str,
                        help="Name of the model for storing and loading purposes.")
    parser.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed. Can be `None` for stochastic behavior.')
    parser.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS)
    parser.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    parser.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")
    parser.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
                       choices=RECON_DIST,
                       help="Form of the likelihood ot use for each pixel.")
    parser.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")
    parser.add_argument('--betaB-initC', type=float,
                       default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    parser.add_argument('--betaB-finC', type=float,
                       default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    parser.add_argument('--betaB-G', type=float,
                       default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")
    parser.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')
    parser.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    parser.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")

    args = parser.parse_args()

    return args


def main(args):
    """Main function for plotting fro pretrained models.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))
    set_seed(args.seed)
    experiment_name = args.name
    model_dir = os.path.join(RES_DIR, experiment_name)
    meta_data = load_metadata(model_dir)
    model = load_model(model_dir)
    model.eval()  # don't sample from latent: use mean
    
    test_loader = get_dataloaders(meta_data["dataset"],
                                  batch_size=args.eval_batchsize,
                                  shuffle=False,
                                  logger=logger)
 
    loss_f = get_loss_f(args.loss,
                            n_data=len(test_loader.dataset),
                            device=device,
                            **vars(args))

    evaluator = Evaluator(model, loss_f,
                          device=device,
                          logger=logger,
                          save_dir=exp_dir,
                          is_progress_bar=not args.no_progress_bar)

    metric, losses = evaluator(test_loader, is_metrics=args.is_metrics, is_losses=not args.no_test)
    fn = os.path.join(model_dir, 'eval.pkl')
    with open(fn, 'wb') as f:
        pickle.dump([metric, losses], f)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
