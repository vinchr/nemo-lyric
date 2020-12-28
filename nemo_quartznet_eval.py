from configparser import ConfigParser
import argparse, os, sys

import Evaluate

'''
try:
    from jamendolyrics import Evaluate
except ImportError:
    print('jamendolyrics is not found.  code assumes that repository is cloned into nemo-lyrics directory. pulling from github, and patching code.')
    os.system('git clone https://github.com/f90/jamendolyrics')
    os.system('patch < misc/Evaluate.patch')
    try:
        from jamendolyrics import Evaluate
    except ImportError:
        print('jamendolyrics is not found.  git clone did not work.')
        sys.exit()
'''



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run alignment evaluation with Jamendo lyrics")
    parser.add_argument('-c',"--config", required=True, default='test.cfg', type=str,help='configuration file for performing alignment.')
    args = parser.parse_args()

    filename = args.config
    config = ConfigParser(inline_comment_prefixes=["#"])
    config.read(filename)
    results = Evaluate.compute_metrics(config)
    Evaluate.print_results(results)
