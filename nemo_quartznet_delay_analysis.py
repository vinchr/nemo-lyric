from configparser import ConfigParser
import argparse, os, sys
import numpy as np
import Evaluate

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run alignment evaluation with Jamendo lyrics")
    parser.add_argument('-c',"--config", required=False, default='test/sample.cfg', type=str,help='configuration file for performing alignment.')
    args = parser.parse_args()

    filename = args.config
    config = ConfigParser(inline_comment_prefixes=["#"])
    config.read(filename)

    delay = np.arange(-0.1,0.5,0.01)
    ae_list = []
    results_list = []

    for i in delay:
        config['main']['DELAY'] = str(i)
        results = Evaluate.compute_metrics(config)
        #Evaluate.print_results(results)
        ae_list.append(results['mean_AE'][0])
        results_list.append(results)

    ndx = np.argmin(ae_list)
    print('Lowest Error was index:',ndx)
    print( Evaluate.print_results( results_list[ndx]) )
    print(ae_list)
