import os
import logging
import sys



def init_logging(config):
    handlers = [logging.StreamHandler()]

    # print(args)
    # quit()

    if 'log_file' in config and config['log_file'] is not None :
        os.makedirs(os.path.dirname(config['log_file']), exist_ok=True)
        handlers.append(logging.FileHandler(config['log_file'], mode='w'))

    logging.basicConfig(handlers=handlers, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    logging.info('COMMAND: %s' % ' '.join(sys.argv))
    logging.info('Arguments: {}'.format(config))