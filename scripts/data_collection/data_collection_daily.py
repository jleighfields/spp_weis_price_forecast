
'''
Gather daily 5 minute lmp files from SPP Weis: https://marketplace.spp.org/groups/operational-data-weis
Also get daily generation capacity files
'''

import sys
import os
import logging

# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# adding module folder to system path
# needed for running scripts as jobs
home = os.getenv('HOME')
module_paths = [
    f'{home}/spp_weis_price_forecast/src',
    f'{home}/Documents/spp_weis_price_forecast/src',
]
for module_path in module_paths:
    if os.path.isdir(module_path):
        log.info('adding module path')
        sys.path.insert(0, module_path)


log.info(f'os.getcwd(): {os.getcwd()}')
log.info(f'os.listdir(): {os.listdir()}')

# from module path
import data_collection as dc


# lookback 10 days to cover holidays and long weekends when data doesn't get updated
dc.collect_upsert_lmp(n_periods=10, daily_file=True)
dc.collect_upsert_gen_cap(n_periods=10)



