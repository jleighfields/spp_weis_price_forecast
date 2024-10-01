# Data collection 5 minute intervals
# Gather public SPP Weis data from https://marketplace.spp.org/groups/operational-data-weis
import sys
import os
import shutil
import pandas as pd
import duckdb

import requests
from io import StringIO

import ibis
import ibis.selectors as s
ibis.options.interactive = True

# logging
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

log.info('GETTING 5 MIN LMPS')
n_periods=12*24 # 12 intervals per hour * 1 day lookback
dc.collect_upsert_lmp(n_periods=n_periods, daily_file=False)

log.info('GETTING MTLF')
dc.collect_upsert_mtlf(n_periods=24)

log.info('GETTING MTRF')
dc.collect_upsert_mtrf(n_periods=24)

log.info('copying database')
shutil.copyfile('data/spp.ddb', '~/spp.ddb')

log.info('FINISHED')



