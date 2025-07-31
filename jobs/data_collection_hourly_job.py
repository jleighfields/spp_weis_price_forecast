# Install the Lightning SDK
# pip install lightning-sdk

from lightning_sdk import Machine, Studio, Status
from dotenv import load_dotenv
import time

# logging
import logging

# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

load_dotenv()  # take environment variables from .env.

log.info("starting Studio...")
s = Studio(name="data-collection", teamspace="spp-weis", user="jleighfields-yst2q")
s.start()
log.info(f'{s.machine = }, {s.status = }')

if s.machine != Machine.CPU:
    s.switch_machine(machine=Machine.CPU)
    # give some time for environment to be created
    time.sleep(30)

# ensure it's turned on and a Studio will wait for 10 minutes before shutting down.
# s.auto_shutdown = True
# s.auto_shutdown_time = 10 * 60  # the time is in seconds for granular control

i = 0
while s.status != Status.Running:
    log.info(f'status: {s.status} - count: {i}')
    i += 1
    if i > 30:
        break
    time.sleep(30)

try:
    s.run("rm ~/spp_weis_price_forecast/data/spp.ddb.wal")
except:
    pass

try:
    s.run("cd ~/spp_weis_price_forecast && python scripts/data_collection/data_collection_hourly.py")

except Exception as e:
        log.error("command failed with error: %s".format(e))

log.info("Job complete")

