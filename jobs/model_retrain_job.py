# Install the Lightning SDK
# pip install lightning-sdk

from lightning_sdk import Machine, Studio, JobsPlugin, MultiMachineTrainingPlugin
from dotenv import load_dotenv
import time

# logging
import logging

# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

load_dotenv()  # take environment variables from .env.

log.info("starting Studio...")
s = Studio(name="model-train", teamspace="spp-weis", user="jleighfields-yst2q")
s.start()
log.info(f's.machine: {s.machine}')
if str(s.machine) != 'L4':
    s.switch_machine(machine=Machine.L4)

# ensure it's turned on and a Studio will wait for 2 minutes before shutting down.
s.auto_shutdown = True
s.auto_shutdown_time = 2 * 60  # the time is in seconds for granular control

i = 0
while str(s.status) != 'Status.Running':
    log.info(f'status: {s.status} - count: {i}')
    i+=1
    if i > 30:
        break
    time.sleep(30)

try:
    s.run("cd ~/spp_weis_price_forecast && python scripts/model_retrain.py")
except Exception as e:
        log.error("command failed with error: ", e)

s.switch_machine(Machine.CPU)

log.info("Job complete")
