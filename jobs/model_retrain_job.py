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

load_dotenv()  # load environment variables from .env

log.info("starting Studio...")
# s = Studio(name="model-train", teamspace="spp-weis", user="jleighfields-yst2q")
s = Studio(name="data-collection", teamspace="spp-weis", user="jleighfields-yst2q")
s.start()


# switch machine for training
log.info(f's.machine: {s.machine}')
if str(s.machine) != 'Machine.DATA_PREP':
    log.info('switching to Machine.DATA_PREP')
    s.switch_machine(machine=Machine.DATA_PREP)


# wait for studio to be running
log.info(f'status: {s.status}')
i = 0
while str(s.status) != 'Status.Running':
    
    i+=1
    if i > 30:
        break
    time.sleep(30)
    log.info(f'status: {s.status} - count: {i}')


# run training script
log.info('start training')
try:
    s.run("cd ~/spp_weis_price_forecast && python scripts/model_retrain.py")
except:
    pass
finally:
    log.info('switching to CPU machine')
    s.switch_machine(Machine.CPU)


log.info("Job complete")
