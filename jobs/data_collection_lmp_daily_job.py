# Install the Lightning SDK
# pip install lightning-sdk

from lightning_sdk import Machine, Studio, JobsPlugin, MultiMachineTrainingPlugin
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

print("starting Studio...")
s = Studio(name="model-train", teamspace="spp-weis", user="jleighfields-yst2q")
s.start(machine=Machine.CPU)
s.run("cd ~/spp_weis_price_forecast && python scripts/model_retrain.py")
print("Stopping Studio")
s.stop()
