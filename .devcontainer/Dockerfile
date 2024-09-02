# see the docs for these notebook images
# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
# https://hub.docker.com/r/jupyter/scipy-notebook/tags/

FROM jupyter/scipy-notebook:python-3.11

# Install dependencies
RUN echo conda install pytorch
# RUN conda install -y  -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.7
RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# RUN conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

RUN echo pip install packages
RUN pip install scikit-learn darts neuralforecast \
pandas pyarrow seaborn plotly \
python-dotenv xgboost streamlit shiny \
jupyterlab jupyterlab-git ipywidgets \
pytest pylint rsconnect-python

RUN pip install polars duckdb "ibis-framework[duckdb]"

RUN echo "Finished setting up..."

EXPOSE 8888-8899 8501

# set jupyter lab configurations
# https://stackoverflow.com/questions/61973485/how-do-i-launch-jupyter-lab-with-the-dark-theme-enabled-by-default
# https://jupyterlab.readthedocs.io/en/stable/user/directories.html#settings
COPY overrides.json /opt/conda/share/jupyter/lab/settings/

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--port=8899","--allow-root","--ServerApp.token=spp_weis"]
