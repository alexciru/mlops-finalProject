FROM pytorch/torchserve:0.3.0-cpu

COPY mlops_finalproject/models/model.py inference_model_32img_20230118_2315.pt mlops_finalproject/inference/mnet_handler.py /home/model-server/
COPY requirements.txt requirements.txt
COPY setup.py setup.py

USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
USER model-server


RUN python -m pip install --upgrade pip
RUN pip install pytorch-lightning
RUN pip install timm

RUN torch-model-archiver \
  --model-name=mnet \
  --version=1.0 \
  --model-file=/home/model-server/model.py \
  --serialized-file=/home/model-server/inference_model_32img_20230118_2315.pt \
  --handler=/home/model-server/mnet_handler.py \
  --export-path=/home/model-server/model-store

CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "mnet=mnet.mar"]