FROM python:3.10.6-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY mlops_finalproject/ mlops_finalproject/
COPY reports/ reports/
COPY models/ models/

WORKDIR /
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
# torch cpu requires special command
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# DVC
COPY data.dvc data.dvc
RUN pip install dvc 'dvc[gs]'
RUN dvc init --no-scm
RUN dvc remote add -d remote gs://german-signs-bucket/
RUN dvc pull

# and finally our own module
RUN pip install -e .

ENTRYPOINT ["python", "-u", "mlops_finalproject/models/train_model.py"]