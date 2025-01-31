FROM python:3.9

RUN apt-get update && \
    apt-get install -y zip htop screen libgl1-mesa-glx

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir kserve==v0.10.1
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "./kserve-detect.py"]
