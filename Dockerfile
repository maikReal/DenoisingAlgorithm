FROM python3.8-slim

RUN pip3 install librosa soundfile

RUN python3 script.py