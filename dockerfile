FROM python:3.7

# WORKDIR /Users/sandy/Desktop/COEN242/PA1

COPY . .


RUN pip install --no-cache-dir pandas psutil


ENTRYPOINT ["python3", "topkword.py"]



