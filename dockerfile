FROM python:3.7

# WORKDIR /Users/sandy/Desktop/COEN242/PA1

COPY . .

# ADD stopword.txt .

RUN pip install --no-cache-dir pandas psutil



# COPY . .

ENTRYPOINT ["python3", "topkword.py"]

# CMD ["10", "10000000", "5"]


