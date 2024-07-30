FROM python:3.11.9-slim-bullseye

WORKDIR /usr/local/bin

RUN pip install pandas scikit-learn scipy

COPY validate.py .
COPY score.py .
COPY metrics.py .

ENTRYPOINT [ "python" ]
CMD [ "score.py" ]
