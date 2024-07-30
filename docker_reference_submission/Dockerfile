#FROM python:3.9-slim
#
#RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
#
#RUN mkdir -p /opt/algorithm /input /output
#
#USER root
#RUN chown -R algorithm:algorithm /opt/algorithm /input /output
#
#USER algorithm
#
#WORKDIR /opt/algorithm
#
#ENV PATH="/home/algorithm/.local/bin:${PATH}"
#
#RUN python -m pip install --user -U pip
#
#COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
#RUN python -m pip install --user -rrequirements.txt
#
#COPY --chown=algorithm:algorithm process.py /opt/algorithm/
#RUN chmod -R 777 /opt/algorithm /input /output
#
#CMD ["python", "-m",  "process"]

#FROM python:3.11.9-slim-bullseye
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /usr/local/bin

COPY . .

RUN python -m pip install --user -rrequirements.txt

ENTRYPOINT [ "python" ]
CMD [ "process.py" ]
