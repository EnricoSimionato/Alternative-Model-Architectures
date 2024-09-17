FROM python
ENV PYTHONPATH=/Alternative-Model-Architectures/src
WORKDIR /Alternative-Model-Architectures
COPY src/ src/
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["python3", "src/neuroflex/experiment_launcher.py", "CONFIG_LOCAL.yaml"]