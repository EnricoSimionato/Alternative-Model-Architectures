FROM python:3.12-slim
RUN apt-get update && apt-get install -y git
# Setting the PYTHONPATH
ENV PYTHONPATH=/Alternative-Model-Architectures/src
# Setting the working directory
WORKDIR /Alternative-Model-Architectures
# Copying the entire contents of the Redundancy-Hunter directory
COPY . .
# Installing dependencies
#RUN pip install --no-cache-dir -r requirements.txt
# Logging in to the Hugging Face model hub
#RUN huggingface-cli login --token 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
# Setting the command to run the script
#CMD ["python3", "src/neuroflex/experiment_launcher.py", "CONFIG_SERVER.yaml"]
# Print all directories and files, including hidden ones
CMD ["bash"]