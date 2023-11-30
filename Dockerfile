FROM continuumio/miniconda3

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "transfer_learning", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure pytorch is installed:"
RUN python -c "import torch"

WORKDIR /app

# Run the application
CMD ["conda", "run", "--no-capture-output", "-n", "transfer_learning", "python", "app.py"]