FROM python:3.9

# Install ChemML
COPY . /chemml
WORKDIR /chemml

RUN pip install -e .

# ChemML entrypoint
CMD ["python", "-c", "import chemml; print('ChemML ready!')"]
