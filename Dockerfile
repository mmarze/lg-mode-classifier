FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .

COPY src/ src/
COPY examples/ examples/

RUN pip install --no-cache-dir .

# CMD ["python", "examples/lgbeam/LG00.py"]
# CMD ["python", "examples/lgbeam/LG02.py"]
# CMD ["python", "examples/lgbeam/LG10.py"]
# CMD ["python", "examples/lgbeam/mixture.py"]
CMD ["python", "examples/lgbeam/vortex.py"]