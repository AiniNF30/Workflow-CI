name: CI/CD MLflow

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

env:
  MODEL_ARTIFACT_PATH: "model"

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Setup Python 3.12.7
        uses: actions/setup-python@v4
        with:
          python-version: 3.12.7

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run training script with MLflow tracking
        run: |
          python MLProject/modelling.py

      - name: Get latest MLflow run_id
        id: get_run_id
        run: |
          RUN_ID=$(ls -td MLProject/mlruns/0/*/ | head -n 1 | cut -d'/' -f4)
          echo "RUN_ID=$RUN_ID" >> $GITHUB_ENV
          echo "Run ID: $RUN_ID"

      - name: Build Docker image from MLflow model
        run: |
          mlflow models build-docker --model-uri "runs:/${{ env.RUN_ID }}/$MODEL_ARTIFACT_PATH" --name "energy-model"

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_HUB_USERNAME }}
          password: ${{ secrets.DOCKER_HUB_ACCESS_TOKEN }}

      - name: Tag Docker image
        run: |
          docker tag energy-model ${{ secrets.DOCKER_HUB_USERNAME }}/energy-model:latest

      - name: Push Docker image to Docker Hub
        run: |
          docker push ${{ secrets.DOCKER_HUB_USERNAME }}/energy-model:latest
