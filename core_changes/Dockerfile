# Use the specific version of Rasa as the base image
FROM rasa/rasa:3.6.16-full

# Copy the new lm_featurizer.py file into the container, replacing the existing one
COPY lm_featurizer.py /opt/venv/lib/python3.10/site-packages/rasa/nlu/featurizers/dense_featurizer/lm_featurizer.py
