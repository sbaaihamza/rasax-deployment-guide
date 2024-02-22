# Use the specific version of Rasa as the base image
ARG RELEASE
ARG LAUNCHPAD_BUILD_ARCH
LABEL org.opencontainers.image.ref.name=ubuntu
LABEL org.opencontainers.image.version=22.04
ADD file:aa9b51e9f0067860cebbc9930374452d1384ec3c59badb5e4733130eedc90329 in / 
CMD ["/bin/bash"]
ENV DEBIAN_FRONTEND=noninteractive
RUN /bin/sh -c apt-get update
RUN /bin/sh -c apt-get update
RUN /bin/sh -c update-alternatives --install
RUN /bin/sh -c useradd -rm
COPY /opt/venv /opt/venv # buildkit
ENV PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV HOME=/app
WORKDIR /app
COPY ./core_changes/lm_featurizer.py /opt/venv/lib/python3.10/site-packages/rasa/nlu/featurizers/dense_featurizer/lm_featurizer.py
RUN /bin/sh -c chgrp -R
USER 1001
VOLUME [/tmp]
SHELL [/bin/bash -o pipefail -c]
EXPOSE map[5005/tcp:{}]
ENTRYPOINT ["rasa"]
CMD ["--help"]
#FROM rasa/rasa:3.6.16-full

# Copy the new lm_featurizer.py file into the container, replacing the existing one
#COPY ./core_changes/lm_featurizer.py /opt/venv/lib/python3.10/site-packages/rasa/nlu/featurizers/dense_featurizer/lm_featurizer.py
