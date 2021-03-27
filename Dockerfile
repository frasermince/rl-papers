from dvcorg/cml-py3:latest
RUN apt-get update && \
    apt-get -y install libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig python-opengl && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip/
CMD ["cml-cloud-runner-entrypoint"]
