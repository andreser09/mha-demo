ARG CUDA_VERSION="11.8.0"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"
ARG MAX_JOBS=4

FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION

# Update package lists and install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    unzip \
    git \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    curl \
    gnupg2 \
    ca-certificates \
    apt-transport-https \
    software-properties-common \
    libreoffice \
    ffmpeg \
    git-lfs \
    xvfb \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt install python3-packaging \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Download and install ChromeDriver
RUN CHROMEDRIVER_VERSION=$(curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE) && \
    wget -N https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip -P /tmp && \
    unzip /tmp/chromedriver_linux64.zip -d /tmp && \
    mv /tmp/chromedriver /usr/local/bin/chromedriver && \
    chmod +x /usr/local/bin/chromedriver && \
    rm /tmp/chromedriver_linux64.zip

# Copy Chromedriver from the builder stage (if applicable)
# COPY --from=builder /usr/local/bin/chromedriver /usr/local/bin/chromedriver

# Set up working directory and copy application code
COPY . /app
WORKDIR /app

#RUN pip3 install poetry

#RUN poetry source add --priority=explicit pytorch-gpu https://download.pytorch.org/whl/cu118

#RUN poetry add --source pytorch-gpu torch torchvision torchaudio

#RUN poetry update
# Install PyTorch and related packages
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#RUN pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install Python package (assuming it has a setup.py)

RUN pip3 install --no-cache-dir -e .

#RUN pip install transformers==4.41.2

#RUN poetry install

RUN pip3 install --no-build-isolation flash_attn

#RUN PIP_NO_BUILD_ISOLATION=1 pip3 install -e .
# Set environment variables
ENV CHROME_BIN=/usr/bin/google-chrome \
    CHROMEDRIVER=/usr/local/bin/chromedriver \
    DISPLAY=:99 \
    DBUS_SESSION_BUS_ADDRESS=/dev/null \
    PYTHONUNBUFFERED=1

# Ensure the PATH environment variable includes the location of the installed packages
ENV PATH /usr/local/bin:$PATH

# Expose the desired port
EXPOSE 8000

# Run the server
#CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000", "--documents", "--media", "--web"]
