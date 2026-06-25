FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        git \
        git-lfs \
        gpg \
        jq \
        libffi-dev \
        lld \
        make \
        nano \
        openssh-client \
        openssh-server \
        pkg-config \
        python3 \
        python3-pip \
        ripgrep \
        rsync \
        tmux \
        unzip \
        vim \
        wget \
        xz-utils \
        zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable \
    && rustup component add rustfmt clippy \
    && chmod -R a+w /usr/local/cargo /usr/local/rustup

RUN mkdir -p /run/sshd \
    && sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

WORKDIR /workspace

COPY runpod/dev-image/entrypoint.sh /usr/local/bin/runpod-entrypoint
RUN chmod +x /usr/local/bin/runpod-entrypoint

CMD ["runpod-entrypoint"]
