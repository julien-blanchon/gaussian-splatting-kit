# --- `colmap` Builder Stage ---
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04 AS colmap_builder

ARG COLMAP_GIT_COMMIT=main
ARG CUDA_ARCHITECTURES=native
ENV QT_XCB_GL_INTEGRATION=xcb_egl

WORKDIR /workdir

# Prepare and empty machine for building.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    && rm -rf /var/lib/apt/lists/*

# Build and install COLMAP.
COPY deps/colmap /colmap
RUN cd /colmap && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ninja && \
    ninja install && \
    cd .. && rm -rf colmap

# # --- `gaussian-splatting-cuda` Builder Stage ---
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04 AS gs_builder

WORKDIR /workdir

# Install dependencies
# we could pin them to specific versions to be extra sure
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    python3-dev \
    libtbb-dev \
    libeigen3-dev \
    unzip \
    g++ \
    libssl-dev \
    build-essential \
    checkinstall \
    wget \
    cmake \
    protobuf-compiler \
 && rm -rf /var/lib/apt/lists/*

# Install cmake 3.25
# RUN apt-get update && apt-get -y install 
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.0/cmake-3.25.0.tar.gz \
 && tar -zvxf cmake-3.25.0.tar.gz \
 && cd cmake-3.25.0 \
 && ./bootstrap \
 && make -j8 \
 && checkinstall --pkgname=cmake --pkgversion="3.25-custom" --default

# Copy necessary files
COPY deps/gaussian-splatting-cuda/cuda_rasterizer ./cuda_rasterizer
COPY deps/gaussian-splatting-cuda/external ./external
COPY deps/gaussian-splatting-cuda/includes ./includes
COPY deps/gaussian-splatting-cuda/parameter ./parameter
COPY deps/gaussian-splatting-cuda/src ./src
COPY deps/gaussian-splatting-cuda/CMakeLists.txt ./CMakeLists.txt

# Download and extract libtorch
RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip \
 && unzip -o libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip -d external/ \
 && rm libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip

# Build (on CPU, this will add compute_35 as build target, which we do not want)
ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
RUN cmake -B build -D CMAKE_BUILD_TYPE=Release -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ -D CUDA_VERSION=11.7 \
 && cmake --build build -- -j8

# --- Runner Stage ---
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04 AS runner

WORKDIR /app

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    imagemagick \
    ffmpeg \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy built artifact from colmap_builder stage
COPY --from=colmap_builder /usr/local/bin/colmap /usr/local/bin/colmap

# Copy built artifact from builder stage
COPY --from=gs_builder /workdir/build/gaussian_splatting_cuda /usr/local/bin/gaussian_splatting_cuda
COPY --from=gs_builder /workdir/external/libtorch /usr/local/libtorch
COPY --from=gs_builder /workdir/parameter /usr/local/bin/parameter

# Setup environment
ENV PATH /usr/local/libtorch/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/libtorch/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LC_ALL C
ENV LANG C

# Install python dependencies
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /app/requirements.txt

COPY services /app/services
COPY server.py /app/server.py

# Fix bug
RUN mkdir /parameter && cp /usr/local/bin/parameter/optimization_params.json /parameter/optimization_params.json

EXPOSE 7860
CMD [ "python3", "-u", "/app/server.py" ]