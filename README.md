### This is animplementation for `Neural Counterfactual Regret Minimization with Bootstrapping Learning`.

- paper: https://arxiv.org/abs/2012.01870
- related projects:
  - This project depends on an early version of [Open_spiel](https://github.com/deepmind/open_spiel) (See `third_party/open_spiel`). 
  - Other dependencies include [Tensorflow](https://github.com/tensorflow/tensorflow) and [project_acpc_server](https://github.com/dmorrill10/project_acpc_server).

# Requirements.
## Manually installation.
1. install required packages. 
    ```bash
    apt update && apt install ssh git build-essential curl wget python3 python3-dev python3-pip python3-setuptools python3-wheel python3-tk autoconf automake libtool libffi-dev
    # install cmake-3.18
    wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
    apt update && apt install cmake
    ```
3. install `gperftools`
    ```bash
    # install libunwind-1.1
    wget http://download.savannah.gnu.org/releases/libunwind/libunwind-1.1.tar.gz 
    tar -xzf libunwind-1.1.tar.gz 
    cd libunwind-1.1    
    ./configure  
    make -j8
    make install 
    # install gperftools
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.8/gperftools-2.8.tar.gz
    tar -xzf gperftools-2.8.tar.gz
    cd gperftools-2.8 
    ./configure
    make -j8
    make install
    ```
4. install `tensorflow-cpp`
    ```bash
    # apt install
    apt install wget unzip autoconf automake libtool zlib1g-dev liblzma-dev
    # install bazel-0.26.1
    wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh
    chmod +x./bazel-0.26.1-installer-linux-x86_64.sh 
    ./bazel-0.26.1-installer-linux-x86_64.sh
    # clone tensorflow and compile tensorflow-cpp-lib
    git clone --branch v1.15.3  https://github.com/tensorflow/tensorflow
    cd tensorflow
    ./configure
    bazel build --config=opt --config=cuda --config=monolithic  //tensorflow:libtensorflow_cc.so
    ```

5. Copy tensorflow head files and binaries to third_party.
    ```bash
    cd third_party && mkdir tensorflow && cd tensorflow
    mkdir include
    cp -r /tensorflow/tensorflow ./include
    cp -r /tensorflow/third_party ./include
    cp -r /tensorflow/bazel-tensorflow/external/nsync ./include
    cp -r /tensorflow/bazel-tensorflow/external/eigen_archive ./include
    cp -r /tensorflow/bazel-tensorflow/external/com_google_absl ./include
    cp -r /tensorflow/bazel-tensorflow/external/com_google_protobuf ./include
    mkdir bazel_include
    rsync -avm --include='*.h'  -f 'hide,! */' /tensorflow/bazel-bin/ ./bazel_include
    mkdir bin
    cp -r /tensorflow/bazel-bin/tensorflow/libtensorflow_cc* ./bin
    ```
6. build algorithms.
    ```bash
    source env.sh
    cd third_party && ./install.sh
    cd ..
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j8
    ```

## Dockerfiles.
One can also build a docker image using `docker_build.sh`.

# How to run the algorithm.
1. Make dirs for results.
```bash
mdkir -p results
mkdir -p models
```
2. call `run_neural_recfr_b_FHP.sh` or `run_neural_recfr_b_HULH.sh` to run the algorithm.

# How to play a match.
1. start an ACPC server.
2. use `player_neural_recfr_b.sh` to call up a Neural ReCFR-B agent.