FROM ubuntu:22.04

RUN apt update && apt install -y wget cmake ninja-build gnupg

RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main" >> /etc/apt/sources.list && \
    echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main" >> /etc/apt/sources.list && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add

RUN apt update && apt install -y clang-16 lldb-16 lld-16 libmlir-16-dev mlir-16-tools

