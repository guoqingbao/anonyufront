# Ufront to Tosa Converter

## How to build?

### Install tools for building
```sh
apt update && apt install -y wget cmake ninja-build gnupg
```

### Install openmp
```sh
apt install libomp-16-dev
```

#### Install LLVM/MLIR packages
```sh
echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main" >> /etc/apt/sources.list && \
    echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main" >> /etc/apt/sources.list && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add

apt update && apt install -y clang-16 lldb-16 lld-16 libmlir-16-dev mlir-16-tools   
```

### Build the project
```sh
cmake .. -G Ninja -DMLIR_DIR=/usr/lib/llvm-16/lib/cmake/mlir && \
    ninja && \
```
## How to use?

#### Partial lowering
```sh
ufront-opt test/ufront.mlir --convert-ufront-to-tosa
```

#### Full lowering
```sh
ufront-opt test/ufront.mlir --convert-ufront-to-tosa --test-convert-elided-to-const
```