{
  lib,
  pkgs,
  mkShell,
  ...
}:

let
  cudaPkgs = pkgs.cudaPackages_12_9;
  llvm = pkgs.llvmPackages_21;

  cuda = {
    arch = "1200";
    smTarget = "sm_120";
    path = cudaPkgs.cudatoolkit;
    version = {
      complete = cudaPkgs.cudaMajorMinorVersion;
      major = cudaPkgs.cudaMajorVersion;
      minor = lib.lists.last (builtins.splitVersion cuda.version.complete);
    };
  };

  buildInputs = [
    cudaPkgs.cudatoolkit
    cudaPkgs.cuda_cudart
    pkgs.stdenv.cc.cc.lib
  ];

  nativeBuildInputs = with pkgs; [
    llvm.clang-tools
    llvm.lldb
    meson
    uv
    pkg-config
    doxygen
    graphviz
  ];
in

mkShell {
  inherit buildInputs nativeBuildInputs;

  CPATH = lib.makeIncludePath [ cuda.path ];

  LD_LIBRARY_PATH = "${
    lib.makeLibraryPath (buildInputs ++ nativeBuildInputs)
  }:/run/opengl-driver/lib";

  shellHook = ''
        if [ ! -e .clangd ]; then
          cat > .clangd <<EOF
    CompileFlags:
      Compiler: ${cuda.path}/bin/nvcc
      Add:
        - -xcuda
        - --cuda-path=${cuda.path}
        - -D__INTELLISENSE__
        - -D__CLANGD__
        - -I${cuda.path}/include
        - -I$(pwd)/include
        - -I$(pwd)/subprojects/cuco/include
        - -I$(pwd)/subprojects/googletest-1.17.0/googletest/include
        - -I$(pwd)/subprojects/openssl-3.0.8/include
        - -I$(pwd)/subprojects/gossip/include
        - -D__LIBCUDAXX__STD_VER=${cuda.version.major}
        - -D__CUDACC_VER_MAJOR__=${cuda.version.major}
        - -D__CUDACC_VER_MINOR__=${cuda.version.minor}
        - -D__CUDA_ARCH__=${cuda.arch}
        - --cuda-gpu-arch=${cuda.smTarget}
        - -D__CUDACC_EXTENDED_LAMBDA__
      Remove:
        - -Xcompiler=*
        - -G
        - "-arch=*"
        - "-Xfatbin*"
        - "-gencode*"
        - "--generate-code*"
        - "--generate-line-info"
        - "--compiler-options*"
        - "--expt-extended-lambda"
        - "--expt-relaxed-constexpr"
        - "-forward-unknown-to-host-compiler"
        - "-Werror=cross-execution-space-call"

    Diagnostics:
      UnusedIncludes: None
      Suppress:
        - variadic_device_fn
        - attributes_not_allowed
        - undeclared_var_use_suggest
        - typename_invalid_functionspec
        - expected_expression
    EOF
          echo ".clangd created by flake shellHook"
        fi
  '';
}
