gqf_inc = include_directories('include')

base_cuda_arch_args = [
    '-gencode=arch=compute_120,code=sm_120',
    '-gencode=arch=compute_90,code=sm_90',
    '-gencode=arch=compute_80,code=sm_80',
]

base_cuda_args = [
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '--generate-line-info',
] + base_cuda_arch_args + [
    # gqf.cu calls aMurmurHash64A/ahash_* in hashutil.cu (cross-TU __device__).
    '-rdc=true',
    # Upstream gpu-filters/gqf is noisy: we only consume the built library.
    '-w',
    '-Xcompiler=-w',
]

gqf_sources = ['src/gqf.cu', 'src/hashutil.cu', 'src/partitioned_counter.cu']

# Build multiple GQF variants with different bit configurations
# Format: [bits_per_slot, r_bits, suffix]
gqf_configs = [
    [8, 8, '8'],
    [16, 16, '16'],
    [32, 32, '32'],
]

gqf_libs = {}
gqf_deps = {}

foreach config : gqf_configs
    bits_per_slot = config[0]
    r_bits = config[1]
    suffix = config[2]

    # Add suffix to function names to avoid symbol conflicts
    config_cuda_args = base_cuda_args + [
        '-D', 'QF_BITS_PER_SLOT=@0@'.format(bits_per_slot),
        '-D', 'R_BITS=@0@'.format(r_bits),
        '-D', 'GQF_SUFFIX=_@0@'.format(suffix),
    ]

    lib_name = 'gqf_@0@'.format(suffix)
    gqf_libs += {suffix: static_library(
        lib_name,
        gqf_sources,
        include_directories: gqf_inc,
        cuda_args: config_cuda_args,
    )}

    gqf_deps += {suffix: declare_dependency(
        include_directories: gqf_inc,
        link_with: gqf_libs[suffix],
        compile_args: ['-D', 'QF_BITS_PER_SLOT=@0@'.format(bits_per_slot)],
        # Device-link libgqf at final executable link only. Repeat arch flags here so
        # nvcc does not fall back to its default single-arch device-link output.
        link_args: base_cuda_arch_args + [
            '-rdc=true',
            '-Xnvlink=--suppress-stack-size-warning',
        ],
    )}
endforeach

# Default dependency uses 8-bit slots (better bits/item vs load factor at power-of-2 sizes).
gqf_dep = gqf_deps['8']
