gqf_inc = include_directories('include')

base_cuda_args = [
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '--generate-line-info',
    '-gencode=arch=compute_120,code=sm_120',
    '-gencode=arch=compute_75,code=compute_75',
    '-rdc=true',
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
    )}
endforeach

# Default dependency uses 16-bit configuration
gqf_dep = gqf_deps['16']
