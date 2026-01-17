benchmark_dep = subproject('google-benchmark').get_variable('google_benchmark_dep')
cpu_cuckoo_dep = subproject('cpu-cuckoo').get_variable('cpu_cuckoo_dep')

cmake = import('cmake')

cuco_opts = cmake.subproject_options()
cuco_opts.add_cmake_defines({
    'CMAKE_BUILD_TYPE': 'Release',
    'BUILD_TESTS': 'OFF',
    'BUILD_BENCHMARKS': 'OFF',
    'BUILD_EXAMPLES': 'OFF',
    'INSTALL_CUCO': 'OFF',
})
cuco_dep = cmake.subproject('cuco', options: cuco_opts).dependency('cuco')

partitioned_opts = cmake.subproject_options()
partitioned_opts.add_cmake_defines({
    'CMAKE_BUILD_TYPE': 'Release',
    'BUILD_TESTING': 'OFF',
})
partitioned_dep = cmake.subproject('partitioned-filters', options: partitioned_opts).dependency('filters')

gpu_filters = subproject('gpu-filters')
gqf_dep = gpu_filters.get_variable('gqf_dep')
gqf_deps = gpu_filters.get_variable('gqf_deps')
tcf_dep = gpu_filters.get_variable('tcf_dep')

bench_deps = [gpu_cuckoo_dep, benchmark_dep, cpu_cuckoo_dep]

benchmarks = {
    'cuckoo-vs-bloom': {
        'src': 'cuckoo_vs_bloom.cu',
        'deps': [cuco_dep],
    },
    'cuckoo-cpu-vs-gpu': {
        'src': 'cuckoo_cpu_vs_gpu.cu',
    },
    'cuckoo-bucket-size': {
        'src': 'cuckoo_bucket_size.cu',
        'extra_args': ['-DCUCKOO_FILTER_DISABLE_256BIT_LOADS'],
    },
    'cuckoo-sorted-vs-unsorted': {
        'src': 'cuckoo_sorted_vs_unsorted.cu',
    },
    'eviction': {
        'src': 'eviction_benchmark.cu',
    },
    'cuckoo-ipc-overhead': {
        'src': 'cuckoo_ipc_overhead.cu',
    },
    'cuckoo-vs-partitioned': {
        'src': 'cuckoo_vs_partitioned.cu',
        'deps': [partitioned_dep],
        'extra_args': ['-w'],
    },
    'cuckoo-vs-gqf': {
        'src': 'cuckoo_vs_gqf.cu',
        'deps': [gqf_dep],
    },
    'cuckoo-vs-tcf': {
        'src': 'cuckoo_vs_tcf.cu',
        'deps': [tcf_dep],
        'extra_args': ['-w'],
    },
    'load-factor': {
        'src': 'load_factor_benchmark.cu',
        'deps': [cuco_dep, tcf_dep, partitioned_dep, gqf_dep],
        'extra_args': ['-w'],
    },
    'fpr': {
        'src': 'fpr_benchmark.cu',
        'deps': [cuco_dep, tcf_dep, partitioned_dep, gqf_dep],
        'extra_args': ['-w'],
    },
    'cuckoo-atomics': {
        'src': 'cuckoo_atomics_comparison.cu',
    },
    'load-width-256bit': {
        'src': 'load_width_benchmark.cu',
    },
    'load-width-128bit': {
        'src': 'load_width_benchmark.cu',
        'extra_args': ['-DCUCKOO_FILTER_DISABLE_256BIT_LOADS'],
    },
    'cuckoo-single-vs-multi-gpu': {
        'src': 'cuckoo_single_vs_multi_gpu.cu',
        'deps': [gossip_dep],
    },
    'multi-gpu-scaling': {
        'src': 'multi_gpu_scaling_benchmark.cu',
        'deps': [gossip_dep],
    },
    'bucket-policy': {
        'src': 'bucket_policy_benchmark.cu',
    },
}

foreach name, config : benchmarks
    executable(
        name,
        config['src'],
        dependencies: bench_deps + config.get('deps', []),
        cuda_args: cuda_args + config.get('extra_args', []),
        install: false,
    )
endforeach

fpr_sweep_configs = {
    '8': gqf_deps['8'],
    '16': gqf_deps['16'],
    '32': gqf_deps['32'],
}

foreach bits, gqf_bits_dep : fpr_sweep_configs
    executable(
        'fpr-sweep-gqf' + bits,
        'fpr_sweep_benchmark.cu',
        dependencies: bench_deps + [cuco_dep, tcf_dep, gqf_bits_dep],
        cuda_args: cuda_args + ['-w', '-DGQF_BITS=' + bits],
        install: false,
    )
endforeach

executable(
    'profiler',
    'profiler_benchmark.cu',
    dependencies: [gpu_cuckoo_dep, cli11_dep, cuco_dep, tcf_dep, benchmark_dep, gqf_dep],
    cuda_args: cuda_args,
    install: false,
)

executable(
    'kmer',
    'kmer_benchmark.cu',
    dependencies: bench_deps + [cuco_dep, tcf_dep, gqf_dep],
    cuda_args: cuda_args + ['-w'],
    install: false,
)
