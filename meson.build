project(
    'cuckoo-filter-cuda',
    ['cuda', 'cpp'],
    version: '0.1',
    meson_version: '>= 1.3.0',
    default_options: [
        'warning_level=3',
        'cpp_std=c++20',
        'optimization=3',
    ],
)

cuda_args = [
    '-std=c++20',
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '--generate-line-info',
    '-gencode=arch=compute_120,code=sm_120',
    '-gencode=arch=compute_90,code=sm_90',
    '-gencode=arch=compute_80,code=sm_80',
    '-gencode=arch=compute_75,code=compute_75',
]

gpu_cuckoo_dep = declare_dependency(
    include_directories: include_directories('include'),
)

cli11_dep = subproject('cli11').get_variable('CLI11_dep')
gossip_dep = subproject('gossip').get_variable('gossip_dep')
kmc_api_dep = subproject('kmc').get_variable('kmc_api_dep')

executable(
    'kmc_to_binary',
    'src/kmc_to_binary.cpp',
    dependencies: [kmc_api_dep, cli11_dep],
    install: true,
)

subdir('examples')

if get_option('BUILD_BENCHMARKS')
    subdir('benchmark')
endif

if get_option('BUILD_TESTS')
    subdir('tests')
endif