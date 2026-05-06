project(
    'Cuckoo-GPU',
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

cuckoogpu_dep = declare_dependency(
    include_directories: include_directories('include'),
)

build_examples = (
    get_option('examples').enabled()
    or (get_option('examples').auto() and not meson.is_subproject())
)

build_benchmarks = (
    get_option('benchmarks').enabled()
    or (get_option('benchmarks').auto() and not meson.is_subproject())
)
build_tests = (
    get_option('tests').enabled()
    or (get_option('tests').auto() and not meson.is_subproject())
)

if build_benchmarks or build_examples
    cli11_dep = subproject('cli11').get_variable('CLI11_dep')
    gossip_dep = subproject('gossip').get_variable('gossip_dep')
endif

if build_examples
    subdir('examples')
endif

if build_benchmarks
    subdir('benchmarks')
endif

if build_tests
    subdir('tests')
endif