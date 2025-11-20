project(
    'cpu-cuckoo',
    'cpp',
    version: '1.0.0',
    default_options: [
        'cpp_std=c++11',
        'warning_level=2',
        'optimization=3'
    ],
)

openssl_dep = subproject('openssl').get_variable('openssl_dep')

inc = include_directories('src')

cpu_cuckoo_lib = static_library(
    'cuckoofilter',
    'src/hashutil.cc',
    include_directories: inc,
    dependencies: openssl_dep,
    install: false,
)

cpu_cuckoo_dep = declare_dependency(
    link_with: cpu_cuckoo_lib,
    include_directories: inc,
    dependencies: openssl_dep,
)

meson.override_dependency('cpu-cuckoo', cpu_cuckoo_dep)