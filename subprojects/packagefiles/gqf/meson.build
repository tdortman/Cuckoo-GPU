project('gqf', 'cpp', 'cuda')

gqf_inc = include_directories('standardQuotientFilter/GPU_version')

cmake = import('cmake')

moderngpu_opts = cmake.subproject_options()
moderngpu_opts.add_cmake_defines(
    {
        'BUILD_TESTING': 'OFF',
    },
)

moderngpu_sub = cmake.subproject('moderngpu', options: moderngpu_opts)
moderngpu_dep = moderngpu_sub.dependency('MODERNGPU')

gqf_lib = static_library(
    'gqf_lib',
    ['standardQuotientFilter/GPU_version/quotientFilter.cu'],
    include_directories: gqf_inc,
    dependencies: [moderngpu_dep],
    cuda_args: ['--extended-lambda']
)
gqf_dep = declare_dependency(include_directories: gqf_inc, link_with: gqf_lib)