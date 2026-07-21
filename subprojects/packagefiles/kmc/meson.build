project('kmc', 'cpp',
    version: '3.2.4',
    default_options: ['cpp_std=c++17', 'warning_level=1', 'optimization=3'],
)

zlib_dep = subproject('zlib').get_variable('zlib_dep')

host_cpu = host_machine.cpu_family()
is_x86 = host_cpu in ['x86', 'x86_64']
is_arm = host_cpu in ['aarch64', 'arm']

cpp = meson.get_compiler('cpp')

kmc_api_sources = files(
    'kmc_api/kmc_file.cpp',
    'kmc_api/kmer_api.cpp',
    'kmc_api/mmer.cpp',
)

kmc_api_lib = static_library('kmc_api',
    kmc_api_sources,
    include_directories: include_directories('.'),
)

kmc_api_dep = declare_dependency(
    link_with: kmc_api_lib,
    include_directories: include_directories('.'),
)

kmc_core_sources = files(
    'kmc_core/mem_disk_file.cpp',
    'kmc_core/rev_byte.cpp',
    'kmc_core/bkb_writer.cpp',
    'kmc_core/cpu_info.cpp',
    'kmc_core/bkb_reader.cpp',
    'kmc_core/fastq_reader.cpp',
    'kmc_core/timer.cpp',
    'kmc_core/develop.cpp',
    'kmc_core/kb_completer.cpp',
    'kmc_core/kb_storer.cpp',
    'kmc_core/kmer.cpp',
    'kmc_core/splitter.cpp',
    'kmc_core/kb_collector.cpp',
    'kmc_core/kmc_runner.cpp',
    'kmc_core/kff_writer.cpp',
)

raduls_libs = []

if is_x86
    # Each raduls file must compile with ONLY its target SIMD level's macro defined.
    raduls_sse2 = static_library('raduls_sse2',
        'kmc_core/raduls_sse2.cpp',
        include_directories: include_directories('.'),
        cpp_args: ['-msse2', '-mno-sse4.1', '-mno-avx', '-mno-avx2',
                   '-U__SSE4_1__', '-U__AVX__', '-U__AVX2__'],
    )
    raduls_libs += raduls_sse2

    raduls_sse41 = static_library('raduls_sse41',
        'kmc_core/raduls_sse41.cpp',
        include_directories: include_directories('.'),
        cpp_args: ['-msse4.1', '-mno-avx', '-mno-avx2',
                   '-U__AVX__', '-U__AVX2__'],
    )
    raduls_libs += raduls_sse41

    raduls_avx = static_library('raduls_avx',
        'kmc_core/raduls_avx.cpp',
        include_directories: include_directories('.'),
        cpp_args: ['-mavx', '-mno-avx2', '-U__AVX2__'],
    )
    raduls_libs += raduls_avx

    raduls_avx2 = static_library('raduls_avx2',
        'kmc_core/raduls_avx2.cpp',
        include_directories: include_directories('.'),
        cpp_args: ['-mavx2'],
    )
    raduls_libs += raduls_avx2
elif is_arm
    raduls_neon = static_library('raduls_neon',
        'kmc_core/raduls_neon.cpp',
        include_directories: include_directories('.'),
    )
    raduls_libs += raduls_neon
endif

kmc_core_lib = static_library('kmc_core',
    kmc_core_sources,
    include_directories: include_directories('.'),
    dependencies: [zlib_dep],
)

kmc_exe = executable('kmc',
    'kmc_CLI/kmc.cpp',
    link_with: [kmc_core_lib, kmc_api_lib] + raduls_libs,
    include_directories: include_directories('.'),
    dependencies: [zlib_dep],
    install: true,
)
