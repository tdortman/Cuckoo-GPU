gtest_dep = subproject('gtest').get_variable('gtest_dep')
gtest_main_dep = subproject('gtest').get_variable('gtest_main_dep')

test_exe = executable(
    'test-cuckoo-filter',
    'test_cuckoo_filter.cu',
    dependencies: [gpu_cuckoo_dep, gtest_dep, gtest_main_dep],
    cuda_args: cuda_args,
    install: false,
)

test('cuckoo-filter-tests', test_exe)
