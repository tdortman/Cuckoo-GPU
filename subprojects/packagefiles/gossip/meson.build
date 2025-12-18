project('gossip', 'cuda', 'cpp',
  version: '1.0.0',
  default_options: ['cpp_std=c++11', 'warning_level=2', 'optimization=3'],
)

run_command('git', 'submodule', 'update', '--init', '--recursive', check: false)

gossip_inc = include_directories('include')
gossip_dep = declare_dependency(
  include_directories: [gossip_inc],
)
