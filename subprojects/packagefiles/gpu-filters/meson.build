project('gpu-filters', 'cpp', 'cuda', default_options: ['optimization=3'])

patch_files = [
    'bulk-tcf/name-collision.patch',
    'bulk-tcf/params.patch',
    'gqf/fix-popcnt.patch',
    'gqf/fix-qf-malloc-device.patch',
    'gqf/rename-hash-functions.patch',
]

foreach patch : patch_files
    message('Applying patch ' + patch)
    run_command('git', 'apply', '--ignore-whitespace', patch, check: true)
endforeach

subdir('bulk-tcf')
subdir('gqf')