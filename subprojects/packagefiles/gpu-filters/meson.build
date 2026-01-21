project('gpu-filters', 'cpp', 'cuda', default_options: ['optimization=3'])

patch_files = [
    'bulk-tcf/name-collision.patch',
    'bulk-tcf/params.patch',
    'gqf/fix-popcnt.patch',
    'gqf/fix-qf-malloc-device.patch',
    'gqf/rename-hash-functions.patch',
]

fs = import('fs')

if not fs.exists('.patched')
    foreach patch : patch_files
        message('Applying patch ' + patch)
        run_command('git', 'apply', '--ignore-whitespace', patch, check: true)
    endforeach
    run_command('touch', '.patched', check: true)
else
    message('Skipping patching')
endif

subdir('bulk-tcf')
subdir('gqf')