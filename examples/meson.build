examples = {
    'cuckoo-filter-ipc': {'src': 'ipc_main.cu'},
    'cuckoo-filter-ipc-server': {'src': 'ipc_server_main.cu'},
    'cuckoo-filter-cuda': {'src': 'main.cu'},
    'cuckoo-filter-multi-gpu': {'src': 'multi_gpu_main.cu', 'deps': [gossip_dep]},
}

foreach name, config : examples
    executable(
        name,
        config['src'],
        dependencies: [gpu_cuckoo_dep, cli11_dep] + config.get('deps', []),
        cuda_args: cuda_args,
        install: true,
    )
endforeach
