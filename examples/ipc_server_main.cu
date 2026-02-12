#include <chrono>
#include <csignal>
#include <cuckoogpu/CuckooFilterIPC.cuh>
#include <iostream>
#include <string>
#include <thread>

using Config = cuckoogpu::Config<uint32_t, 16, 500, 128, 16, cuckoogpu::XorAltBucketPolicy>;
static constexpr char SERVER_NAME[] = "benchmark_server";

cuckoogpu::FilterIPCServer<Config>* g_server = nullptr;

void handleSignal(int signal) {
    (void)signal;

    if (g_server) {
        g_server->stop();
    }
    exit(0);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << std::format("Usage: {} <capacity>\n", argv[0]);
        return 1;
    }

    size_t capacity = std::stoul(argv[1]);

    signal(SIGTERM, handleSignal);
    signal(SIGINT, handleSignal);

    try {
        // Unlink any old shared memory just in case
        shm_unlink(("/cuckoo_filter_" + std::string(SERVER_NAME)).c_str());

        cuckoogpu::FilterIPCServer<Config> server(SERVER_NAME, capacity);
        g_server = &server;
        server.start();

        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

    } catch (const std::exception& e) {
        std::cerr << std::format("Server failed to start: {}\n", e.what());
        return 1;
    }
}