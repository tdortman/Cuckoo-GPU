#include <thrust/host_vector.h>
#include <chrono>
#include <CLI/CLI.hpp>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include "CuckooFilterIPC.cuh"

using Config = CuckooConfig<uint64_t, 16, 500, 256, 16>;

void runServer(const std::string& name, size_t capacity, bool forceShutdown = false) {
    std::cout << std::format("Starting server with capacity: {}\n", capacity);
    if (forceShutdown) {
        std::cout << "Force shutdown mode enabled (pending requests will be cancelled)\n";
    }

    try {
        CuckooFilterIPCServer<Config> server(name, capacity);
        server.start();

        std::cout << "Server running. Press Enter to stop...\n";
        std::cin.get();

        server.stop(forceShutdown);
        std::cout << "Server stopped.\n";

        auto filter = server.getFilter();
        std::cout << std::format("Final load factor: {}\n", filter->loadFactor());
        std::cout << std::format("Occupied slots: {}\n", filter->occupiedSlots());
        std::cout << std::format("Capacity: {}\n", filter->capacity());
        std::cout << std::format("Size in bytes: {}\n", filter->sizeInBytes());

    } catch (const std::exception& e) {
        std::cerr << std::format("Server error: {}\n", e.what());
    }
}

void runClient(const std::string& name, int clientId, size_t numKeys) {
    std::cout << std::format("Client {} starting...\n", clientId);

    try {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        CuckooFilterIPCClient<Config> client(name);

        thrust::host_vector<uint64_t> h_keys(numKeys);
        std::random_device rd;
        std::mt19937_64 gen(rd() + clientId);
        std::uniform_int_distribution<uint64_t> dis(1, UINT32_MAX);
        for (size_t i = 0; i < numKeys; i++) {
            h_keys[i] = dis(gen);
        }

        thrust::device_vector<uint64_t> d_keys = h_keys;
        thrust::device_vector<bool> d_results(numKeys);

        auto start = std::chrono::high_resolution_clock::now();

        size_t occupiedAfterInsert = client.insertMany(d_keys);

        std::cout << std::format(
            "Client {} inserted {} keys (filter now has {} occupied slots)\n",
            clientId,
            numKeys,
            occupiedAfterInsert
        );

        client.containsMany(d_keys, d_results);

        thrust::host_vector<bool> h_results = d_results;
        size_t found = 0;
        for (bool result : h_results) {
            if (result) {
                found++;
            }
        }
        std::cout << std::format("Client {} found {}/{} keys\n", clientId, found, numKeys);

        size_t deleteCount = numKeys / 2;
        thrust::device_vector<uint64_t> d_keysToDelete(
            d_keys.begin(), d_keys.begin() + deleteCount
        );
        size_t occupiedAfterDelete = client.deleteMany(d_keysToDelete);

        std::cout << std::format(
            "Client {} deleted {} keys (filter now has {} occupied slots)\n",
            clientId,
            deleteCount,
            occupiedAfterDelete
        );

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << std::format("Client {} completed in {}ms\n", clientId, duration.count());

    } catch (const std::exception& e) {
        std::cerr << std::format("Client {} error: {}\n", clientId, e.what());
    }
}

int main(int argc, char** argv) {
    CLI::App app{"Cuckoo Filter IPC Server/Client"};
    app.require_subcommand(1);

    auto* serverCmd = app.add_subcommand("server", "Run the IPC server");

    std::string serverName;
    int serverCapacityExp = 25;
    bool forceShutdown = false;

    serverCmd->add_option("name", serverName, "Server name for IPC")->required();

    serverCmd->add_option("-c,--capacity", serverCapacityExp, "Capacity exponent (capacity = 2^x)")
        ->default_val(25)
        ->check(CLI::PositiveNumber);

    serverCmd->add_flag("-f,--force", forceShutdown, "Force shutdown (cancel pending requests)");

    auto* clientCmd = app.add_subcommand("client", "Run IPC client(s)");

    std::string clientType = "normal";
    int numClients = 1;
    int clientCapacityExp = 25;
    double targetLoadFactor = 0.95;

    clientCmd->add_option("name", serverName, "Server name to connect to")->required();

    clientCmd->add_option("-n,--num-clients", numClients, "Number of concurrent clients")
        ->default_val(1)
        ->check(CLI::PositiveNumber);

    clientCmd
        ->add_option(
            "-c,--capacity",
            clientCapacityExp,
            "Capacity exponent (must match server, keys = 2^x * loadFactor)"
        )
        ->default_val(25)
        ->check(CLI::PositiveNumber);

    clientCmd->add_option("-l,--load-factor", targetLoadFactor, "Target load factor")
        ->check(CLI::Range(0.0, 1.0))
        ->default_val(0.95);

    CLI11_PARSE(app, argc, argv);

    if (*serverCmd) {
        size_t capacity = 1ULL << serverCapacityExp;
        runServer(serverName, capacity, forceShutdown);
    } else if (*clientCmd) {
        size_t numKeys = (1ULL << clientCapacityExp) * targetLoadFactor;

        std::vector<std::thread> clientThreads;
        for (int i = 0; i < numClients; i++) {
            clientThreads.emplace_back(runClient, serverName, i, numKeys);
        }

        for (auto& thread : clientThreads) {
            thread.join();
        }
    }

    return 0;
}