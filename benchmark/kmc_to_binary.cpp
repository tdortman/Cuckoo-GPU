/*
  KMC database to binary converter for benchmarks.

  Reads a KMC database directly using the KMC API and outputs our binary format:
    - uint64_t: Number of k-mers (N)
    - N x uint64_t: 2-bit encoded k-mers
*/

#include <kmc_api/kmc_file.h>
#include <CLI/CLI.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

constexpr uint64_t encodeNucleotide(char c) {
    switch (c) {
        case 'A':
        case 'a':
            return 0b00;
        case 'C':
        case 'c':
            return 0b01;
        case 'G':
        case 'g':
            return 0b10;
        case 'T':
        case 't':
            return 0b11;
        default:
            return 0xFF;  // Invalid
    }
}

uint64_t encodeKmer(const std::string& kmer) {
    uint64_t encoded = 0;
    for (const char c : kmer) {
        uint64_t val = encodeNucleotide(c);
        if (val == 0xFF) {
            return UINT64_MAX;  // Invalid k-mer
        }
        encoded = (encoded << 2) | val;
    }
    return encoded;
}

int main(int argc, char* argv[]) {
    CLI::App app{"Convert KMC database to binary format for CUDA benchmarks"};

    std::string inputPrefix;
    std::string outputFile;

    app.add_option("input", inputPrefix, "KMC database prefix (without .kmc_pre/.kmc_suf)")
        ->required();
    app.add_option("output", outputFile, "Output binary file path")->required();

    CLI11_PARSE(app, argc, argv);

    // Open KMC database
    CKMCFile kmcDb;
    if (!kmcDb.OpenForListing(inputPrefix)) {
        std::cerr << "Error: Cannot open KMC database: " << inputPrefix << "\n";
        return 1;
    }

    // Get database info
    uint32 kmerLength, mode, counterSize, lutPrefixLength, signatureLen, minCount;
    uint64 maxCount, totalKmers;
    kmcDb.Info(
        kmerLength, mode, counterSize, lutPrefixLength, signatureLen, minCount, maxCount, totalKmers
    );

    std::cout << "KMC database info:\n"
              << "  K-mer length: " << kmerLength << "\n"
              << "  Total k-mers: " << totalKmers << "\n";

    if (kmerLength > 32) {
        std::cerr << "Error: K-mer length " << kmerLength << " exceeds maximum of 32\n";
        kmcDb.Close();
        return 1;
    }

    // Open output file
    std::ofstream out(outputFile, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Cannot open output file: " << outputFile << "\n";
        kmcDb.Close();
        return 1;
    }

    // Write placeholder for count
    uint64_t count = 0;
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // Read and convert k-mers
    CKmerAPI kmerObj(kmerLength);
    std::string kmerStr;
    uint32 counter;
    uint64_t validCount = 0;
    uint64_t invalidCount = 0;

    while (kmcDb.ReadNextKmer(kmerObj, counter)) {
        kmerObj.to_string(kmerStr);
        uint64_t encoded = encodeKmer(kmerStr);

        if (encoded == UINT64_MAX) {
            invalidCount++;
            continue;
        }

        out.write(reinterpret_cast<const char*>(&encoded), sizeof(encoded));
        validCount++;

        // Progress indicator
        if (validCount % 100'000'000 == 0) {
            std::cout << "  Processed " << validCount << " k-mers...\n";
        }
    }

    // Seek back and write actual count
    out.seekp(0);
    out.write(reinterpret_cast<const char*>(&validCount), sizeof(validCount));
    out.close();

    kmcDb.Close();

    std::cout << "Wrote " << validCount << " k-mers to " << outputFile << "\n";
    if (invalidCount > 0) {
        std::cout << "Skipped " << invalidCount << " invalid k-mers\n";
    }

    return 0;
}
