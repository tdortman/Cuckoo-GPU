{
  description = "GPU-Accelerated Cuckoo Filter";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    snowfall-lib = {
      url = "github:snowfallorg/lib";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    inputs.snowfall-lib.mkFlake {
      inherit inputs;
      src = ./.;

      snowfall = {
        root = ./nix;
        namespace = "cuckoogpu";
      };

      channels-config = {
        allowUnfree = true;
      };

      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      alias = {
        packages.default = "cuckoogpu";
      };
    };
}
