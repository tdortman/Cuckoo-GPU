{ inputs, lib, stdenvNoCC, ... }:

stdenvNoCC.mkDerivation {
  pname = "cuckoogpu";
  version = "0.1.0";

  src = inputs.self;

  dontBuild = true;

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/include"
    cp -r "$src/include/." "$out/include/"

    runHook postInstall
  '';

  meta = {
    description = "GPU-Accelerated Cuckoo Filter";
    license = lib.licenses.boost;
    platforms = [
      "x86_64-linux"
      "aarch64-linux"
    ];
  };
}
