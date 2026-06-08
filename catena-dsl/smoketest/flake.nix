{
  description = "Catena GPU codegen smoketests";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      rocmEnv =
        pkgs:
        pkgs.symlinkJoin {
          name = "rocm-path";
          paths = [
            pkgs.rocmPackages.clang
            pkgs.rocmPackages.clr
            pkgs.rocmPackages.hip-common
            pkgs.rocmPackages.hipcc
            pkgs.rocmPackages.rocm-core
            pkgs.rocmPackages.rocm-device-libs
            pkgs.rocmPackages.rocm-runtime
          ];
        };
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          rocmPath = rocmEnv pkgs;
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.gnumake
              pkgs.rocmPackages.clang
              pkgs.rocmPackages.hipcc
              pkgs.rocmPackages.rocminfo
            ];
            shellHook = ''
              export ROCM_PATH=${rocmPath}
              export HIP_PATH=${rocmPath}
              export HIP_CLANG_PATH=${pkgs.rocmPackages.clang}/bin
              export DEVICE_LIB_PATH=${pkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode
              export HIP_FLAGS="--rocm-path=${rocmPath} --rocm-device-lib-path=${pkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode"
            '';
          };
        }
      );
    };
}
