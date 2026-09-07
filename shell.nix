{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (ps: with ps; [
      # Python packages
      kagglehub 
      pandas
      jupyter
      numpy
      scikit-learn
      matplotlib
      seaborn
      tqdm
      transformers
      torch
      torchvision
      torchaudio
    ]))
    
    # System dependencies needed for NumPy and PyTorch
    pkgs.stdenv.cc.cc.lib  # For libstdc++
    pkgs.glibc            # Standard C library
    pkgs.zlib             # Compression library
    pkgs.gcc              # GCC compiler
    pkgs.gnumake          # Make utility
  ];

  # Set environment variables
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.glibc}/lib:$LD_LIBRARY_PATH
  '';
}