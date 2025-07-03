{
  pkgs,
  lib,
  config,
  ...
}:
let

in
{

  packages = [
    #   pkgs.pandoc
    #   gdk
    #   pkgs.tcl
    #   pkgs.tclx
    pkgs.udev
    pkgs.bashInteractive
  ];

  # env.LD_LIBRARY_PATH = lib.makeLibraryPath [
  #   pkgs.stdenv.cc.cc.lib
  #   pkgs.libGL
  #   pkgs.file
  #   pkgs.libz
  #   pkgs.gcc-unwrapped
  #   pkgs.stdenv
  # ];

  # https://devenv.sh/languages/python/
  languages.python = {
    enable = true;
    uv.enable = true;
  };

  languages.javascript = {
    enable = true;
    pnpm = {
      enable = true;
      install.enable = true;
    };
  };
  enterShell = '''';

  # git-hooks.hooks = {
  #   ruff.enable = true;
  #   rustfmt.enable = true;
  # };
  #
  # See full reference at https://devenv.sh/reference/options/
}
