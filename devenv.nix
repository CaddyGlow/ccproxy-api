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
    pkgs.duckdb
    pkgs.stdenv.cc.cc.lib
    pkgs.glibc
    pkgs.zlib
    pkgs.stdenv
    pkgs.playwright-driver.browsers
  ];

  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.stdenv
  ];

  env = {
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD = 1;
    PLAYWRIGHT_BROWSERS_PATH = pkgs.playwright-driver.browsers;
    PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS = true;
  };

  # https://devenv.sh/languages/python/
  languages.python = {
    enable = true;
    uv.enable = true;
  };

  languages.javascript = {
    enable = true;
    bun = {
      enable = true;
    };
    pnpm = {
      enable = true;
      install.enable = false;
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
