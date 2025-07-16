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
    pkgs.chromium
    # Playwright dependencies
    pkgs.glib
    pkgs.nss
    pkgs.nspr
    pkgs.atk
    pkgs.at-spi2-atk
    pkgs.libdrm
    pkgs.libxkbcommon
    pkgs.gtk3
    pkgs.pango
    pkgs.cairo
    pkgs.gdk-pixbuf
    pkgs.xorg.libX11
    pkgs.xorg.libxcb
    pkgs.xorg.libXcomposite
    pkgs.xorg.libXdamage
    pkgs.xorg.libXext
    pkgs.xorg.libXfixes
    pkgs.xorg.libXrandr
    pkgs.mesa
    pkgs.libgbm
    pkgs.expat
    pkgs.libxkbcommon
    pkgs.alsa-lib
    pkgs.at-spi2-core
    pkgs.cups
    pkgs.dbus
    pkgs.fontconfig
    pkgs.freetype
  ];

  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.glibc
    pkgs.zlib
    pkgs.stdenv
    pkgs.glib
    pkgs.nss
    pkgs.nspr
    pkgs.atk
    pkgs.at-spi2-atk
    pkgs.libdrm
    pkgs.libxkbcommon
    pkgs.gtk3
    pkgs.pango
    pkgs.cairo
    pkgs.gdk-pixbuf
    pkgs.xorg.libX11
    pkgs.xorg.libxcb
    pkgs.xorg.libXcomposite
    pkgs.xorg.libXdamage
    pkgs.xorg.libXext
    pkgs.xorg.libXfixes
    pkgs.xorg.libXrandr
    pkgs.mesa
    pkgs.libgbm
    pkgs.expat
    pkgs.libxkbcommon
    pkgs.alsa-lib
    pkgs.at-spi2-core
    pkgs.cups
    pkgs.dbus
    pkgs.fontconfig
    pkgs.freetype
  ];

  env = {
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
