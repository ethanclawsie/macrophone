cask "macrophone" do
  version "__VERSION__"
  sha256 "__SHA256__"

  url "https://github.com/ethanclawsie/macrophone/releases/download/v#{version}/Macrophone-#{version}.zip"
  name "Macrophone"
  desc "Hold-to-talk local Whisper dictation for macOS"
  homepage "https://github.com/ethanclawsie/macrophone"

  depends_on arch: :arm64
  depends_on macos: ">= :sonoma"

  app "Macrophone.app"

  # The app is ad-hoc signed (not notarized), so clear the download quarantine flag
  # to let it open without a Gatekeeper prompt.
  postflight do
    system_command "/usr/bin/xattr",
                   args: ["-dr", "com.apple.quarantine", "#{appdir}/Macrophone.app"]
  end

  uninstall quit: "com.ethanclawsie.macrophone"

  zap trash: [
    "~/Library/Application Support/Macrophone",
    "~/Library/Logs/Macrophone",
    "~/Library/Preferences/com.ethanclawsie.macrophone.plist",
  ]
end
