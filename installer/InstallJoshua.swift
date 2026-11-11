// "Install Joshua": the installer with a window, for people who would rather not use Terminal.
//
// It downloads the same install.sh as the command on the website and runs it with JOSHUA_GUI=1.
// The script then prints its questions as "@@ASK|default|question" lines: this app shows them
// with Yes and No buttons and writes the answer back. Everything else the script prints goes
// to the log; the "▸ " step titles become the line under the title.
//
// Build: swiftc -O -parse-as-library InstallJoshua.swift -o InstallJoshua
import Cocoa

let scriptURL = ProcessInfo.processInfo.environment["JOSHUA_INSTALL_URL"] ?? "https://ask.joshua.earth/install.sh"  // the variable is only for testing

@main
final class Installer: NSObject, NSApplicationDelegate {
    static func main() {
        let app = NSApplication.shared
        let delegate = Installer()
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        app.run()
    }

    var window: NSWindow!
    let heading = NSTextField(labelWithString: "Install Joshua")
    let status = NSTextField(wrappingLabelWithString: "A private AI assistant that runs on this Mac. Installing takes about 20 minutes, mostly downloads, and you'll answer a few questions along the way. Joshua goes in your Applications folder; everything else in a folder called AskJoshua in your home.")
    let logView = NSTextView()
    let logScroll = NSScrollView()
    let questionBox = NSStackView()
    let detail = NSTextField(wrappingLabelWithString: "")
    let question = NSTextField(wrappingLabelWithString: "")
    let yes = NSButton(title: "Yes", target: nil, action: nil)
    let no = NSButton(title: "No", target: nil, action: nil)
    let mainButton = NSButton(title: "Install", target: nil, action: nil)
    let spinner = NSProgressIndicator()

    var process: Process?
    var answers: FileHandle?
    var undecoded = Data()   // bytes that don't yet make a whole character
    var finished = ""        // the log, complete lines
    var line = ""            // the line being written (a progress bar rewrites it with \r)
    var context: [String] = []   // what the script said since the last question: it explains the question
    var failure = ""
    var installed = false
    let testDir = ProcessInfo.processInfo.environment["JOSHUA_TEST_SNAPSHOTS"]  // testing only: answer by itself and save pictures of the window
    var shots = 0

    func snapshot() {
        guard let dir = testDir, let view = window.contentView, let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: rep)
        shots += 1
        try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: "\(dir)/shot-\(shots).png"))
    }

    func applicationDidFinishLaunching(_ note: Notification) {
        buildMenu()
        heading.font = .systemFont(ofSize: 26, weight: .semibold)
        status.font = .systemFont(ofSize: 13)
        status.textColor = .secondaryLabelColor

        logView.isEditable = false
        logView.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        logView.textColor = .secondaryLabelColor
        logView.backgroundColor = .textBackgroundColor
        let scroll = logScroll
        scroll.isHidden = true  // shown once the installation starts
        scroll.documentView = logView
        scroll.hasVerticalScroller = true
        scroll.borderType = .lineBorder
        logView.autoresizingMask = [.width]
        scroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 220).isActive = true

        detail.font = .systemFont(ofSize: 12)
        detail.textColor = .secondaryLabelColor
        question.font = .systemFont(ofSize: 15, weight: .semibold)
        for (b, sel) in [(yes, #selector(answerYes)), (no, #selector(answerNo))] {
            b.target = self; b.action = sel; b.bezelStyle = .rounded
        }
        let buttons = NSStackView(views: [NSView(), no, yes])
        buttons.distribution = .fill
        questionBox.orientation = .vertical
        questionBox.alignment = .leading
        questionBox.spacing = 8
        for v in [detail, question, buttons] { questionBox.addArrangedSubview(v) }
        buttons.widthAnchor.constraint(equalTo: questionBox.widthAnchor).isActive = true
        questionBox.isHidden = true

        spinner.style = .spinning
        spinner.controlSize = .small
        spinner.isDisplayedWhenStopped = false
        mainButton.target = self
        mainButton.action = #selector(mainAction)
        mainButton.bezelStyle = .rounded
        mainButton.keyEquivalent = "\r"
        let bottom = NSStackView(views: [spinner, NSView(), mainButton])

        let icon = NSImageView(image: NSApp.applicationIconImage)
        icon.widthAnchor.constraint(equalToConstant: 56).isActive = true
        icon.heightAnchor.constraint(equalToConstant: 56).isActive = true
        let top = NSStackView(views: [icon, heading])
        top.spacing = 14

        let stack = NSStackView(views: [top, status, scroll, questionBox, bottom])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 14
        stack.edgeInsets = NSEdgeInsets(top: 22, left: 24, bottom: 20, right: 24)
        for v in [status, scroll, questionBox, bottom] {
            v.widthAnchor.constraint(equalTo: stack.widthAnchor, constant: -48).isActive = true
        }

        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 640, height: 560),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "Install Joshua"
        window.contentView = stack
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        if testDir != nil {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.snapshot(); self.start() }
        }
    }

    func buildMenu() {  // Cmd-Q, and copy from the log
        let menu = NSMenu()
        let appItem = NSMenuItem(); menu.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Quit Install Joshua", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        let editItem = NSMenuItem(); menu.addItem(editItem)
        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = edit
        NSApp.mainMenu = menu
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ note: Notification) {
        if let p = process, p.isRunning { p.terminate() }  // closing the window stops the installation
    }

    // ---------- the button at the bottom: Install, then Open Joshua (or Try again) ----------
    @objc func mainAction() {
        if installed { openJoshua(); return }
        start()
    }

    func start() {
        finished = ""; line = ""; context = []; failure = ""
        mainButton.isHidden = true
        logScroll.isHidden = false
        spinner.startAnimation(nil)
        heading.stringValue = "Installing Joshua"
        status.stringValue = "Downloading the installer…"

        let script = FileManager.default.temporaryDirectory.appendingPathComponent("joshua-install.sh").path
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        // fetch the current installer, then run it; "$0" is the file it was saved to
        p.arguments = ["-c", "curl -fsSL '\(scriptURL)' -o \"$0\" && exec /bin/bash \"$0\"", script]
        var env = ProcessInfo.processInfo.environment
        env["JOSHUA_GUI"] = "1"
        env["TERM"] = "dumb"
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        p.environment = env
        let out = Pipe(), inp = Pipe()
        p.standardOutput = out
        p.standardError = out
        p.standardInput = inp
        answers = inp.fileHandleForWriting
        out.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            if data.isEmpty { h.readabilityHandler = nil; return }
            DispatchQueue.main.async { self?.received(data) }
        }
        p.terminationHandler = { [weak self] p in
            DispatchQueue.main.async { self?.ended(p.terminationStatus) }
        }
        do { try p.run(); process = p } catch { ended(1) }
    }

    // ---------- what the script prints ----------
    func received(_ data: Data) {
        undecoded.append(data)
        var cut = undecoded.count  // a character can be split between two reads: decode up to the last whole one
        var text: String? = nil
        while cut > 0, cut >= undecoded.count - 3 {
            if let s = String(data: undecoded.prefix(cut), encoding: .utf8) { text = s; break }
            cut -= 1
        }
        if text == nil && undecoded.count > 8 {  // not text at all: show it as well as we can
            text = String(decoding: undecoded, as: UTF8.self); cut = undecoded.count
        }
        guard let s = text else { return }
        undecoded = Data(undecoded.suffix(from: undecoded.startIndex + cut))
        let clean = s.replacingOccurrences(of: "\u{1B}\\[[0-9;?]*[A-Za-z]", with: "", options: .regularExpression)  // terminal colours and cursor moves
        for ch in clean {
            switch ch {
            case "\r": line = ""                        // a progress bar redraws its line
            case "\n", "\r\n": lineDone(line); line = ""
            default: line.append(ch)
            }
        }
        if finished.count > 300_000 { finished = String(finished.suffix(200_000)) }
        logView.string = finished + line
        logView.scrollToEndOfDocument(nil)
    }

    func lineDone(_ l: String) {
        if l.hasPrefix("@@ASK|") {  // a question: "@@ASK|y|Install image generation?"
            let parts = l.split(separator: "|", maxSplits: 2).map(String.init)
            if parts.count == 3 { ask(parts[2], yesByDefault: parts[1] == "y") }
            return
        }
        finished += l + "\n"
        let t = l.trimmingCharacters(in: .whitespaces)
        if t.hasPrefix("▸ ") {
            status.stringValue = String(t.dropFirst(2)) + "…"
            context = []
        } else if t.hasPrefix("✗ ") {
            failure = String(t.dropFirst(2))
        } else if !t.isEmpty {
            context.append(t)
            if context.count > 8 { context.removeFirst() }
        }
    }

    func ask(_ q: String, yesByDefault: Bool) {
        // the lines just before the question explain it ("Images: Joshua can create pictures…")
        detail.stringValue = context.filter { !$0.hasPrefix("✓") && !$0.hasPrefix("!") }.suffix(5).joined(separator: "\n")
        detail.isHidden = detail.stringValue.isEmpty
        question.stringValue = q
        yes.keyEquivalent = yesByDefault ? "\r" : ""
        no.keyEquivalent = yesByDefault ? "" : "\r"
        questionBox.isHidden = false
        spinner.stopAnimation(nil)
        context = []
        NSApp.requestUserAttention(.informationalRequest)
        if testDir != nil {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.snapshot(); self.answer(yesByDefault ? "y" : "n") }
        }
    }

    @objc func answerYes() { answer("y") }
    @objc func answerNo() { answer("n") }

    func answer(_ a: String) {
        questionBox.isHidden = true
        finished += "  → " + (a == "y" ? "Yes" : "No") + "\n"
        answers?.write((a + "\n").data(using: .utf8)!)
        spinner.startAnimation(nil)
    }

    // ---------- the end ----------
    func ended(_ code: Int32) {
        spinner.stopAnimation(nil)
        questionBox.isHidden = true
        mainButton.isHidden = false
        if code == 0 {
            installed = true
            heading.stringValue = "Joshua is installed"
            status.stringValue = "You'll find it in your Applications folder and in Launchpad. Drag it to the Dock to keep it at hand."
            mainButton.title = "Open Joshua"
        } else {
            heading.stringValue = "The installation stopped"
            status.stringValue = failure.isEmpty
                ? "Something went wrong: the details are in the log above. Check your internet connection and try again; nothing is lost, it continues where it stopped."
                : failure
            mainButton.title = "Try again"
        }
        NSApp.requestUserAttention(.informationalRequest)
        if testDir != nil {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.snapshot(); NSApp.terminate(nil) }
        }
    }

    func openJoshua() {
        let places = ["/Applications/Joshua.app", NSHomeDirectory() + "/Applications/Joshua.app"]
        if let app = places.first(where: { FileManager.default.fileExists(atPath: $0) }) {
            NSWorkspace.shared.open(URL(fileURLWithPath: app))
        }
        NSApp.terminate(nil)
    }
}
