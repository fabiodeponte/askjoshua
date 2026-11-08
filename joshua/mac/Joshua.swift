// Joshua.app: a window with Joshua inside, nothing else. It starts Joshua if it isn't running yet.
// Build (on a Mac):  swiftc -O -target arm64-apple-macos13 Joshua.swift -o Joshua
import Cocoa
import WebKit

let URL_JOSHUA = URL(string: "http://127.0.0.1:7860")!
let START = NSHomeDirectory() + "/AskJoshua/start.sh"

class App: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate, WKDownloadDelegate {
    var window: NSWindow!
    var web: WKWebView!

    func applicationDidFinishLaunching(_ n: Notification) {
        buildMenu()
        web = WKWebView(frame: .zero, configuration: WKWebViewConfiguration())
        web.navigationDelegate = self
        web.uiDelegate = self
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 820),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "Joshua"
        window.contentView = web
        window.center()
        window.setFrameAutosaveName("Joshua")  // remembers size and position
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        web.loadHTMLString("<body style='background:#dfdbc3;color:#3b2322;font:20px Menlo;display:grid;place-items:center;height:90vh'>Starting Joshua…</body>", baseURL: nil)
        waitForJoshua(startedIt: false)
    }

    // Ask Joshua's address every second; start it once if nobody answers, then show it.
    func waitForJoshua(startedIt: Bool) {
        URLSession.shared.dataTask(with: URL_JOSHUA) { _, resp, _ in
            DispatchQueue.main.async {
                if resp != nil { self.web.load(URLRequest(url: URL_JOSHUA)); return }
                if !startedIt { self.startJoshua() }
                DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.waitForJoshua(startedIt: true) }
            }
        }.resume()
    }

    func startJoshua() {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = ["-c", "JOSHUA_NO_OPEN=1 nohup '\(START)' >/dev/null 2>&1 &"]
        try? p.run()
    }

    // Links to other sites open in the normal browser.
    func webView(_ w: WKWebView, decidePolicyFor a: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if let u = a.request.url, u.host != nil, u.host != "127.0.0.1", u.host != "localhost" {
            NSWorkspace.shared.open(u); decisionHandler(.cancel); return
        }
        decisionHandler(a.shouldPerformDownload ? .download : .allow)
    }
    func webView(_ w: WKWebView, createWebViewWith c: WKWebViewConfiguration, for a: WKNavigationAction, windowFeatures f: WKWindowFeatures) -> WKWebView? {
        if let u = a.request.url { NSWorkspace.shared.open(u) }  // target=_blank
        return nil
    }

    // Files Joshua offers for download go to the Downloads folder.
    func webView(_ w: WKWebView, decidePolicyFor r: WKNavigationResponse, decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        decisionHandler(r.canShowMIMEType ? .allow : .download)
    }
    func webView(_ w: WKWebView, navigationAction a: WKNavigationAction, didBecome d: WKDownload) { d.delegate = self }
    func webView(_ w: WKWebView, navigationResponse r: WKNavigationResponse, didBecome d: WKDownload) { d.delegate = self }
    func download(_ d: WKDownload, decideDestinationUsing r: URLResponse, suggestedFilename name: String, completionHandler: @escaping (URL?) -> Void) {
        let dir = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
        var dest = dir.appendingPathComponent(name), i = 1
        while FileManager.default.fileExists(atPath: dest.path) {
            dest = dir.appendingPathComponent("\((name as NSString).deletingPathExtension) \(i).\((name as NSString).pathExtension)"); i += 1
        }
        completionHandler(dest)
    }
    func downloadDidFinish(_ d: WKDownload) { NSSound(named: "Glass")?.play() }

    // Choosing files to upload.
    func webView(_ w: WKWebView, runOpenPanelWith p: WKOpenPanelParameters, initiatedByFrame f: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = p.allowsMultipleSelection
        panel.canChooseDirectories = p.allowsDirectories
        panel.beginSheetModal(for: window) { completionHandler($0 == .OK ? panel.urls : nil) }
    }

    // The page's own alert and confirm boxes.
    func webView(_ w: WKWebView, runJavaScriptAlertPanelWithMessage m: String, initiatedByFrame f: WKFrameInfo, completionHandler: @escaping () -> Void) {
        let a = NSAlert(); a.messageText = m; a.runModal(); completionHandler()
    }
    func webView(_ w: WKWebView, runJavaScriptConfirmPanelWithMessage m: String, initiatedByFrame f: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let a = NSAlert(); a.messageText = m; a.addButton(withTitle: "OK"); a.addButton(withTitle: "Cancel")
        completionHandler(a.runModal() == .alertFirstButtonReturn)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }
    @objc func reload() { web.load(URLRequest(url: URL_JOSHUA)) }

    // Menus, so that Cmd-C, Cmd-V, Cmd-R, Cmd-Q and the rest work.
    func buildMenu() {
        let main = NSMenu()
        func menu(_ title: String, _ items: [(String, Selector?, String)]) {
            let m = NSMenu(title: title)
            for (t, s, k) in items { m.addItem(t == "-" ? .separator() : NSMenuItem(title: t, action: s, keyEquivalent: k)) }
            let holder = NSMenuItem(); holder.submenu = m; main.addItem(holder)
        }
        menu("Joshua", [("Hide Joshua", #selector(NSApplication.hide(_:)), "h"), ("-", nil, ""),
                        ("Quit Joshua", #selector(NSApplication.terminate(_:)), "q")])
        menu("Edit", [("Undo", Selector(("undo:")), "z"), ("Redo", Selector(("redo:")), "Z"), ("-", nil, ""),
                      ("Cut", #selector(NSText.cut(_:)), "x"), ("Copy", #selector(NSText.copy(_:)), "c"),
                      ("Paste", #selector(NSText.paste(_:)), "v"), ("Select All", #selector(NSText.selectAll(_:)), "a")])
        menu("View", [("Reload", #selector(reload), "r")])
        menu("Window", [("Minimize", #selector(NSWindow.miniaturize(_:)), "m"), ("Close", #selector(NSWindow.performClose(_:)), "w")])
        NSApp.mainMenu = main
    }
}

let app = NSApplication.shared
let delegate = App()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
