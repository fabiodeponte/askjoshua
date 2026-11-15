// "Install Joshua" for Windows: the installer with a window, for people who would rather not use PowerShell.
//
// It downloads the same install.ps1 as the command on the website and runs it with JOSHUA_GUI=1.
// The script then prints its questions as "@@ASK|default|question" lines: this program shows them
// with Yes and No buttons and writes the answer back. Everything else the script prints goes
// to the log; the "> " step titles become the line under the title, and move the progress bar.
//
// Build (on Linux with Mono, or on Windows with the csc that comes with .NET Framework):
//   mcs -target:winexe -win32icon:joshua.ico -r:System.Windows.Forms.dll -r:System.Drawing.dll -out:"Install Joshua.exe" InstallJoshua.cs
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows.Forms;

class Installer : Form
{
    static readonly string ScriptUrl = Environment.GetEnvironmentVariable("JOSHUA_INSTALL_URL") ?? "https://ask.joshua.earth/install.ps1";  // the variable is only for testing
    static readonly string TestDir = Environment.GetEnvironmentVariable("JOSHUA_TEST_SNAPSHOTS");  // testing only: answer by itself and save pictures of the window
    const int Steps = 8;  // the "> " titles install.ps1 prints on a full run

    static readonly Color Cream = Color.FromArgb(0xF3, 0xEF, 0xDC), Ink = Color.FromArgb(0x2A, 0x26, 0x22),
                          Dim = Color.FromArgb(0x75, 0x60, 0x4F), Green = Color.FromArgb(0x3F, 0x8A, 0x49);

    readonly Label heading = new Label(), status = new Label(), detail = new Label(), question = new Label();
    readonly TextBox log = new TextBox();
    readonly ProgressBar bar = new ProgressBar();
    readonly Panel questionBox = new Panel();
    readonly Button yes = new Button(), no = new Button(), main = new Button();

    Process process;
    readonly StringBuilder finished = new StringBuilder();
    readonly System.Collections.Generic.List<string> context = new System.Collections.Generic.List<string>();
    string failure = "";
    bool installed;
    int step, shots;

    static readonly string LogFile = Path.Combine(Path.GetTempPath(), "Install Joshua.log");  // what happened, if the window closes by surprise

    static void Note(string text)
    {
        try { File.AppendAllText(LogFile, DateTime.Now.ToString("HH:mm:ss ") + text + Environment.NewLine); } catch { }
    }

    static void Crash(object e)
    {
        Note("ERROR " + e);
        MessageBox.Show("Install Joshua stopped because of an error:\n\n" + e + "\n\nThe details are in " + LogFile,
                        "Install Joshua", MessageBoxButtons.OK, MessageBoxIcon.Error);
    }

    [STAThread]
    static void Main()
    {
        Note("started, Windows " + Environment.OSVersion + ", .NET " + Environment.Version);
        Application.SetUnhandledExceptionMode(UnhandledExceptionMode.CatchException);
        Application.ThreadException += (s, e) => Crash(e.Exception);
        AppDomain.CurrentDomain.UnhandledException += (s, e) => Crash(e.ExceptionObject);
        try
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Installer());
        }
        catch (Exception e) { Crash(e); }
        Note("closed");
    }

    Installer()
    {
        Text = "Install Joshua";
        try { Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath); }  // fails when the program sits on a network folder
        catch { Icon = SystemIcons.Application; }
        BackColor = Cream; ForeColor = Ink;
        Font = new Font("Segoe UI", 10f);
        ClientSize = new Size(680, 600);
        MinimumSize = new Size(560, 480);
        StartPosition = FormStartPosition.CenterScreen;
        Padding = new Padding(24, 20, 24, 18);

        var logo = new PictureBox { Image = Icon.ToBitmap(), SizeMode = PictureBoxSizeMode.Zoom, Size = new Size(56, 56), Location = new Point(24, 20) };
        heading.Text = "Install Joshua";
        heading.Font = new Font("Segoe UI Semibold", 20f);
        heading.AutoSize = true;
        heading.Location = new Point(92, 30);

        status.Text = "A private AI assistant that runs on this computer. Installing takes about 20 minutes, mostly downloads, " +
                      "and you'll answer a few questions along the way. Joshua goes in the Start menu; everything else in a folder " +
                      "called AskJoshua in your user folder. No administrator rights are needed.";
        status.ForeColor = Dim;
        status.Location = new Point(24, 92);
        status.Size = new Size(632, 64);
        status.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;

        bar.Location = new Point(24, 160); bar.Size = new Size(632, 8); bar.Maximum = Steps; bar.Visible = false;
        bar.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;

        log.Multiline = true; log.ReadOnly = true; log.ScrollBars = ScrollBars.Vertical; log.WordWrap = true;
        log.Font = new Font("Consolas", 9f); log.BackColor = Color.White; log.ForeColor = Dim;
        log.BorderStyle = BorderStyle.FixedSingle;
        log.Location = new Point(24, 178); log.Size = new Size(632, 250);
        log.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
        log.Visible = false;

        questionBox.Location = new Point(24, 438); questionBox.Size = new Size(632, 110);
        questionBox.Anchor = AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
        questionBox.Visible = false;
        detail.ForeColor = Dim; detail.Font = new Font("Segoe UI", 9f);
        detail.Location = new Point(0, 0); detail.Size = new Size(632, 52); detail.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
        question.Font = new Font("Segoe UI Semibold", 11f);
        question.Location = new Point(0, 56); question.Size = new Size(440, 48); question.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
        Style(no, "No", false); no.Location = new Point(452, 64); no.Anchor = AnchorStyles.Top | AnchorStyles.Right;
        Style(yes, "Yes", true); yes.Location = new Point(546, 64); yes.Anchor = AnchorStyles.Top | AnchorStyles.Right;
        yes.Click += delegate { Answer("y"); };
        no.Click += delegate { Answer("n"); };
        questionBox.Controls.AddRange(new Control[] { detail, question, no, yes });

        Style(main, "Install", true);
        main.Size = new Size(140, 36);
        main.Location = new Point(516, 552);
        main.Anchor = AnchorStyles.Bottom | AnchorStyles.Right;
        main.Click += delegate { if (installed) OpenJoshua(); else Start(); };
        AcceptButton = main;

        Controls.AddRange(new Control[] { logo, heading, status, bar, log, questionBox, main });
        FormClosing += delegate { try { if (process != null && !process.HasExited) process.Kill(); } catch { } };  // closing the window stops the installation
        if (TestDir != null) Shown += delegate { Later(1000, delegate { Snapshot(); Start(); }); };
    }

    static void Style(Button b, string text, bool primary)
    {
        b.Text = text; b.Size = new Size(86, 34); b.FlatStyle = FlatStyle.Flat; b.Cursor = Cursors.Hand;
        b.FlatAppearance.BorderColor = Green;
        b.BackColor = primary ? Green : Cream; b.ForeColor = primary ? Color.White : Green;
        b.Font = new Font("Segoe UI Semibold", 10f);
    }

    void Later(int ms, Action a) { var t = new Timer { Interval = ms }; t.Tick += delegate { t.Stop(); t.Dispose(); a(); }; t.Start(); }

    void Snapshot()
    {
        if (TestDir == null) return;
        using (var b = new Bitmap(ClientSize.Width, ClientSize.Height))
        {
            DrawToBitmap(b, new Rectangle(0, 0, ClientSize.Width, ClientSize.Height));
            b.Save(Path.Combine(TestDir, "shot-" + (++shots) + ".png"));
        }
    }

    // ---------- the installation: fetch install.ps1, run it, read what it prints ----------
    void Start()
    {
        finished.Clear(); context.Clear(); failure = ""; step = 0;
        main.Visible = false; log.Visible = true; bar.Visible = true; bar.Value = 0;
        heading.Text = "Installing Joshua";
        status.Text = "Downloading the installer…";
        string script = Path.Combine(Path.GetTempPath(), "joshua-install.ps1");
        try
        {
            if (File.Exists(ScriptUrl)) File.Copy(ScriptUrl, script, true);
            else
            {
                ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
                using (var w = new WebClient()) w.DownloadFile(ScriptUrl, script);
            }
        }
        catch (Exception e) { failure = "Couldn't download the installer: " + e.Message + ". Check your internet connection and try again."; Ended(1); return; }

        var info = new ProcessStartInfo("powershell.exe", "-NoProfile -ExecutionPolicy Bypass -File \"" + script + "\"")
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true, RedirectStandardInput = true,
            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8
        };
        info.EnvironmentVariables["JOSHUA_GUI"] = "1";
        info.EnvironmentVariables.Remove("JOSHUA_YES");
        info.EnvironmentVariables.Remove("PSModulePath");  // started from PowerShell 7, this would point Windows PowerShell at modules it can't load  // in this window the questions are answered with the buttons (or, in a test, by the test)
        process = new Process { StartInfo = info, EnableRaisingEvents = true };
        process.OutputDataReceived += (s, e) => { if (e.Data != null) BeginInvoke((Action)(() => Line(e.Data))); };
        process.ErrorDataReceived += (s, e) => { if (e.Data != null) BeginInvoke((Action)(() => Line(e.Data))); };
        process.Exited += delegate { BeginInvoke((Action)(() => Ended(process.ExitCode))); };
        Note("running " + script);
        try { process.Start(); process.BeginOutputReadLine(); process.BeginErrorReadLine(); }
        catch (Exception e) { failure = "Couldn't start PowerShell: " + e.Message; Ended(1); }
    }

    void Line(string raw)
    {
        Note(raw);
        string l = Regex.Replace(raw, "\u001B\\[[0-9;?]*[A-Za-z]", "");  // terminal colours and cursor moves
        if (l.Contains("\r")) l = l.Substring(l.LastIndexOf('\r') + 1);  // a progress bar redraws its line: keep the last state
        if (l.StartsWith("@@ASK|"))  // a question: "@@ASK|y|Install the deep-thinking version?"
        {
            var parts = l.Split(new[] { '|' }, 3);
            if (parts.Length == 3) Ask(parts[2], parts[1] == "y");
            return;
        }
        if (Regex.IsMatch(l, @"(pulling|downloading|verifying)\b.*\d+%", RegexOptions.IgnoreCase))  // a download's progress: shown, not kept
        {
            status.Text = l.Trim();
            return;
        }
        finished.AppendLine(l);
        if (finished.Length > 300000) finished.Remove(0, 100000);
        log.Text = finished.ToString();
        log.SelectionStart = log.TextLength; log.ScrollToCaret();
        string t = l.Trim();
        if (t.StartsWith("> "))
        {
            status.Text = t.Substring(2) + "…";
            bar.Value = Math.Min(Steps, ++step);
            context.Clear();
        }
        else if (t.StartsWith("X ")) failure = t.Substring(2);
        else if (t.Length > 0 && !t.StartsWith("PS>") && !t.StartsWith("***"))
        {
            context.Add(t);
            if (context.Count > 8) context.RemoveAt(0);
        }
    }

    void Ask(string q, bool yesByDefault)
    {
        // the lines just before the question explain it ("Deep thinking: a second version of the model…")
        var why = context.FindAll(x => !x.StartsWith("OK") && !x.StartsWith("!"));
        detail.Text = string.Join("\n", why.GetRange(Math.Max(0, why.Count - 3), Math.Min(3, why.Count)));
        question.Text = q;
        AcceptButton = yesByDefault ? yes : no;
        questionBox.Visible = true;
        log.Height = questionBox.Top - log.Top - 10;
        context.Clear();
        Activate();
        if (TestDir != null) Later(800, delegate { Snapshot(); Answer(yesByDefault ? "y" : "n"); });
    }

    void Answer(string a)
    {
        questionBox.Visible = false;
        log.Height = main.Top - log.Top - 16;
        finished.AppendLine("  -> " + (a == "y" ? "Yes" : "No"));
        log.Text = finished.ToString();
        log.SelectionStart = log.TextLength; log.ScrollToCaret();
        try { process.StandardInput.WriteLine(a); process.StandardInput.Flush(); } catch { }
    }

    // ---------- the end ----------
    void Ended(int code)
    {
        Note("installer ended with code " + code);
        questionBox.Visible = false;
        main.Visible = true;
        if (code == 0)
        {
            installed = true;
            bar.Value = Steps;
            heading.Text = "Joshua is installed";
            status.Text = "You'll find it in the Start menu: type Joshua. Right-click it there to pin it to the taskbar.";
            main.Text = "Open Joshua";
        }
        else
        {
            heading.Text = "The installation stopped";
            status.Text = failure.Length > 0 ? failure
                : "Something went wrong: the details are in the log above. Check your internet connection and try again; nothing is lost, it continues where it stopped.";
            main.Text = "Try again";
        }
        Activate();
        if (TestDir != null) Later(1000, delegate { Snapshot(); Environment.Exit(code); });
    }

    void OpenJoshua()
    {
        string start = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "AskJoshua", "start.ps1");
        if (File.Exists(start))
            Process.Start(new ProcessStartInfo("powershell.exe", "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \"" + start + "\"") { UseShellExecute = false, CreateNoWindow = true });
        Close();
    }
}
