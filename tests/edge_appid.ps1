# Opens Joshua the way start.ps1 does (Edge --app) and prints the taskbar identity Windows sees for that window:
# its AppUserModelID (which taskbar button it groups under) and its relaunch icon (what a pinned button shows).
Add-Type @"
using System; using System.Runtime.InteropServices; using System.Text; using System.Collections.Generic;
public static class Probe {
  [StructLayout(LayoutKind.Sequential, Pack = 4)] public struct PK { public Guid f; public uint p; }
  [StructLayout(LayoutKind.Explicit)] public struct PV { [FieldOffset(0)] public ushort vt; [FieldOffset(8)] public IntPtr p; [FieldOffset(16)] public IntPtr pad; }
  [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IPS { int GetCount(out uint c); int GetAt(uint i, out PK k); int GetValue(ref PK k, out PV v); int SetValue(ref PK k, ref PV v); int Commit(); }
  [DllImport("shell32.dll")] static extern int SHGetPropertyStoreForWindow(IntPtr h, ref Guid iid, [MarshalAs(UnmanagedType.Interface)] out IPS ps);
  public delegate bool EW(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] static extern bool EnumWindows(EW cb, IntPtr l);
  [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] static extern int GetWindowThreadProcessId(IntPtr h, out int pid);
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  static string Get(IPS ps, uint id) {
    var k = new PK { f = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), p = id }; PV v;
    if (ps.GetValue(ref k, out v) != 0 || v.vt != 31) return "";
    return Marshal.PtrToStringUni(v.p);
  }
  public static List<string> Windows(int[] pids) {
    var r = new List<string>(); var set = new HashSet<int>(pids);
    EnumWindows((h, l) => {
      int pid; GetWindowThreadProcessId(h, out pid);
      if (!set.Contains(pid) || !IsWindowVisible(h)) return true;
      var t = new StringBuilder(300); GetWindowText(h, t, 300);
      var iid = new Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"); IPS ps;
      string id = "", cmd = "", icon = "", name = "";
      if (SHGetPropertyStoreForWindow(h, ref iid, out ps) == 0) { id = Get(ps, 5); cmd = Get(ps, 2); icon = Get(ps, 3); name = Get(ps, 4); }
      r.Add("title=" + t + " | ID=" + id + " | RelaunchCommand=" + cmd + " | RelaunchIcon=" + icon + " | RelaunchName=" + name);
      return true; }, IntPtr.Zero);
    return r;
  }
}
"@
$edge = @("${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
Start-Process -FilePath $edge -ArgumentList "--app=http://127.0.0.1:7860", "--no-first-run"
Start-Sleep 15
$pids = @(Get-Process msedge -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
$out = [Probe]::Windows($pids) -join "%0A"
Write-Host "::notice title=Edge app window::$out"
