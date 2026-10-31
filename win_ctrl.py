# -*- coding: utf-8 -*-
"""Windows 控件精确操作工具（基于 uiautomation）

背景（2026-10-31 实测）：
  本机物理屏幕 2560x1440，逻辑像素 2048x1152（DPI 125%），
  截图 1366x768（比例 1.874）。按截图坐标点击会有累积偏差。
  uiautomation 直接返回物理像素坐标，且能按控件名定位，
  完全绕开坐标换算问题。

用法：
  # 列出窗口内所有可点击控件
  python win_ctrl.py list "窗口标题片段"

  # 按控件名点击
  python win_ctrl.py click "窗口标题片段" "按钮名"

  # 按控件名输入文本
  python win_ctrl.py type "窗口标题片段" "输入框名" "要输入的文字"

  # 按控件名读文本
  python win_ctrl.py read "窗口标题片段" "控件名"

  # 列出所有顶层窗口
  python win_ctrl.py windows

  # 屏幕信息
  python win_ctrl.py screen
"""
import sys
import time

try:
    import uiautomation as auto
except ImportError:
    print("需要 uiautomation: python -m pip install uiautomation")
    sys.exit(1)


def w(r):
    return r.right - r.left


def h(r):
    return r.bottom - r.top


def find_window(title_part):
    """按标题片段找顶层窗口"""
    root = auto.GetRootControl()
    for win in root.GetChildren():
        try:
            if win.ControlTypeName == "WindowControl" and title_part in (win.Name or ""):
                return win
        except Exception:
            continue
    return None


def walk(ctrl, depth=0, max_depth=14, results=None, max_items=400):
    """递归遍历控件树"""
    if results is None:
        results = []
    if depth > max_depth or len(results) >= max_items:
        return results
    try:
        for c in ctrl.GetChildren():
            try:
                results.append((c, depth))
                if len(results) >= max_items:
                    return results
            except Exception:
                pass
            walk(c, depth + 1, max_depth, results, max_items)
    except Exception:
        pass
    return results


def cmd_screen():
    root = auto.GetRootControl()
    r = root.BoundingRectangle
    print("物理屏幕: %dx%d  at (%d,%d)" % (w(r), h(r), r.left, r.top))
    print("说明: uiautomation 用物理像素坐标")
    print("      截图通常是逻辑像素，二者有 DPI 比例差")


def cmd_windows():
    root = auto.GetRootControl()
    print("%-8s %-60s" % ("类型", "标题"))
    print("-" * 70)
    for win in root.GetChildren():
        try:
            t = win.ControlTypeName
            if t in ("WindowControl", "PaneControl"):
                r = win.BoundingRectangle
                if w(r) > 100:
                    print("%-8s %-50s (%dx%d)" % (t.replace("Control", ""), (win.Name or "")[:48], w(r), h(r)))
        except Exception:
            continue


def cmd_list(title_part):
    win = find_window(title_part)
    if not win:
        print("找不到窗口: %s" % title_part)
        return
    print("窗口: %s" % win.Name[:70])
    r = win.BoundingRectangle
    print("位置: (%d,%d) 大小 %dx%d" % (r.left, r.top, w(r), h(r)))
    print()
    items = walk(win)
    print("共 %d 个控件" % len(items))
    print()
    print("%-16s %-40s %s" % ("类型", "名称", "位置"))
    print("-" * 76)
    n = 0
    for c, d in items:
        try:
            ct = c.ControlTypeName
            if ct in ("ButtonControl", "EditControl", "HyperlinkControl",
                      "CheckBoxControl", "RadioButtonControl", "ComboBoxControl",
                      "ListItemControl", "MenuItemControl", "TabItemControl"):
                nm = (c.Name or "")[:38]
                br = c.BoundingRectangle
                if w(br) > 0:
                    print("%-16s %-40s @(%d,%d) %dx%d" % (
                        ct.replace("Control", ""), nm, br.left, br.top, w(br), h(br)))
                    n += 1
                    if n >= 60:
                        break
        except Exception:
            continue


def cmd_click(title_part, ctrl_name):
    win = find_window(title_part)
    if not win:
        print("找不到窗口: %s" % title_part)
        return
    items = walk(win)
    for c, d in items:
        try:
            if (c.Name or "").strip() == ctrl_name.strip():
                br = c.BoundingRectangle
                print("找到: [%s] %s @(%d,%d)" % (c.ControlTypeName, c.Name, br.left, br.top))
                c.Click(simulateMove=False)
                print("已点击")
                return
        except Exception:
            continue
    print("找不到控件: %s" % ctrl_name)
    print("提示: 用 list 看可用的控件名")


def cmd_type(title_part, ctrl_name, text):
    win = find_window(title_part)
    if not win:
        print("找不到窗口: %s" % title_part)
        return
    items = walk(win)
    for c, d in items:
        try:
            if (c.Name or "").strip() == ctrl_name.strip():
                print("找到: [%s] %s" % (c.ControlTypeName, c.Name))
                c.Click(simulateMove=False)
                time.sleep(0.3)
                c.SendKeys(text, waitTime=0.01)
                print("已输入 %d 字符" % len(text))
                return
        except Exception:
            continue
    print("找不到控件: %s" % ctrl_name)


def cmd_read(title_part, ctrl_name):
    win = find_window(title_part)
    if not win:
        print("找不到窗口: %s" % title_part)
        return
    items = walk(win)
    for c, d in items:
        try:
            if (c.Name or "").strip() == ctrl_name.strip():
                print("[%s] %s" % (c.ControlTypeName, c.Name))
                print("Value: %s" % getattr(c, "GetValuePattern", lambda: "")())
                return
        except Exception:
            continue
    print("找不到控件: %s" % ctrl_name)


def cmd_find_text(title_part, text):
    """在窗口内找名字包含指定文字的控件"""
    win = find_window(title_part)
    if not win:
        print("找不到窗口: %s" % title_part)
        return
    items = walk(win)
    print("包含『%s』的控件:" % text)
    for c, d in items:
        try:
            nm = c.Name or ""
            if text in nm:
                br = c.BoundingRectangle
                print("  [%s] %s @(%d,%d) %dx%d" % (
                    c.ControlTypeName.replace("Control", ""), nm[:50],
                    br.left, br.top, w(br), h(br)))
        except Exception:
            continue


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1].lower()

    if cmd == "screen":
        cmd_screen()
    elif cmd == "windows":
        cmd_windows()
    elif cmd == "list" and len(sys.argv) >= 3:
        cmd_list(sys.argv[2])
    elif cmd == "click" and len(sys.argv) >= 4:
        cmd_click(sys.argv[2], sys.argv[3])
    elif cmd == "type" and len(sys.argv) >= 5:
        cmd_type(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "read" and len(sys.argv) >= 4:
        cmd_read(sys.argv[2], sys.argv[3])
    elif cmd == "find" and len(sys.argv) >= 4:
        cmd_find_text(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
