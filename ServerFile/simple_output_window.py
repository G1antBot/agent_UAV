import threading
import time

class OutputWindow:
    """最小的简洁输出窗口：在独立线程中启动一个 Tkinter 窗口，提供 show(msg) 方法。
    如果运行环境不支持 GUI（例如 headless），会静默降级为 no-op。"""

    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()
        # 延迟导入 tkinter，以防在不支持 GUI 的环境中导入失败
        try:
            import tkinter as tk
            self._tk = tk
        except Exception:
            self._tk = None
        if self._tk is None:
            # headless 环境：降级为简单列表缓存
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---- 高级接口 ----
    def show_command(self, text: str):
        self.show(f"【指令输入】 {text}")

    def show_result(self, text: str):
        self.show(f"【结果】 {text}")

    def show_ai_parse(self, parse: dict):
        return

    def update_steps(self, idx: int, text: str, status: str):
        return

    def update_status(self, info: dict):
        return

    def append_log(self, msg: str):
        return

    def _run(self):
        tk = self._tk
        self.root = tk.Tk()
        self.root.title("UAV 简洁输出")
        self.text = tk.Text(self.root, height=12, width=60)
        self.text.pack(fill=tk.BOTH, expand=True)
        # 字体与标签
        try:
            import tkinter.font as tkfont
            default_font = tkfont.nametofont("TkDefaultFont")
            self._font_cmd = default_font.copy()
            self._font_cmd.configure(size=max(default_font.cget('size') + 4, 14), weight="bold")
        except Exception:
            self._font_cmd = None
        # 文本标签样式
        try:
            if self._font_cmd is not None:
                self.text.tag_configure('command', font=self._font_cmd, foreground='#003366')
        except Exception:
            pass
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X)
        clear_btn = tk.Button(btn_frame, text="清除", command=lambda: self.text.delete("1.0", "end"))
        clear_btn.pack(side=tk.LEFT, padx=4, pady=4)
        close_btn = tk.Button(btn_frame, text="关闭窗口", command=self._close)
        close_btn.pack(side=tk.LEFT, padx=4, pady=4)

        # 定期轮询队列并写入文本框
        def poll():
            try:
                with self._lock:
                    while self._queue:
                        msg = self._queue.pop(0)
                        if isinstance(msg, str) and msg.startswith("【指令输入】"):
                            try:
                                self.text.insert("end", msg + "\n", 'command')
                            except Exception:
                                self.text.insert("end", msg + "\n")
                        else:
                            self.text.insert("end", msg + "\n")
                        self.text.see("end")
            except Exception:
                pass
            self.root.after(200, poll)

        self.root.after(200, poll)
        try:
            self.root.mainloop()
        except Exception:
            pass

    def _close(self):
        try:
            if getattr(self, 'root', None):
                self.root.destroy()
        except Exception:
            pass

    def show(self, msg: str):
        if self._tk is None:
            # headless 模式：打印到 stdout 作为后备
            try:
                print(msg)
            except Exception:
                pass
            return
        with self._lock:
            self._queue.append(msg)
