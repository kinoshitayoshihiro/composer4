# monitor/progress_live.py を作成
from pathlib import Path
import re, time, os, sys
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

TQDM_BARLEN = 40


def bar(pct: float, width=TQDM_BARLEN):
    fill = int(pct * width)
    return "█" * fill + "─" * (width - fill)


def tail_text(path: Path, n=4000):
    # 大きなログでも末尾だけ効率よく読む
    with path.open("rb") as f:
        try:
            f.seek(-n, os.SEEK_END)
        except OSError:
            pass
        return f.read().decode("utf-8", errors="ignore")


def monitor(log_path: str, title="Training", update_sec=60, total_epochs_hint=15):
    log = Path(log_path)
    if not log.exists():
        raise FileNotFoundError(f"log not found: {log}")

    # tqdm形式: "epoch 1/15:" または "Epoch [1/15]" 両方に対応
    pat_epoch = re.compile(r"[Ee]poch\s*[\[]?(\d+)\s*/\s*(\d+)[\]]?")
    pat_loss = re.compile(r"(?:^|[^a-zA-Z_])(loss|train[_-]?loss)\s*[:=]\s*([0-9]*\.?[0-9]+)")
    pat_vloss = re.compile(r"(?:^|[^a-zA-Z_])(val[_-]?loss)\s*[:=]\s*([0-9]*\.?[0-9]+)")
    # tqdmのプログレスバー形式も認識: "8051/50016"
    pat_batch = re.compile(r"(?:batch|step|)\s*[\|]?\s*(\d+)/(\d+)\s*\[", re.IGNORECASE)

    hist_loss, hist_vloss = [], []
    cur_epoch, tot_epoch = 0, total_epochs_hint

    while True:
        txt = tail_text(log, n=200_000)

        # epoch
        for m in reversed(list(pat_epoch.finditer(txt))):
            cur_epoch = int(m.group(1))
            tot_epoch = int(m.group(2))
            break

        # losses（最後に出た値を採用）
        m_loss = list(pat_loss.finditer(txt))
        m_vloss = list(pat_vloss.finditer(txt))
        last_loss = float(m_loss[-1].group(2)) if m_loss else None
        last_vloss = float(m_vloss[-1].group(2)) if m_vloss else None
        if last_loss is not None:
            hist_loss.append(last_loss)
        if last_vloss is not None:
            hist_vloss.append(last_vloss)

        # inner batch/step progress（あれば）
        inner_pct_txt = ""
        m_batch = list(pat_batch.finditer(txt))
        if m_batch:
            bcur = int(m_batch[-1].group(1))
            btot = int(m_batch[-1].group(2))
            if btot > 0:
                inner_pct = bcur / btot
                inner_pct_txt = f" | step {bcur}/{btot} [{int(inner_pct*100):3d}%]"

        # 画面更新
        clear_output(wait=True)
        epoch_pct = cur_epoch / max(tot_epoch, 1)
        print(f"{title}")
        print("─" * (TQDM_BARLEN + 30))
        print(
            f"Epoch {cur_epoch}/{tot_epoch} [{int(epoch_pct*100):3d}%] {bar(epoch_pct)}{inner_pct_txt}"
        )

        if hist_loss:
            plt.figure(figsize=(6.5, 3))
            plt.plot(hist_loss, label="loss")
            if hist_vloss:
                plt.plot(hist_vloss, label="val_loss")
            plt.title(title + " — losses")
            plt.xlabel("updates")
            plt.ylabel("loss")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        time.sleep(update_sec)


# 使い方：
# monitor("logs/guitar_duv_raw.log", "🎸 Guitar DUV", update_sec=60)
