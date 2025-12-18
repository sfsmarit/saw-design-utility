import pandas as pd
from matplotlib import pyplot as plt

from utils.spec import Spec


class CornerLot:
    def __init__(self) -> None:
        self.spec: Spec
        self.df_raw: pd.DataFrame
        self.df_judge: pd.DataFrame

    def is_passed(self, spec_name: str, value):
        d = self.spec.get(spec_name)
        return int(d["min"] <= value <= d["max"])

    def load(self, result_file, spec_file, is_mont: bool = True):
        # Specの読み込み
        self.spec = Spec(spec_file)

        # バンドの識別
        # Spec名の最後のセクションがバンド名になっている
        self.spec.df['band'] = self.spec.df.index.str.rsplit('_').str[-1]
        # B から始まる文字以外が含まれている = シングルバンド
        if (~self.spec.df['band'].str.startswith('B')).any():
            self.spec.df["band"] = "ANY"

        # リザルトファイルの読み込み
        if is_mont:
            # 列以外の読み込み
            self.df_raw = pd.read_csv(result_file, skiprows=4, header=None, encoding='utf-8-sig')

            # 再度先頭に戻して1行目を読み込み、列名を取得
            result_file.seek(0)
            first_line = result_file.readline().decode('utf-8-sig').strip()
            cols = first_line.split(',')

            # 空列を追加
            idx = cols.index("MparFileName")
            n_empty_cols = len(self.df_raw.columns) - len(cols)

            # 列を修正
            self.df_raw.columns = cols[:idx+1] + [""]*n_empty_cols + cols[idx+1:]
        else:
            self.df_raw = pd.read_csv(result_file, encoding="utf-8-sig")

        # 列名を大文字に変換
        self.df_raw.columns = self.df_raw.columns.str.upper()
        # 列名から[]を削除
        self.df_raw.columns = [c[:c.index("[")] if "[" in c else c for c in self.df_raw.columns]
        # Specに含まれる列だけ残す
        self.df_raw = self.sync(self.df_raw)
        # 数値に変換
        self.df_raw = self.df_raw.apply(pd.to_numeric, errors='coerce')
        # nan削除
        self.df_raw = self.df_raw.dropna()

        # 合否判定
        self.df_judge = self.df_raw.apply(lambda sr: sr.apply(lambda v: self.is_passed(sr.name, v)))
        failure_rates = (1 - self.df_judge.sum() / len(self.df_judge)) * 100
        failure_rates.name = "failrate"
        self.spec.df = self.spec.df.join(failure_rates, how="inner")

    def sync(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = [s for s in self.spec.df.index if s in df]
        return df[new_cols]

    def filter(self,
               fstart: float | None = None,
               fstop: float | None = None,
               words_to_exlude: list[str] | None = None):
        self.spec.filter(fstart=fstart,
                         fstop=fstop,
                         words_to_exlude=words_to_exlude)
        self.df_raw = self.sync(self.df_raw)
        self.df_judge = self.sync(self.df_judge)

    def plot_failure_rate(self):
        bands = self.spec.df['band'].unique()
        n_bands = len(bands)

        # グラフサイズ
        w, h = 8, 10
        fig, axes = plt.subplots(1, n_bands, figsize=(w*n_bands, h))

        if n_bands == 1:
            axes = [axes]

        # バンドごとに描画
        for band, ax in zip(bands, axes):
            # バンドの選択
            df = self.spec.df[self.spec.df["band"] == band]

            # Failした項目を抽出
            df = df[df["failrate"] > 0]

            if not len(df):
                # print(f"No failed entry for {band}")
                continue

            # ソート
            df = df.sort_values(by=["failrate"], ascending=True)

            # 描画
            df.plot.barh(y="failrate", ax=ax, color='dodgerblue')
            ax.set_title(f"{band}")
            ax.set_xlabel("Failure rate [%]")
            ax.legend().set_visible(False)
            ax.set_xlim(0, 100)

            # グラフに文字を追加
            for i, bar in enumerate(ax.containers[0]):
                spec_name = df.index[i]
                fstart = df.loc[spec_name, 'fstart']
                fstop = df.loc[spec_name, 'fstop']
                rate = bar.get_width()

                txt = f"({len(ax.containers[0])-i})   {fstart*1e-6:.1f}-{fstop*1e-6:.1f}MHz   {rate:.1f}%"  # type: ignore
                ax.text(3, bar.get_y() + bar.get_height()/2, txt, va='center', ha='left')

        plt.tight_layout(pad=12)
        return fig, axes


if __name__ == "__main__":
    result_csv = r"C:\Users\marit\Documents\cornerlot\DG032N\DG032N_cornerlot_char(row).csv"
    spec_csv = r"C:\Users\marit\Documents\specification\FI\DF033PQ677MD-F5.csv"
    is_mont = "mont" in result_csv

    freq_range = [0, 1000e6]
    spec_words_to_exlude = ['IR', 'OPB', 'SHB', "RIPPLE"]

    cornerlot = CornerLot()
    cornerlot.load(result_csv, spec_csv, is_mont=is_mont)
    cornerlot.filter(freq_range[0], freq_range[1], spec_words_to_exlude)
    cornerlot.plot_failure_rate()
    plt.show()
