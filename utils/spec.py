import pandas as pd
from sfsaw.utils.enums import Unit


SPEC_TO_UNIT = {
    "ABS": Unit.ABS,
    "LOGMAG": Unit.DB,
    "SWR": Unit.VSWR,
    "DELAY": Unit.SEC,
}


class Spec:
    """
    A class to handle FI specification data.

    Parameters
    ----------
    file : str, optional
        Path to the specification CSV file. If provided, the file is loaded on initialization.

    Attributes
    ----------
    df : pandas.DataFrame
        A DataFrame containing parsed specification data, indexed by spec name.
    """

    def __init__(self, file: str, upper: bool = True):
        self.generate_dataframe(file)
        if upper:
            self.upper()

    def generate_dataframe(self, file: str) -> pd.DataFrame:
        """
        Parse a specification CSV file and construct a DataFrame.

        Parameters
        ----------
        file : str
            Path to the specification CSV file.
        """
        df = pd.read_csv(file, encoding='shift-JIS')

        # トレース定義表の読み込み
        start_idx = df[df.iloc[:, 0] == "Trace Number"].index[0]
        stop_idx = df[df.iloc[:, 0] == "Number of Measurement"].index[0]
        trace_table = df.iloc[start_idx:stop_idx]
        trace_table.columns = trace_table.iloc[0]
        trace_table = trace_table[1:]

        traces = []

        # テーブルから一行ずつ取得
        for _, row in trace_table.iterrows():
            traces.append({
                "meas": row['Meas'],
                "po": int(row['Meas'][1]),
                "pi": int(row['Meas'][2]),
                "unit": SPEC_TO_UNIT[row['FORMAT']],
            })

        # Spec表の読み込み
        start_idx = df[df.iloc[:, 0] == "Measurement Number"].index[0]
        spec_table = df.iloc[start_idx:]
        spec_table.columns = spec_table.iloc[0]
        spec_table = spec_table[1:]

        specs = {
            "name": [],
            "unit": [],
            "po": [],
            "pi": [],
            "fstart": [],
            "fstop": [],
            "min": [],
            "max": [],
            "offset": [],
        }

        # テーブルから一行ずつ取得
        for _, row in spec_table.iterrows():
            # 無効なSpecを除外
            if row['Judge Valid(ON)/Invalid(OFF)'] == "OFF":
                continue

            trace_num = int(row['Trace Number'])
            trace = traces[trace_num-1]

            specs["name"].append(row["TITLE"])
            specs["unit"].append(trace["unit"])
            specs["po"].append(trace["po"])
            specs["pi"].append(trace["pi"])
            specs["fstart"].append(float(row['START[MHz]'])*1e6)
            specs["fstop"].append(float(row['STOP[MHz]'])*1e6)
            specs["min"].append(float(row['LowerLimit']))
            specs["max"].append(float(row['UpperLimit']))
            specs["offset"].append(row['Offset'])

        # データフレームに変換
        self.df = pd.DataFrame(specs)
        self.df.set_index("name", inplace=True)

        return self.df

    @property
    def names(self) -> list[str]:
        return self.df.index.to_list()

    def upper(self):
        self.df.index = self.df.index.str.upper()

    def get(self, spec_name: str) -> dict:
        return self.df.loc[spec_name, :].to_dict()

    def filter(self,
               fstart: float | None = None,
               fstop: float | None = None,
               words_to_exlude: list[str] | None = None):
        """
        Filter the specification DataFrame by frequency range and keyword exclusion.

        Parameters
        ----------
        fstart : float, optional
            Minimum frequency in Hz. Only specs with `fstop` greater than or equal to this value will be retained.
        fstop : float, optional
            Maximum frequency in Hz. Only specs with `fstart` less than or equal to this value will be retained.
        words_to_exclude : list of str, optional
            List of substrings. Any spec whose name contains one of these will be excluded.
        """
        if fstart is not None:
            self.df = self.df[fstart <= self.df["fstop"]]

        if fstop is not None:
            self.df = self.df[self.df["fstart"] <= fstop]

        if words_to_exlude:
            self.df = self.df[~self.df.index.str.contains('|'.join(words_to_exlude))]

    def to_SNSS_string(self, ckt_name: str) -> str:
        text = "\t------------\t--------\t--------\t-----\t----\t-----\t--------\t--\t----\t-----\n"
        text += "\tSPEC {# ITEM,\tF1[MHz],\tF2[MHz],\tRESP,\tVAL,\tUNIT,\tMAX/MIN,\tW,\tERR,\tFUNC}\n"
        text += "\t------------\t--------\t--------\t-----\t----\t-----\t--------\t--\t----\t-----\n"
        for _, row in self.df.iterrows():
            name = row.name
            f1 = row["fstart"]*1e-6
            f2 = row["fstop"]*1e-6
            resp = f"S({row['po']},{row['pi']},{ckt_name})"
            val, maxmin = (row["min"], "MIN") if row["max"] >= 999 else (row["max"], "MAX")
            unit = row["unit"].value[0]
            text += f"\tSPEC {{{name},\t{f1},\t{f2},\t{resp},\t{val},\t{unit},\t{maxmin},\t0,\tWORST,\tLIN}}\n"
        return text[:-1]
