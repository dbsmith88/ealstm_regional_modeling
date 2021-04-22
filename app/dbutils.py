import sqlite3
from typing import List, Tuple

import numpy as np
import pandas as pd
from numba import njit
import os


# Database catchment characteristics ignored
INVALID_ATTR = [
    'AreaSqKM', 'CentroidLatitude', 'CentroidLongitude', 'ComID', 'DELTALEVEL', 'DIRECTION', 'DivDASqKM',
    'DivEffect', 'Divergence', 'DnDrainCou', 'DnHydroseq', 'DnLevel', 'DnLevelPat', 'DnMinorHyd', 'ELEVFIXED',
    'FROMCOMID', 'FROMHYDSEQ', 'FROMLVLPAT', 'Fcode', 'Fdate', 'FromMeas', 'FromNode', 'GAPDISTKM', 'HWTYPE',
    'HasGeo', 'Hydroseq', 'LengthKM', 'LevelPathI', 'MAXELEVRAW', 'MAXELEVSMO', 'MINELEVRAW', 'MINELEVSMO',
    'NODENUMBER', 'OutDiv', 'Pathlength', 'ReachCode', 'Region', 'RtnDiv', 'SLOPELENKM', 'STATUSFLAG',
    'StartFlag', 'StreamCalc', 'StreamLeve', 'StreamOrde', 'TOCOMID', 'TOHYDSEQ', 'TOLVLPAT', 'TOTMA',
    'TerminalFl', 'TerminalPa', 'ThinnerCod', 'Tidal', 'ToMeas', 'ToNode', 'TotDASqKM', 'UpHydroseq',
    'UpLevelPat', 'VPUIn', 'VPUOut', 'WBAreaType', 'omcat', 'pctbl2011cat', 'pctcrop2011cat',
    'pctgrs2011cat', 'pcthay2011cat', 'pcthbwet2011cat', 'pctice2011cat', 'pctmxfst2011cat',
    'pctow2011cat', 'pctshrb2011cat', 'pcturbhi2011cat', 'pcturblo2011cat', 'pcturbmd2011cat',
    'pcturbop2011cat', 'pctwdwet2011cat', 'prcp_tot', 'precip8110cat', 'runoffcat', 'tmax8110cat', 'tmin8110cat'
]

# DB scaler values calculated from hydro_data.data_metrics.py
# DB_SCALER = {
#     'input_means': np.array(
#         [3.712443580719469, 208.72621428471857, 301.86262972212364, 90.14117415174269, 290.9826865721821,
#          282.1923567598892, 43193.262035473235, 1.0362304370796214]),
#     'input_stds': np.array(
#         [10.007500564603152, 107.08220303864564, 30.000556446613594, 201.25006759299765, 6.8093145665405785,
#          4.288644686248762, 7281.311437861614, 0.571235454347913]),
#     'output_mean': np.array([1.49996196]),
#     'output_std': np.array([3.62443672])
# }

DB_SCALER = {
    'input_means': np.array(
        [3.712443580719469, 208.72621428471857, 290.9826865721821, 282.1923567598892, 43193.262035473235]),
    'input_stds': np.array(
        [10.007500564603152, 107.08220303864564, 6.8093145665405785, 4.288644686248762, 7281.311437861614]),
    'output_mean': np.array([1.49996196]),
    'output_std': np.array([3.62443672])
}


class DBController:
    TIMEOUT = 1.0

    def __init__(self, db_path: str):
        self.connected = False
        self.db_path = db_path
        self.conn = self.connect()

    def connect(self):
        self.connected = True
        return sqlite3.connect(self.db_path, timeout=self.TIMEOUT)

    def close(self):
        self.connected = False
        self.conn.close()


def load_attributes(db_path: str,
                    gauges: List,
                    keep_features: List = None) -> pd.DataFrame:
    """Load attributes from database file into DataFrame

    Parameters
    ----------
    db_path : str
        Path to sqlite3 database file
    basins : List
        List containing the 8-digit USGS gauge id
    drop_lat_lon : bool
        If True, drops latitude and longitude column from final data frame, by default True
    keep_features : List
        If a list is passed, a pd.DataFrame containing these features will be returned. By default,
        returns a pd.DataFrame containing the features used for training.

    Returns
    -------
    pd.DataFrame
        Attributes in a pandas DataFrame. Index is USGS gauge id. Latitude and Longitude are
        transformed to x, y, z on a unit sphere.
    """
    if keep_features is None:
        keep_features = []
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        query = "SELECT DISTINCT D.parameter from CatchmentData as D INNER JOIN CatchmentGage as C WHERE D.COMID=C.COMID"
        cur.execute(query)
        columns = cur.fetchall()
        df_columns = ["gageID"]
        for c in columns:
            c = c[0]
            if c not in INVALID_ATTR and c not in keep_features:
                df_columns.append(c)
        data = []
        for g in gauges:
            row = {"gageID": g}
            query = "SELECT D.parameter, D.value from CatchmentData as D INNER JOIN CatchmentGage as C WHERE D.COMID=C.COMID AND gageID=?"
            value = (g,)
            cur.execute(query, value)
            attributes = cur.fetchall()
            for a in attributes:
                if a[0] not in INVALID_ATTR and a not in keep_features:
                    row[a[0]] = a[1]
            data.append(row)
        df = pd.DataFrame(data=data, columns=df_columns)
    df.set_index('gageID', inplace=True)
    return df


def normalize_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Normalize features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to normalize
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Normalized features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """

    if variable == 'inputs':
        feature = (feature - DB_SCALER["input_means"]) / DB_SCALER["input_stds"]
    elif variable == 'output':
        feature = (feature - DB_SCALER["output_mean"]) / DB_SCALER["output_std"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Rescale features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to rescale
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Rescaled features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """
    if variable == 'inputs':
        feature = feature * DB_SCALER["input_stds"] + DB_SCALER["input_means"]
    elif variable == 'output':
        feature = feature * DB_SCALER["output_std"] + DB_SCALER["output_mean"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape data into LSTM many-to-one input samples

    Parameters
    ----------
    x : np.ndarray
        Input features of shape [num_samples, num_features]
    y : np.ndarray
        Output feature of shape [num_samples, 1]
    seq_length : int
        Length of the requested input sequences.

    Returns
    -------
    x_new: np.ndarray
        Reshaped input features of shape [num_samples*, seq_length, num_features], where 
        num_samples* is equal to num_samples - seq_length + 1, due to the need of a warm start at
        the beginning
    y_new: np.ndarray
        The target value for each sample in x_new
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new


def load_forcing(db_path: str, gage_id: str, aggregate: bool = True) -> Tuple[pd.DataFrame, float]:
    """Load forcing data from sqlite database.

    Parameters
    ----------
    db_path : str
        Path to the sqlite database containing all the forcing data
    gage_id : str
        8-digit USGS gauge id
    aggregate : bool
        Default=False, if true then aggregate forcing values to daily sums/means

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the forcing data
    area: float
        Catchment area from the NHDPlus dataset

    Raises
    ------
    RuntimeError
        If not forcing file was found.
    """
    if not os.path.exists(db_path):
        raise RuntimeError(f"Database file not found at {db_path}")

    db = DBController(db_path=db_path)
    query = "SELECT * FROM ForcingData WHERE gageID='{}'".format(gage_id)
    df = pd.read_sql_query(query, db.conn)
    dates = (df.year.map(str) + "/" + df.mnth.map(str) + "/" + df.day.map(str))
    df["date"] = pd.to_datetime(dates, format="%Y/%m/%d")
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    query = "SELECT value FROM CatchmentData as D INNER JOIN CatchmentGage as G WHERE D.COMID=G.COMID AND D.parameter='AreaSqKM' AND G.gageID=?"
    values = (gage_id,)
    c = db.conn.cursor()
    c.execute(query, values)
    area = float(c.fetchone()[0])

    db.close()
    df_temp = df.copy()

    if aggregate:
        agg_df = pd.DataFrame()
        agg_df['prcp'] = df[['prcp']].groupby([df["date"].dt.year, df["date"].dt.month, df["date"].dt.day]).sum().prcp
        agg_df['srad'] = df[['srad']].groupby([df["date"].dt.year, df["date"].dt.month, df["date"].dt.day]).mean().srad
        agg_df['lrad'] = df[['lrad']].groupby([df["date"].dt.year, df["date"].dt.month, df["date"].dt.day]).mean().lrad
        agg_df['swe'] = df[['swe']].groupby([df["date"].dt.year, df["date"].dt.month, df["date"].dt.day]).sum().swe
        agg_df['tmax'] = df[['tmax']].groupby([df["date"].dt.year, df["date"].dt.month, df["date"].dt.day]).max().tmax
        agg_df['tmin'] = df_temp[['tmax']].groupby(
            [df_temp["date"].dt.year, df_temp["date"].dt.month, df_temp["date"].dt.day]).min().tmax
        agg_df['vp'] = df[['vp']].groupby([df["date"].dt.year, df["date"].dt.month, df["date"].dt.day]).mean().vp
        agg_df['et'] = df[['et']].groupby([df["date"].dt.year, df["date"].dt.month, df["date"].dt.day]).sum().et
        dates = (df.year.map(str) + "/" + df.mnth.map(str) + "/" + df.day.map(str)).unique()
        agg_df.index = pd.to_datetime(dates, format="%Y/%m/%d")
        df = agg_df

    return df, area


def load_discharge(db_path: str, gage_id: str, area: float) -> pd.Series:
    """[summary]

    Parameters
    ----------
    db_path : str
        Path to the sqlite database containing all the discharge data
    gage_id : str
        8-digit USGS gauge id
    area : float
        Catchment area, used to normalize the discharge to mm/day

    Returns
    -------
    pd.Series
        A Series containing the discharge values.

    Raises
    ------
    RuntimeError
        If no discharge file was found.
    """
    if not os.path.exists(db_path):
        raise RuntimeError(f"Database file not found at {db_path}")

    db = DBController(db_path=db_path)
    query = "SELECT * FROM StreamflowData WHERE gageID='{}'".format(gage_id)
    df = pd.read_sql_query(query, db.conn)
    dates = (df.year.map(str) + "/" + df.mnth.map(str) + "/" + df.day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    db.close()
    df = df[~df.index.duplicated(keep='first')]
    # normalize discharge from cubic feed per second to mm per day
    df = df.rename(columns={"streamflow": "QObs"})
    # df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10 ** 6)
    df.QObs = df.QObs * 86400 / (area * 10 ** 6)
    df.QObs = df.QObs.astype(np.float)
    return df.QObs
