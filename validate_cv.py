"""Expanding timestamp CV diagnostics confined to the pre-validation training era.

No new hyperparameter selection and no final-test access. Eight unique timestamp
slots form the 2-hour gap regardless of 30 regions or source row ordering.
"""
import json,pathlib,sys
import joblib,numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'urbanflow'))
from ml.features import training_features,recursive
from ml.train import metrics

def main():
    panel,x=training_features(pd.read_parquet(ROOT/'artifacts/counts_ny.parquet'))
    ready=x.notna().all(axis=1)
    allowed=ready & (panel.datetime<pd.Timestamp('2016-02-15',tz='America/New_York')-pd.Timedelta('2h'))
    unique=pd.DatetimeIndex(panel.loc[allowed,'datetime'].unique()).sort_values()
    records=[]
    for name in ['lightgbm','xgboost']:
        artifact=joblib.load(ROOT/f'artifacts/candidates/{name}_v2.joblib')
        for fold,(a,b) in enumerate(TimeSeriesSplit(n_splits=3,gap=8).split(unique)):
            train=panel.datetime.isin(unique[a])&ready;val=panel.datetime.isin(unique[b])&ready
            assert panel.loc[train,'datetime'].max()<panel.loc[val,'datetime'].min()-pd.Timedelta('2h')
            assert set(panel.loc[train,'region'])==set(panel.loc[val,'region'])==set(range(30))
            model=clone(artifact['model']);model.fit(x[train],panel.loc[train,'total_pickups'])
            one=metrics(panel.loc[val,'total_pickups'],np.maximum(0,model.predict(x[val])))
            origin=unique[b][0]
            wide=panel.pivot(index='datetime',columns='region',values='total_pickups')
            pred=recursive(model,wide.loc[wide.index<origin].to_numpy(),origin,list(range(30)),48)
            truth=wide.reindex(pd.date_range(origin,periods=48,freq='15min')).to_numpy()
            records.append(dict(model=name,fold=fold+1,train_start=str(unique[a][0]),train_end=str(unique[a][-1]),
                validation_start=str(unique[b][0]),validation_end=str(unique[b][-1]),gap_slots=8,
                one_step=one,recursive_48=metrics(truth,pred)))
    (ROOT/'artifacts/chronological_cv.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
