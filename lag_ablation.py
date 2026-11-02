"""Fixed-hyperparameter validation-only ablation of daily/weekly lag additions."""
import json,pathlib,sys,time
import joblib,pandas as pd
from sklearn.base import clone
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'urbanflow'))
from ml.features import training_features,chronological_split
from ml.train import backtest,metrics

class SubsetModel:
    def __init__(self,model,columns):self.model=model;self.columns=columns
    def predict(self,x):return self.model.predict(x[self.columns])

def main():
    panel,x=training_features(pd.read_parquet(ROOT/'artifacts/counts_ny.parquet'))
    train,_,_=chronological_split(panel);train &= x.notna().all(axis=1)
    artifact=joblib.load(ROOT/'artifacts/candidates/lightgbm_v2.joblib')
    cols=[c for c in x if c not in ['lag_96','lag_672']]
    model=clone(artifact['model']);model.fit(x.loc[train,cols],panel.loc[train,'total_pickups'])
    origins=pd.to_datetime(['2016-02-16 00:00','2016-02-20 12:00','2016-02-27 23:00']).tz_localize('America/New_York')
    predictions,_=backtest(SubsetModel(model,cols),panel,origins,'lightgbm_without_daily_weekly_lags')
    result=dict(without_added_lags=metrics(predictions.actual,predictions.prediction),
        with_added_lags=json.loads((ROOT/'artifacts/results.json').read_text())['lightgbm']['validation_recursive'],
        removed=['lag_96','lag_672'],policy='Same train data and fixed hyperparameters, validation only; rolling windows retained. No final candidate changes after seeing test results.')
    out=ROOT/'artifacts/ablations';out.mkdir(exist_ok=True)
    joblib.dump(dict(model=model,feature_columns=cols),out/'lightgbm_without_daily_weekly.joblib')
    (out/'lag_ablation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
