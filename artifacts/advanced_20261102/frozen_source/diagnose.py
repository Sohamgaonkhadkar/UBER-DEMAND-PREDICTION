import json,time
import joblib,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml.features import training_features,recursive
from .protocol import *

def main():
    plan=initialize()
    panel,x=training_features(pd.read_parquet(ROOT/'artifacts/counts_ny.parquet'))
    wide=panel.pivot(index='datetime',columns='region',values='total_pickups')
    fit=panel.datetime<pd.Timestamp('2016-02-15',tz=TZ)
    means=panel[fit].groupby('region').total_pickups.mean()
    peaks=panel[fit].groupby('region').total_pickups.quantile(.95)
    rows=[]
    for name in ['xgboost','lightgbm']:
        model=joblib.load(ROOT/f'artifacts/candidates/{name}_v2.joblib')['model']
        teacher=np.maximum(0,model.predict(x.loc[panel.datetime>=pd.Timestamp('2016-03-01',tz=TZ)]))
        teacher=pd.Series(teacher,index=pd.MultiIndex.from_frame(panel.loc[panel.datetime>=pd.Timestamp('2016-03-01',tz=TZ),['datetime','region']]))
        for origin in origins('2016-03-01','2016-04-01'):
            before=wide.loc[wide.index<origin].to_numpy();vol=before[-8:].std(axis=0)/(before[-8:].mean(axis=0)+1)
            pred=recursive(model,before,origin,list(range(30)),48)
            truth=wide.reindex(pd.date_range(origin,periods=48,freq='15min')).to_numpy()
            for k in range(48):
                ts=origin+pd.Timedelta(minutes=15*k)
                for r in range(30):
                    rows.append((name,str(origin),str(ts),k+1,r,float(truth[k,r]),float(pred[k,r]),float(teacher.loc[(ts,r)]),
                        ts.hour,ts.day_of_week,int(ts.day_of_week>=5),int(ts.hour in [7,8,9,17,18,19]),int(ts.hour<6),
                        int(truth[k,r]>=peaks[r]),float(vol[r]),'low' if means[r]<=means.quantile(1/3) else 'high' if means[r]>=means.quantile(2/3) else 'medium'))
            print('diagnosed',name,origin,flush=True)
    d=pd.DataFrame(rows,columns=['model','origin','datetime','step','region','actual','prediction','teacher_prediction','hour','dow','weekend','peak_hour','night','peak_demand','origin_volatility','demand_tier'])
    d.to_parquet(OUT/'diagnostic_predictions.parquet',index=False)
    summary={}
    for name,g in d.groupby('model'):
        summary[name]=dict(recursive=metrics(g.actual,g.prediction),teacher_forced=metrics(g.actual,g.teacher_prediction),
            by_origin={o:metrics(z.actual,z.prediction) for o,z in g.groupby('origin')})
    (REPORT/'baseline_diagnostics.json').write_text(json.dumps(summary,indent=2))
    for column in ['step','region','hour','dow','weekend','peak_hour','night','peak_demand','demand_tier']:
        records=[]
        for (name,key),g in d.groupby(['model',column]):records.append(dict(model=name,group=str(key),n=len(g),**metrics(g.actual,g.prediction)))
        pd.DataFrame(records).to_csv(REPORT/f'diagnostics_by_{column}.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,g in d.groupby('model'):
        z=[(k,metrics(a.actual,a.prediction),metrics(a.actual,a.teacher_prediction)) for k,a in g.groupby('step')]
        axes[0].plot([k for k,m,t in z],[m['WAPE'] for k,m,t in z],label=f'{name} recursive')
        axes[0].plot([k for k,m,t in z],[t['WAPE'] for k,m,t in z],ls='--',label=f'{name} teacher-forced')
        axes[1].plot([k for k,m,t in z],[m['bias_pct'] for k,m,t in z],label=name)
    axes[0].set(xlabel='Lead (15-minute steps)',ylabel='WAPE at lead (%)');axes[1].set(xlabel='Lead',ylabel='Signed count bias (%)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(REPORT/'feedback_drift.png',dpi=140);plt.close(fig)
    print(json.dumps({k:{s:v[s] for s in ['recursive','teacher_forced']} for k,v in summary.items()},indent=2),flush=True)

if __name__=='__main__':main()
