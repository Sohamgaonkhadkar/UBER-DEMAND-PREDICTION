"""Spent-development diagnostics, including clearly labeled oracle totals."""
import joblib,numpy as np,pandas as pd
from .protocol import *

def main():
    initialize();data=panel();rows=[];conditional=[]
    for month in ['2016-03','2016-04','2016-05']:
        z=np.load(OUT/('preliminary_'+month+'.npz'));a=z['actual'];nextmonth=str((pd.Timestamp(month+'-01')+pd.offsets.MonthBegin(1)).date());oo=origins(month+'-01',nextmonth)
        times=pd.DatetimeIndex([o+pd.Timedelta(minutes=15*k) for o in oo for k in range(48)])
        for name in ['hourly_disaggregated','half_hour_disaggregated']:
            m=joblib.load(OUT/'models'/month/(name+'.joblib'));f=m.factor;p=z[name];actual_totals=a.reshape(len(a),48//f,f,30).sum(axis=2)
            predicted_totals=p.reshape(len(a),48//f,f,30).sum(axis=2);oracle=[]
            for k in range(48//f):
                t=oo+pd.Timedelta(minutes=15*f*k);shares=m.shares[np.asarray(t.day_of_week),np.asarray(t.hour*4+t.minute//15)]
                oracle.append(np.moveaxis(shares,2,1)*actual_totals[:,k,None,:])
            oracle=np.concatenate(oracle,axis=1)
            rows.append(dict(month=month,model=name,regional_quarter_WAPE=metrics(a,p)['WAPE'],regional_coarse_WAPE=metrics(actual_totals,predicted_totals)['WAPE'],oracle_actual_totals_with_train_shares_quarter_WAPE=metrics(a,oracle)['WAPE']))
        cutoff=pd.Timestamp(month+'-01',tz=TZ)-pd.Timedelta('2h');training=data[data.datetime<cutoff];q=training.groupby('region').total_pickups.quantile(.95).to_numpy()
        for name in ['pure_xgboost','pure_lightgbm','calendar_feedback','sparse_nb','temporal_reconciled','cross_temporal']:
            p=z[name];event=np.broadcast_to((times.date==pd.Timestamp('2016-05-30').date()).reshape(len(a),48,1),a.shape)
            masks={'night':np.broadcast_to((times.hour<6).reshape(len(a),48,1),a.shape),'weekend':np.broadcast_to((times.day_of_week>=5).reshape(len(a),48,1),a.shape),'airports':np.broadcast_to(np.isin(np.arange(30),[3,6])[None,None,:],a.shape),'sparse':np.broadcast_to(np.isin(np.arange(30),[2,24,28])[None,None,:],a.shape),'realized_train95_peak':a>=q[None,None,:]}
            if event.any():masks['Memorial_Day']=event;masks['other_days']=~event
            for label,mask in masks.items():conditional.append(dict(month=month,model=name,regime=label,n=int(mask.sum()),**metrics(a[mask],p[mask])))
    pd.DataFrame(rows).to_csv(REPORT/'coarse_method_diagnostics.csv',index=False)
    pd.DataFrame(conditional).to_csv(REPORT/'development_strategy_conditions.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
