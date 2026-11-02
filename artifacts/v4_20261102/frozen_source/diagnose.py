"""Describe already-spent calendar failures, without fitting corrections."""
import holidays
import numpy as np,pandas as pd,json,psutil,shutil
from .protocol import *

def calendar_status(times):
    times=pd.DatetimeIndex(times).tz_convert(TZ)
    h=holidays.US(years=sorted(set(times.year)));dates=times.normalize()
    return np.asarray([d.date() in h for d in dates]),h

def main():
    initialize();data=panel();wide=data.pivot(index='datetime',columns='region',values='total_pickups')
    hol,h=calendar_status(wide.index)
    holiday_days=[dict(date=str(d),name=name,rows=int((wide.index.date==d).sum())) for d,name in h.items() if wide.index[0].date()<=d<=wide.index[-1].date()]
    save(REPORT/'calendar_coverage.json',dict(holidays=holiday_days,independent_holiday_dates=len(holiday_days),
        existing_features=['public_holiday','holiday_adjacent (day before OR after; direction not distinguished)'],
        calendar_profile='Weekday/quarter medians; holiday dates pooled with ordinary weekdays; no holiday-specific calendar state.',
        limits='March and April have no US federal holiday; May has one. Jan-Feb has only three distinct holiday dates. Event identity/time/region interactions cannot be reliably learned from one summer event.',
        resources=dict(available_RAM_GiB=psutil.virtual_memory().available/1024**3,free_disk_GiB=shutil.disk_usage(ROOT).free/1024**3)))
    z=np.load(PREVIOUS/'final_test_predictions.npz');result=json.loads((PREVIOUS/'final_test_results.json').read_text())
    oo=pd.to_datetime(result['origins'],utc=True).tz_convert(TZ);times=pd.DatetimeIndex([o+pd.Timedelta(minutes=15*k) for o in oo for k in range(48)])
    dates=np.asarray(times.date);ref=pd.Timestamp('2016-05-30').date();offset=np.array([(d-ref).days for d in dates]);rows=[]
    for model in ['matched_refit_current','matched_refit_tuned_xgboost','matched_refit_tuned_lightgbm']:
        a=z['actual'].reshape(-1,30);p=z[model].reshape(-1,30)
        for day in [-2,-1,0,1]:
            m=offset==day
            for label,regions in [('all',list(range(30))),('airports',[3,6]),('sparse',[2])]:
                if m.any():rows.append(dict(model=model,offset_days=day,region_group=label,**metrics(a[m][:,regions],p[m][:,regions])))
            for hour in range(24):
                mask=m&(times.hour==hour)
                if mask.any():rows.append(dict(model=model,offset_days=day,target_hour=hour,region_group='all_hour',**metrics(a[mask],p[mask])))
        for r in range(30):
            mask=offset==0
            rows.append(dict(model=model,offset_days=0,region_group=str(r),**metrics(a[mask,r],p[mask,r])))
    pd.DataFrame(rows).to_csv(REPORT/'spent_may_holiday_analysis.csv',index=False)
    s=wide[wide.index<pd.Timestamp('2016-03-01',tz=TZ)]
    sparse=pd.DataFrame(dict(region=s.columns,mean=s.mean().values,variance=s.var().values,zero_rate=(s==0).mean().values))
    sparse['variance_to_mean']=sparse.variance/(sparse['mean']+1e-9);sparse['poisson_zero_at_unconditional_mean']=np.exp(-sparse['mean'])
    sparse.to_csv(REPORT/'sparse_count_diagnosis.csv',index=False)
    print('Calendar coverage:',holiday_days,flush=True);print(sparse.sort_values('mean').head().to_string(index=False),flush=True)

if __name__=='__main__':main()
