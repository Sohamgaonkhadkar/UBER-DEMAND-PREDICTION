"""Deep analysis of spent April outcomes; no model selection uses these errors."""
import json
import numpy as np,pandas as pd,joblib
from .protocol import *

def main():
    initialize()
    result=json.loads((OLD/'final_test_results.json').read_text());z=np.load(OLD/'final_test_predictions.npz')
    a=z['actual'];p=z[result['selected_id']]
    panel=read_development();cut=pd.Timestamp('2016-04-01',tz=TZ)-pd.Timedelta('2h')
    training=panel[panel.datetime<cut];wide=panel.pivot(index='datetime',columns='region',values='total_pickups')
    oo=pd.to_datetime(result['test_origins'],utc=True).tz_convert(TZ);pos=wide.index.get_indexer(oo)
    current=joblib.load(OLD/'advanced_candidate_v3.joblib')['model'];calendar=current.calendar.forecast(wide.to_numpy(dtype=np.float32),wide.index,pos,48)
    geo=json.loads((OLD/'geography.json').read_text());coordinates=np.array(geo['centers'])*geo['scale']+geo['mean']
    q=training.groupby('region').total_pickups.quantile(.95).to_numpy()
    times=pd.DatetimeIndex([t+pd.Timedelta(minutes=15*k) for t in oo for k in range(48)])
    peak=a>=q[None,None,:];night=(times.hour<6).reshape(len(oo),48)
    rows=[]
    for r in range(30):
        y,v=a[:,:,r],p[:,:,r]
        row=dict(region=r,centroid_longitude=coordinates[r,0],centroid_latitude=coordinates[r,1],
            mean_demand=float(y.mean()),coefficient_of_variation=float(y.std()/(y.mean()+1e-9)),zero_fraction=float(np.mean(y==0)),
            **metrics(y,v))
        row.update({f'peak_{k}':v for k,v in metrics(y[peak[:,:,r]],v[peak[:,:,r]]).items()})
        row.update({f'night_{k}':v for k,v in metrics(y[night],v[night]).items()})
        row['geographic_note']='JFK-centered (centroid/reference proximity)' if r==3 else 'LaGuardia-centered (centroid/reference proximity)' if r==6 else 'Land use not verified; do not treat display name as an exact boundary'
        rows.append(row)
    pd.DataFrame(rows).to_csv(REPORT/'spent_april_region_diagnosis.csv',index=False)
    by_time=[]
    af=a.reshape(-1,30);pf=p.reshape(-1,30)
    import holidays
    hol=holidays.US(years=[2016])
    for field,values in [('hour',times.hour),('weekday',times.day_of_week),('weekend',times.day_of_week>=5),('public_holiday',[t.date() in hol for t in times])]:
        values=np.asarray(values)
        for value in np.unique(values):by_time.append(dict(field=field,value=str(value),**metrics(af[values==value],pf[values==value])))
    pd.DataFrame(by_time).to_csv(REPORT/'spent_april_time_diagnosis.csv',index=False)
    lead=[]
    for k in range(48):
        for r in range(30):lead.append(dict(step=k+1,region=r,**metrics(a[:,k,r],p[:,k,r])))
    pd.DataFrame(lead).to_csv(REPORT/'spent_april_region_by_lead.csv',index=False)
    analyses={}
    for label,mask in [('realized_top95',peak),('calendar_expected_top95',calendar>=q[None,None,:]),('forecast_top95',p>=q[None,None,:])]:
        analyses[label]=dict(n=int(mask.sum()),**metrics(a[mask],p[mask]))
    analyses['last_lead_calendar_bias_pct']=metrics(a[:,47],calendar[:,47])['bias_pct']
    analyses['prediction_actual_increment_ratio']=float(np.abs(np.diff(p,axis=1)).mean()/np.abs(np.diff(a,axis=1)).mean())
    analyses['last_lead_std_ratio']=float(p[:,47].std()/a[:,47].std())
    analyses['architecture']=dict(first_lead_calendar_weight=0.,last_lead_calendar_weight=float(.6*(1-np.exp(-47/12))),
        last_lead_inner_week_anchor=.3,effective_last_GBDT_weight=float((1-.6*(1-np.exp(-47/12)))*.7),
        gain_pseudocount=100,gain_decay_steps=12,profile_statistic='training weekday/quarter median, within-weekday circular smoothing')
    analyses['interpretation']=['Realized-peak conditioning selects positive future innovations; its negative bias alone does not justify a forecast-time multiplier.',
        'Late forecast/calendar bias is negative. Calendar gain mostly expires by 12h; reliability may vary by origin/region.',
        'Only 45% as much quarter-to-quarter variation does not prove avoidable error: predictions should omit unpredictable noise; late cross-sectional SD retains about 96%.',
        'Airport regions have high relative errors and sparse overnight demand; flight-arrival/schedule causality remains unverified.',
        'April is spent. All novel gate/correction coefficients and selection are confined to earlier development data.']
    (REPORT/'diagnosis.json').write_text(json.dumps(analyses,indent=2))
    # Point estimates identify airport-centered clusters, not exact catchment polygons.
    faa=[dict(name='JFK',latitude=40+38/60+23.741/3600,longitude=-(73+46/60+43.292/3600)),dict(name='LGA',latitude=40+46/60+38/3600,longitude=-(73+52/60+21/3600))]
    distances=[]
    for r,airport in zip([3,6],faa):
        lon,lat=coordinates[r];x=(lon-airport['longitude'])*np.cos(np.deg2rad(lat));y=lat-airport['latitude']
        distances.append(dict(region=r,reference=airport,centroid_distance_km=float(111.2*np.hypot(x,y))))
    (REPORT/'airport_centroid_verification.json').write_text(json.dumps(dict(verification=distances,sources=['https://www.faa.gov/air_traffic/publications/atpubs/aip_html/part3_ad_2.0_new_york.html','https://www.faa.gov/documentLibrary/media/Order/2022-08-19_FAA_Order_JO_7400.11G.pdf'],limit='Reference proximity does not prove cluster boundaries or flight-driven causation.'),indent=2))
    print(json.dumps(analyses,indent=2))

if __name__=='__main__':main()
