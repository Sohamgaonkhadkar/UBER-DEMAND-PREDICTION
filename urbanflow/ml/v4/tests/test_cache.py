from types import SimpleNamespace
import numpy as np,pandas as pd
from ml.advanced.core import HISTORY,FeatureConfig,training_rows
from ml.v4 import fit

def test_reused_2016_cache_matches_full_causal_feature_computation(tmp_path,monkeypatch):
    monkeypatch.setattr(fit,'OUT',tmp_path)
    def signal(start,offset):
        t=pd.date_range(start,periods=HISTORY+8,freq='15min',tz='America/New_York')
        values=np.arange(len(t),dtype=np.float32)[:,None]%96+np.arange(30,dtype=np.float32)[None,:]+offset
        return pd.DataFrame(values,index=t,columns=range(30))
    november=signal('2015-11-02',10);jan=signal('2016-01-01',30)
    model=SimpleNamespace(config=FeatureConfig(),neighbors=None)
    fit.rows(jan,model)
    joined=pd.concat([november,jan]);x,y,t=fit.rows(joined,model)
    p=fit.valid_training_positions(joined.index)
    direct,actual,tt=training_rows(joined.to_numpy(dtype=np.float32),joined.index,p,model.config,np.ones(30))
    np.testing.assert_array_equal(x.to_numpy(),direct.to_numpy())
    np.testing.assert_array_equal(y,actual)
    assert t.equals(pd.DatetimeIndex(tt))
