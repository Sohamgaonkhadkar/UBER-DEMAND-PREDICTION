"""Inventory changes, verify original model hashes and attach final run provenance."""
import hashlib,json,pathlib,subprocess
import mlflow
ROOT=pathlib.Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()

def main():
    before=json.loads((ROOT/'reports/baseline/original.json').read_text())
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'lgbm_taxi_pipeline.pkl',ROOT/'urbanflow/backend/data/lgbm_taxi_pipeline.pkl']}
    assert set(hashes.values())=={before['artifact_sha256']}
    inventory={}
    for name,folder in [('experimentation',ROOT),('production',ROOT/'urbanflow')]:
        inventory[name]=dict(branch=git('branch','--show-current',cwd=folder),head=git('rev-parse','HEAD',cwd=folder),
            status=git('status','--short',cwd=folder).splitlines(),tracked_files=git('ls-files',cwd=folder).splitlines())
    (ROOT/'reports/repository_inventory.json').write_text(json.dumps(inventory,indent=2))
    changed=dict(original_model_hashes=hashes,
        root_changes=inventory['experimentation']['status'],production_changes=inventory['production']['status'],
        preexisting_user_changes=['Breaking_NYC_to_Regions.ipynb','final model.ipynb','lightgbm.ipynb','output.png'],
        generated_artifacts=[str(p.relative_to(ROOT)) for p in (ROOT/'artifacts').rglob('*') if p.is_file()],
        reports=[str(p.relative_to(ROOT)) for p in (ROOT/'reports').rglob('*') if p.is_file() and 'baseline' not in p.parts])
    (ROOT/'reports/changed_files.json').write_text(json.dumps(changed,indent=2))
    uri='sqlite:///'+str(ROOT/'local_tracking.db')
    client=mlflow.MlflowClient(tracking_uri=uri)
    experiment=client.get_experiment_by_name('urbanflow_local_repair')
    results=json.loads((ROOT/'artifacts/results.json').read_text())
    evidence=[]
    for run in client.search_runs([experiment.experiment_id]):
        name=run.data.tags.get('mlflow.runName')
        if name in results and abs(run.data.metrics.get('test_recursive_WAPE',-1)-results[name]['test_recursive']['WAPE'])<1e-8:
            client.set_tag(run.info.run_id,'effective_dataset','counts_ny.parquet')
            client.set_tag(run.info.run_id,'timezone','America/New_York; invalid DST source clock quarantined')
            for path in ['dataset_metadata.json','splits.json','candidates/feature_schema.json','chronological_cv.json']:
                client.log_artifact(run.info.run_id,str(ROOT/'artifacts'/path),artifact_path='final_provenance')
            evidence.append(dict(run_id=run.info.run_id,model=name,artifact_uri=run.info.artifact_uri))
    (ROOT/'reports/mlflow_final_runs.json').write_text(json.dumps(evidence,indent=2))
    print('original model hashes unchanged; final MLflow runs:',len(evidence))

if __name__=='__main__':main()
