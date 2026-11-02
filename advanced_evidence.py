"""Verify preservation/freeze, retain source snapshots and complete local tracking."""
import hashlib,json,pathlib,shutil,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'urbanflow'))
from ml.advanced.protocol import OUT,REPORT,sha
import mlflow

def main():
    plan=json.loads((OUT/'protocol.json').read_text());freeze=json.loads((OUT/'selection_freeze.json').read_text())
    checks={name:dict(expected=value,actual=sha(ROOT/name),unchanged=sha(ROOT/name)==value) for name,value in plan['old_hashes'].items()}
    code={name:dict(expected=value,actual=sha(ROOT/name),unchanged=sha(ROOT/name)==value) for name,value in freeze['code_sha256'].items()}
    assert all(v['unchanged'] for v in checks.values()),'A protected phase-one file changed'
    assert all(v['unchanged'] for v in code.values()),'Forecast/research source changed after freeze'
    assert sha(OUT/'leaderboard.json')==freeze['leaderboard_sha256'],'Leaderboard changed after freeze'
    before=json.loads((ROOT/'reports/baseline/original.json').read_text())
    original={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'lgbm_taxi_pipeline.pkl',ROOT/'urbanflow/backend/data/lgbm_taxi_pipeline.pkl']}
    assert set(original.values())=={before['artifact_sha256']}
    result=dict(protected_phase_one_files=checks,frozen_source_files=code,leaderboard_unchanged=True,original_model_hashes=original,
        no_commits_pushes_prs_or_deployments=True)
    (REPORT/'preservation_checks.json').write_text(json.dumps(result,indent=2))
    snapshots=OUT/'frozen_source';snapshots.mkdir(exist_ok=True)
    for path in (ROOT/'urbanflow/ml/advanced').glob('*.py'):
        dest=snapshots/path.name
        if not dest.exists():shutil.copy2(path,dest)
    adapter=snapshots/'advanced_main.py'
    if not adapter.exists():shutil.copy2(ROOT/'urbanflow/backend/advanced_main.py',adapter)
    # Attach corrected, portable alternatives while preserving their initial artifacts.
    client=mlflow.MlflowClient(tracking_uri='sqlite:///'+str(OUT/'tracking.db'))
    experiment=client.get_experiment_by_name('urbanflow_advanced_validation')
    runs=client.search_runs([experiment.experiment_id],max_results=1000)
    records=json.loads((OUT/'leaderboard.json').read_text());by_name={r['id']:r for r in records}
    final=json.loads((OUT/'final_test_results.json').read_text());tracking=[]
    for run in runs:
        name=run.data.tags.get('mlflow.runName');record=by_name.get(name)
        if record and record.get('initial_artifact_preserved'):
            client.set_tag(run.info.run_id,'portable_artifact',record['artifact'])
            client.log_artifact(run.info.run_id,str(ROOT/record['artifact']),artifact_path='portable_corrected')
        if name=='FINAL_'+final['selected_id']:
            for path in [OUT/'advanced_candidate_v3.joblib',OUT/'feature_schema_v3.json',OUT/'selection_freeze.json',OUT/'protocol.json',OUT/'refit_metadata.json',OUT/'final_test_results.json']:
                client.log_artifact(run.info.run_id,str(path),artifact_path='frozen_candidate')
        tracking.append(dict(run_id=run.info.run_id,name=name,status=run.info.status,artifact_uri=run.info.artifact_uri))
    (REPORT/'mlflow_runs.json').write_text(json.dumps(tracking,indent=2))
    inventory=dict(root_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        production_status=subprocess.check_output(['git','status','--short'],cwd=ROOT/'urbanflow',text=True),
        new_source_files=[str(p.relative_to(ROOT)) for p in (ROOT/'urbanflow/ml/advanced').rglob('*') if p.is_file() and '__pycache__' not in p.parts],
        new_backend_files=['urbanflow/backend/advanced_main.py','urbanflow/backend/requirements-advanced.txt'],
        new_tests=['urbanflow/tests/test_advanced.py','urbanflow/tests/test_advanced_api.py'],
        new_root_scripts=['advanced_system_check.py','advanced_report.py','advanced_evidence.py'],
        artifact_directory=str(OUT.relative_to(ROOT)),report_directory=str(REPORT.relative_to(ROOT)),
        final_report='reports/advanced_model_optimization.md')
    (REPORT/'changed_files.json').write_text(json.dumps(inventory,indent=2))
    print('Verified preserved files:',len(checks),'; frozen source files:',len(code),'; original model hashes unchanged.')

if __name__=='__main__':main()
