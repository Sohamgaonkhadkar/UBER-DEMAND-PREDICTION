"""Check preservation and freeze integrity without touching earlier artifacts."""
import json, sys, subprocess, shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'urbanflow'))
from ml.nextgen.protocol import ROOT, OUT, REPORT, sha

def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    freeze=json.loads((OUT/'selection_freeze.json').read_text())
    protected=[]
    for relative,expected in sorted(protocol['protected_sha256'].items()):
        p=ROOT/relative
        protected.append(dict(path=relative,exists=p.exists(),sha256_matches=p.exists() and sha(p)==expected))
    source=[]
    for relative,expected in freeze['code_sha256'].items():
        p=ROOT/relative;source.append(dict(path=relative,sha256_matches=sha(p)==expected))
    checks=dict(protected_files=len(protected),all_previous_files_preserved=all(r['sha256_matches'] for r in protected),
        frozen_source_unchanged=all(r['sha256_matches'] for r in source),
        validation_unchanged_since_selection=sha(OUT/'validation_results.json')==freeze['validation_sha256'],
        previous_files=protected,frozen_source=source,
        scope='Local only; no commit, push, PR, deploy or production default change was performed.')
    original=json.loads((ROOT/'reports/advanced_20261102/preservation_checks.json').read_text())['original_model_hashes']
    checks['original_model_hashes_unchanged']={relative:sha(ROOT/relative)==expected for relative,expected in original.items()}
    (REPORT/'preservation_checks.json').write_text(json.dumps(checks,indent=2))
    snapshot=OUT/'frozen_source';snapshot.mkdir(exist_ok=True)
    for relative in freeze['code_sha256']:
        dest=snapshot/Path(relative).name
        if not dest.exists():shutil.copyfile(ROOT/relative,dest)
        if sha(dest)!=freeze['code_sha256'][relative]:raise AssertionError('Frozen source snapshot mismatch')
    inventory=dict(new_sources=[str(p.relative_to(ROOT)) for p in (ROOT/'urbanflow/ml/nextgen').glob('*') if p.is_file()],
        new_test='urbanflow/tests/test_nextgen.py',new_root_helpers=['nextgen_system_check.py','nextgen_report.py','nextgen_evidence.py'],
        report='reports/next_generation_model_research.md',artifacts='artifacts/nextgen_20261102',evidence='reports/nextgen_20261102')
    for name,path in [('root',ROOT),('urbanflow',ROOT/'urbanflow')]:
        result=subprocess.run(['git','status','--short'],cwd=path,text=True,capture_output=True)
        inventory[name+'_git_status']=result.stdout
    (REPORT/'changed_files.json').write_text(json.dumps(inventory,indent=2))
    print(json.dumps({k:v for k,v in checks.items() if k not in ['previous_files','frozen_source']},indent=2))
    assert checks['all_previous_files_preserved'] and checks['frozen_source_unchanged'] and checks['validation_unchanged_since_selection'] and all(checks['original_model_hashes_unchanged'].values())

if __name__=='__main__':main()
