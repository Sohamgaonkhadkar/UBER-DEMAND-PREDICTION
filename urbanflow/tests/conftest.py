import pathlib,sys
BASE=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'backend'))
