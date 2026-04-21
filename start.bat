@echo off

cd /d "%~dp0"

call conda activate traffic_gnn
python run_gui.py