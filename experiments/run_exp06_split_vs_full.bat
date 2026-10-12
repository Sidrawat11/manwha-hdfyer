@echo off
echo Running split-vs-full experiment on the GPU (about 1 hour). Safe to stop and rerun: it resumes.
wsl.exe --cd /mnt/d/ManwhaHDFyer -- bash -lc "source venv/bin/activate && python experiments/exp06_split_vs_full.py 2>&1 | tee -a experiments/out/exp06_split_vs_full/gpu_run.log"
pause
