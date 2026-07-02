# cm-benchmark
It constructs a benchmark for evaluating cognitive map cognition in VLMs


This script is for processing and creating spatial descriptions of the csv files

```bash
python -m spatial_description_generator \
 --csv_path_navigation filename \
 --csv_path_objects filename \
 --json_path filename \
 --json_filename filename
```
# --episode_key string

Usage example on src/cm-benchmark/utils:

```bash
python -m spatial_description_generator \
 --csv_path_navigation /home/andreina/Documents/Programs/Dataset/Generated/navigation/05_06_2026_17_02_54_768901/navigation-Procedural.csv \
 --csv_path_objects /home/andreina/Documents/Programs/Dataset/Generated/navigation/05_06_2026_17_02_54_768901/objects-Procedural.csv \
--output_path /home/andreina/Documents/Repo/cm-benchmark/src/cm-benchmark/storage/ai2thor/nav_data \
--output_filename nav_data1.json
```

